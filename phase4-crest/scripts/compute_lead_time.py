"""Compute lead time between model trigger and hazard onset at fixed FAR points.

For each holdout positive event, the model's risk score is sampled at
multiple timestamps inside the event's fixed 10s pre-onset window
(window_start_s = onset - 10, window_end_s = onset, per
scripts/build_labels.py build_windows()), tracking the SAME ego vehicle
identity chosen at onset (choose_ego_for_event) backward through the window
at the dataset's native sampling rate (NGSIM 10Hz, MiTra 30Hz, from
configs/base.yaml sampling).

OPERATING POINTS (false alarm rate thresholds):
    Thresholds for 1%, 5%, 10% FAR are derived from the CALIBRATION split's
    negative examples (outputs/features/cal.npz), NOT the holdout split, to
    avoid leaking holdout information into the definition of the operating
    points. FAR = P(score >= threshold | negative), so threshold = the
    (100 - FAR)th percentile of calibration negative scores.

TRIGGER AND LEAD TIME:
    Scanning forward in time from window_start_s to window_end_s (=onset),
    trigger_time is the first timestamp at which the model's score for the
    tracked ego vehicle exceeds the FAR threshold. If the ego vehicle is not
    present in the scene at a given timestamp (e.g. it entered the recorded
    segment partway through the window), that timestamp is skipped.

    lead_time_s = onset - trigger_time   (positive = warned before hazard)

    If the score never crosses the threshold within the window, the event is
    recorded as "missed" for that operating point (no lead time).

Uses the raw (uncalibrated) sigmoid model output for thresholding. Since
Platt scaling (outputs/checkpoints/platt.json) is a monotonic transform of
the logit, a percentile-based threshold on raw probabilities is equivalent
to the same percentile threshold in Platt-calibrated probability space; this
script uses raw probabilities directly for simplicity.

Output: outputs/results/lead_time.json
  {
    "far_thresholds": {"far_1pct": ..., "far_5pct": ..., "far_10pct": ...},
    "operating_points": {
      "far_1pct": {"n_detected": ..., "n_missed": ..., "detection_rate": ...,
                    "mean_lead_time_s": ..., "median_lead_time_s": ...},
      "far_5pct": {...}, "far_10pct": {...}
    },
    "events": [ {dataset_source, event_onset_s, far_1pct_lead_s, far_5pct_lead_s,
                 far_10pct_lead_s}, ... ]
  }
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from crest.datasets.crest_dataset import load_source_df
from crest.features.snapshot import build_snapshot_features
from crest.models.architecture import CRESTConfig, CRESTModel

from precompute_features import (
    MAX_RADIUS_M,
    TOP_K,
    choose_ego_for_event,
    fast_scene,
)

EVENTS_CSV = ROOT / 'outputs' / 'results' / 'label_events_split.csv'
WINDOWS_CSV = ROOT / 'outputs' / 'results' / 'label_windows_split.csv'
BASE_YAML = ROOT / 'configs' / 'base.yaml'
MODEL_YAML = ROOT / 'configs' / 'model.yaml'
CKPT_PATH = ROOT / 'outputs' / 'checkpoints' / 'baseline.pt'
CAL_FEATURES_PATH = ROOT / 'outputs' / 'features' / 'cal.npz'
OUT_PATH = ROOT / 'outputs' / 'results' / 'lead_time.json'
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

MAP_DIM = 8
FAR_LEVELS = {'far_1pct': 0.01, 'far_5pct': 0.05, 'far_10pct': 0.10}


def load_model_config(path: Path) -> CRESTConfig:
    with open(path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    arch = cfg['architecture']
    input_dim = len(arch['input_features'])
    return CRESTConfig(
        ego_input_dim=input_dim,
        ego_hidden_dims=arch['ego_encoder']['hidden_dims'],
        ego_output_dim=arch['ego_encoder']['output_dim'],
        neighbor_input_dim=input_dim,
        neighbor_hidden_dims=arch['neighbor_encoder']['hidden_dims'],
        neighbor_output_dim=arch['neighbor_encoder']['output_dim'],
        map_input_dim=arch['map_encoder']['input_dim'],
        map_hidden_dims=arch['map_encoder']['hidden_dims'],
        map_output_dim=arch['map_encoder']['output_dim'],
        fusion_hidden_dims=arch['fusion']['hidden_dims'],
        dropout=arch.get('dropout', 0.1),
    )


def load_model(device):
    cfg = load_model_config(MODEL_YAML)
    model = CRESTModel(cfg).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device, weights_only=True))
    model.eval()
    return model


def compute_far_thresholds(model, device) -> dict:
    """Derive FAR thresholds from the calibration split's negative examples."""
    data = np.load(CAL_FEATURES_PATH)
    ego_neg = torch.from_numpy(data['ego_neg'])
    neighbors_neg = torch.from_numpy(data['neighbors_neg'])
    mask_neg = torch.from_numpy(data['mask_neg'])
    map_neg = torch.zeros((len(ego_neg), MAP_DIM), dtype=torch.float32)

    with torch.no_grad():
        scores = model(ego_neg.to(device), neighbors_neg.to(device), mask_neg.to(device), map_neg.to(device))
    scores = scores.cpu().numpy()

    thresholds = {}
    for name, far in FAR_LEVELS.items():
        thresholds[name] = float(np.quantile(scores, 1.0 - far))
    return thresholds


def score_ego_at_time(model, device, df, ego_id, t):
    scene = fast_scene(df, t)
    if scene.empty or ego_id not in scene['vehicle_id'].values:
        return None
    ego, neigh, mask = build_snapshot_features(scene, ego_id, top_k=TOP_K, max_radius_m=MAX_RADIUS_M)
    ego_t = torch.from_numpy(ego.astype(np.float32)).unsqueeze(0)
    neigh_t = torch.from_numpy(neigh.astype(np.float32)).unsqueeze(0)
    mask_t = torch.from_numpy(mask.astype(bool)).unsqueeze(0)
    map_t = torch.zeros((1, MAP_DIM), dtype=torch.float32)
    with torch.no_grad():
        score = model(ego_t.to(device), neigh_t.to(device), mask_t.to(device), map_t.to(device))
    return float(score.item())


def compute_lead_time_for_event(model, device, df, ego_id, window_start_s, onset_s, step_s, thresholds):
    """Scan forward from window_start_s to onset_s; return first-crossing lead time per FAR level."""
    result = {name: None for name in thresholds}
    triggered = {name: False for name in thresholds}

    t = window_start_s
    while t <= onset_s + 1e-9:
        score = score_ego_at_time(model, device, df, ego_id, t)
        if score is not None:
            for name, thresh in thresholds.items():
                if not triggered[name] and score >= thresh:
                    result[name] = float(onset_s - t)
                    triggered[name] = True
        if all(triggered.values()):
            break
        t += step_s

    return result


def main():
    device = torch.device('cpu')
    model = load_model(device)
    thresholds = compute_far_thresholds(model, device)
    print(f'FAR thresholds (from calibration negatives): {thresholds}')

    with open(BASE_YAML, encoding='utf-8') as f:
        base_cfg = yaml.safe_load(f)

    events_df = pd.read_csv(EVENTS_CSV)
    events = events_df[events_df['split'] == 'holdout'].reset_index(drop=True)
    print(f'Processing {len(events)} holdout events')

    cache = {}
    records = []

    for i, event in events.iterrows():
        src = event['dataset_source']
        if src not in cache:
            df = load_source_df(src)
            df = df.sort_values('time_s').reset_index(drop=True)
            df['_time_idx'] = df['time_s']
            cache[src] = df.set_index('_time_idx', drop=False)
        df = cache[src]

        step_s = 1.0 / (base_cfg['sampling']['mitra_hz'] if src.startswith('mitra') else base_cfg['sampling']['ngsim_hz'])

        onset_s = float(event['start_time_s'])
        window_start_s = onset_s - base_cfg['prediction']['horizon_s']

        scene_onset = fast_scene(df, onset_s)
        if scene_onset.empty:
            continue
        ego_id = choose_ego_for_event(scene_onset, event)

        lead_times = compute_lead_time_for_event(model, device, df, ego_id, window_start_s, onset_s, step_s, thresholds)

        record = {'dataset_source': src, 'event_onset_s': onset_s}
        record.update({f'{name}_lead_s': lead_times[name] for name in thresholds})
        records.append(record)

        if i % 10 == 0:
            print(f'  processed {i}/{len(events)}', flush=True)

    operating_points = {}
    for name in thresholds:
        lead_values = [r[f'{name}_lead_s'] for r in records if r[f'{name}_lead_s'] is not None]
        n_detected = len(lead_values)
        n_missed = len(records) - n_detected
        operating_points[name] = {
            'n_detected': n_detected,
            'n_missed': n_missed,
            'detection_rate': float(n_detected / len(records)) if records else 0.0,
            'mean_lead_time_s': float(np.mean(lead_values)) if lead_values else None,
            'median_lead_time_s': float(np.median(lead_values)) if lead_values else None,
        }

    output = {
        'far_thresholds': thresholds,
        'operating_points': operating_points,
        'events': records,
    }
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    print(json.dumps(operating_points, indent=2))
    print(f'Saved {OUT_PATH}')


if __name__ == '__main__':
    main()
