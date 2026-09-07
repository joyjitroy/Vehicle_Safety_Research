#!/usr/bin/env python3
"""
I-24 MOTION Holdout Inference
Converts filtered I-24 MOTION trajectory JSON files to CREST feature snapshots,
runs the frozen baseline (B1 ego-kinematic) model, and reports AUPRC.

Run from the project root:
    python scripts/infer_i24_holdout.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from crest.models.architecture import CRESTConfig, CRESTModel
from crest.features.snapshot import build_snapshot_features

import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================

FILTERED_DIR = ROOT / 'data' / 'processed' / 'i24_filtered'
CKPT_PATH    = ROOT / 'outputs' / 'checkpoints' / 'baseline.pt'
MODEL_YAML   = ROOT / 'configs' / 'model.yaml'
OUT_PATH     = ROOT / 'outputs' / 'results' / 'i24_holdout_eval.json'
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

FT_TO_M      = 0.3048        # I-24 MOTION positions are in feet
SNAP_INTERVAL_S = 2.0        # one snapshot every 2 seconds
TOP_K        = 10            # neighbours per snapshot
MAX_RADIUS_M = 300.0         # neighbour search radius
MAP_DIM      = 8             # map features (zeros, same as eval_holdout)
BATCH_SIZE   = 256
SEED         = 42

RNG = np.random.default_rng(SEED)

FILES = {
    'hazard': [
        'nov21_post1_hazard.json',
        'nov23_post3_hazard.json',
        'dec02_post12_hazard.json',
    ],
    'clean': [
        'nov21_post1_clean.json',
        'nov23_post3_clean.json',
        'dec02_post12_clean.json',
    ],
}

# =============================================================================
# STEP 1 — Convert trajectory records to canonical per-frame DataFrame
# =============================================================================

def trajectories_to_df(records: list) -> pd.DataFrame:
    """Expand per-vehicle trajectory arrays into one row per timestamp."""
    rows = []
    for rec in records:
        ts  = rec['timestamp']
        xs  = rec['x']
        ys  = rec['y']
        vid = rec['id']
        n   = len(ts)
        if n < 3:
            continue

        # Convert positions from feet to metres
        x_m = np.array(xs, dtype=np.float64) * FT_TO_M
        y_m = np.array(ys, dtype=np.float64) * FT_TO_M
        t   = np.array(ts, dtype=np.float64)

        # Speed and longitudinal acceleration via finite differences
        dt       = np.diff(t)
        dt       = np.where(dt < 1e-6, 1e-6, dt)   # guard div-by-zero
        dx       = np.diff(x_m)
        speed    = dx / dt                           # m/s (signed)
        acc      = np.diff(speed) / dt[:-1]          # m/s²

        # Align arrays to interior indices (lose first and last point)
        n_inner = n - 2
        if n_inner < 1:
            continue

        rows.append(pd.DataFrame({
            'vehicle_id': vid,
            'time_s':     t[1:-1],
            'x_m':        x_m[1:-1],
            'y_m':        y_m[1:-1],
            'speed_ms':   speed[:-1],
            'lon_acc_ms2': acc,
            'lat_acc_ms2': np.nan,
        }))

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# =============================================================================
# STEP 2 — Build snapshot features from a scene DataFrame
# =============================================================================

def scenes_from_df(df: pd.DataFrame) -> list:
    """
    Bucket rows into 2-second intervals and return one scene DataFrame per bucket.
    Ego is chosen as the slowest vehicle (queue proxy for hazard scenes).
    """
    if df.empty:
        return []

    t_min = df['time_s'].min()
    t_max = df['time_s'].max()
    snaps = []
    t = t_min
    while t <= t_max:
        scene = df[(df['time_s'] >= t - 0.5) & (df['time_s'] < t + 0.5)].copy()
        if len(scene) >= 2:
            # deduplicate: keep one row per vehicle (nearest to t)
            scene = scene.copy()
            scene['dt'] = (scene['time_s'] - t).abs()
            scene = scene.sort_values('dt').drop_duplicates('vehicle_id').drop(columns='dt')
            if len(scene) >= 2:
                snaps.append(scene)
        t += SNAP_INTERVAL_S
    return snaps


def extract_features_from_scenes(scenes: list, choose_slowest: bool) -> tuple:
    """Build (ego, neighbors, mask) arrays from a list of scene DataFrames."""
    egos, neighs, masks = [], [], []
    for scene in scenes:
        if choose_slowest:
            ego_id = scene.loc[scene['speed_ms'].abs().idxmin(), 'vehicle_id']
        else:
            ego_id = scene.iloc[RNG.integers(0, len(scene))]['vehicle_id']
        try:
            e, n, m = build_snapshot_features(
                scene, ego_id, top_k=TOP_K, max_radius_m=MAX_RADIUS_M
            )
            egos.append(e)
            neighs.append(n)
            masks.append(m)
        except Exception:
            continue
    return egos, neighs, masks


# =============================================================================
# STEP 3 — Load model
# =============================================================================

def load_model():
    with open(MODEL_YAML, encoding='utf-8') as f:
        cfg_dict = yaml.safe_load(f)
    arch = cfg_dict['architecture']
    input_dim = len(arch['input_features'])
    cfg = CRESTConfig(
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
    model = CRESTModel(cfg)
    model.load_state_dict(
        torch.load(CKPT_PATH, map_location='cpu', weights_only=True)
    )
    model.eval()
    return model


# =============================================================================
# STEP 4 — Run inference in batches
# =============================================================================

def run_inference(model, ego_arr, neigh_arr, mask_arr):
    ego_t   = torch.from_numpy(ego_arr)
    neigh_t = torch.from_numpy(neigh_arr)
    mask_t  = torch.from_numpy(mask_arr)
    map_t   = torch.zeros(len(ego_arr), MAP_DIM, dtype=torch.float32)

    probs = []
    with torch.no_grad():
        for i in range(0, len(ego_arr), BATCH_SIZE):
            out = model(
                ego_t[i:i+BATCH_SIZE],
                neigh_t[i:i+BATCH_SIZE],
                mask_t[i:i+BATCH_SIZE],
                map_t[i:i+BATCH_SIZE],
            )
            probs.extend(out.cpu().numpy().tolist())
    return np.clip(np.array(probs), 1e-6, 1 - 1e-6)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print('Loading model ...')
    model = load_model()

    all_egos, all_neighs, all_masks, all_labels = [], [], [], []

    for label_val, fnames in [(1.0, FILES['hazard']), (0.0, FILES['clean'])]:
        label_name = 'hazard' if label_val == 1.0 else 'clean'
        for fname in fnames:
            fpath = FILTERED_DIR / fname
            if not fpath.exists():
                print(f'  MISSING: {fpath}')
                continue

            print(f'  Processing {fname} ({label_name}) ...')
            with open(fpath) as f:
                records = json.load(f)

            df = trajectories_to_df(records)
            if df.empty:
                print(f'    No rows after conversion — skipping')
                continue

            scenes = scenes_from_df(df)
            print(f'    {len(records):,} trajectories → {len(df):,} rows → {len(scenes)} scenes')

            egos, neighs, masks = extract_features_from_scenes(
                scenes, choose_slowest=(label_val == 1.0)
            )

            if not egos:
                print(f'    No valid snapshots — skipping')
                continue

            all_egos.extend(egos)
            all_neighs.extend(neighs)
            all_masks.extend(masks)
            all_labels.extend([label_val] * len(egos))

    if not all_egos:
        sys.exit('No features extracted. Check filtered JSON files.')

    ego_arr   = np.stack(all_egos).astype(np.float32)
    neigh_arr = np.stack(all_neighs).astype(np.float32)
    mask_arr  = np.stack(all_masks)
    labels    = np.array(all_labels, dtype=np.float32)

    print(f'\nTotal snapshots: {len(labels):,}  '
          f'(hazard={int(labels.sum())}, clean={int((labels==0).sum())})')

    print('Running inference ...')
    probs = run_inference(model, ego_arr, neigh_arr, mask_arr)

    auprc  = float(average_precision_score(labels, probs))
    brier  = float(brier_score_loss(labels, probs))

    results = {
        'dataset':       'I-24 MOTION INCEPTION v1.0',
        'model':         'CREST B1 (ego-kinematic baseline)',
        'checkpoint':    str(CKPT_PATH.name),
        'n_snapshots':   int(len(labels)),
        'n_hazard':      int(labels.sum()),
        'n_clean':       int((labels == 0).sum()),
        'auprc':         round(auprc, 4),
        'brier':         round(brier, 4),
        'mean_prob':     round(float(probs.mean()), 4),
    }

    with open(OUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    print('\n' + '='*50)
    print('I-24 MOTION HOLDOUT RESULTS')
    print('='*50)
    for k, v in results.items():
        print(f'  {k:<20s}: {v}')
    print(f'\nResults saved to: {OUT_PATH}')


if __name__ == '__main__':
    main()
