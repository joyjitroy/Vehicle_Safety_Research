"""Precompute A3 ablation features: ego time-series + simulated V2V BSMs.

Unlike A1/A2 (which populate the unused map_encoder input slot), A3 modifies
the NEIGHBOR tensor itself: instead of perfect ground-truth neighbor states
(as used in B1), neighbor observations are passed through
crest.features.v2x_sim.RSUSimulator to simulate realistic Basic Safety
Message (BSM) reception between the ego vehicle and nearby vehicles over a
V2V communication range.

V2V RANGE:
    configs/base.yaml does not freeze a V2V-specific range (only the RSU
    sweep radii [150, 300, 500] for A4). We use 300m, the middle of that
    frozen RSU sweep and a standard DSRC/C-V2X effective BSM range cited in
    the V2X literature. This choice is documented here and in the paper
    Methods section.

DETECTION UNCERTAINTY:
    Uses the frozen defaults from configs/base.yaml v2x.detection_uncertainty
    (missed_detection_prob=0.05, position_noise_std_m=1.0, stale_track_prob=0.05,
    stale_track_lag_s=0.2), applied via RSUSimulator.observe(). The ego
    vehicle's own self-state is assumed perfectly known (standard assumption;
    only OTHER vehicles' states are subject to BSM simulation).

Ego and negative-sampling logic is identical to scripts/precompute_features.py
(B1); only the neighbor tensor differs. The map_encoder input slot remains
zero, consistent with A1/A2 each isolating one input channel at a time.

Outputs NPZ files in outputs/features/:
  a3_train.npz, a3_cal.npz, a3_holdout.npz
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from crest.datasets.crest_dataset import load_source_df
from crest.features.snapshot import build_snapshot_features
from crest.features.v2x_sim import DetectionUncertaintyConfig, RSUConfig, RSUSimulator

from precompute_features import (
    MAX_RADIUS_M,
    NEGATIVE_ONLY_SOURCES,
    TOP_K,
    choose_ego_for_event,
    choose_random_ego,
    fast_scene,
    positive_ranges,
    sample_negative_time,
)

EVENTS_CSV = ROOT / 'outputs' / 'results' / 'label_events_split.csv'
WINDOWS_CSV = ROOT / 'outputs' / 'results' / 'label_windows_split.csv'
BASE_YAML = ROOT / 'configs' / 'base.yaml'
OUT_DIR = ROOT / 'outputs' / 'features'
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAP_DIM = 8
V2V_RANGE_M = 300.0  # documented assumption; see module docstring
RNG = np.random.default_rng(42)


def load_v2x_config() -> RSUConfig:
    with open(BASE_YAML, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    du = cfg['v2x']['detection_uncertainty']
    uncertainty = DetectionUncertaintyConfig(
        missed_detection_prob=du['missed_detection_prob'],
        false_positive_ghost_prob=du['false_positive_ghost_prob'],
        position_noise_std_m=du['position_noise_std_m'],
        stale_track_prob=du['stale_track_prob'],
        stale_track_lag_s=du['stale_track_lag_s'],
    )
    return RSUConfig(coverage_radius_m=V2V_RANGE_M, uncertainty=uncertainty)


def simulate_v2v_scene(scene: pd.DataFrame, ego_id: int, simulator: RSUSimulator, rng) -> pd.DataFrame:
    """Return a scene where non-ego vehicles are replaced by simulated BSM observations."""
    ego_row = scene[scene['vehicle_id'] == ego_id].iloc[0]
    others = scene[scene['vehicle_id'] != ego_id]
    observed_others = simulator.observe(others, float(ego_row['x_m']), float(ego_row['y_m']), rng=rng)
    return pd.concat([ego_row.to_frame().T, observed_others], ignore_index=True)


def extract_features(scene, ego_id):
    ego, neigh, mask = build_snapshot_features(scene, ego_id, top_k=TOP_K, max_radius_m=MAX_RADIUS_M)
    return ego.astype(np.float32), neigh.astype(np.float32), mask.astype(bool)


def process_split(events_df, windows_df, simulator, split_name):
    events = events_df[events_df['split'] == split_name].reset_index(drop=True)
    print(f'\n[A3] Processing {split_name}: {len(events)} positive events')
    if events.empty:
        return None

    features = []
    cache = {}
    sources = events['dataset_source'].unique()

    all_sources = set(sources)
    if split_name != 'holdout':
        all_sources.update(NEGATIVE_ONLY_SOURCES)
    for src in all_sources:
        print(f'  Loading {src} ...', flush=True)
        cache[src] = load_source_df(src)
        cache[src] = cache[src].sort_values('time_s').reset_index(drop=True)
        cache[src]['_time_idx'] = cache[src]['time_s']
        cache[src] = cache[src].set_index('_time_idx', drop=False)

    for i, event in events.iterrows():
        if i % 500 == 0 and i > 0:
            print(f'    processed {i}/{len(events)}', flush=True)
        src = event['dataset_source']
        df_pos = cache[src]

        scene_pos = fast_scene(df_pos, float(event['start_time_s']))
        if scene_pos.empty:
            continue
        ego_pos_id = choose_ego_for_event(scene_pos, event)
        v2v_scene_pos = simulate_v2v_scene(scene_pos, ego_pos_id, simulator, RNG)
        f_ego_pos, f_neigh_pos, mask_pos = extract_features(v2v_scene_pos, ego_pos_id)

        ranges = positive_ranges(windows_df, src)
        t_neg, scene_neg = sample_negative_time(df_pos, ranges)
        neg_src = src
        if t_neg is None:
            for fallback in NEGATIVE_ONLY_SOURCES:
                t_neg, scene_neg = sample_negative_time(cache[fallback], [])
                neg_src = fallback
                if t_neg is not None:
                    break
        if t_neg is None:
            print(f'Warning: could not sample negative for event {i}; skipping')
            continue
        ego_neg_id = choose_random_ego(scene_neg)
        v2v_scene_neg = simulate_v2v_scene(scene_neg, ego_neg_id, simulator, RNG)
        f_ego_neg, f_neigh_neg, mask_neg = extract_features(v2v_scene_neg, ego_neg_id)

        features.append({
            'ego_pos': f_ego_pos, 'neighbors_pos': f_neigh_pos, 'mask_pos': mask_pos,
            'ego_neg': f_ego_neg, 'neighbors_neg': f_neigh_neg, 'mask_neg': mask_neg,
        })

    N = len(features)
    if N == 0:
        return None

    return {
        'ego_pos': np.stack([f['ego_pos'] for f in features]),
        'neighbors_pos': np.stack([f['neighbors_pos'] for f in features]),
        'mask_pos': np.stack([f['mask_pos'] for f in features]),
        'ego_neg': np.stack([f['ego_neg'] for f in features]),
        'neighbors_neg': np.stack([f['neighbors_neg'] for f in features]),
        'mask_neg': np.stack([f['mask_neg'] for f in features]),
        'labels_pos': np.full(N, 1.0, dtype=np.float32),
        'labels_neg': np.full(N, 0.0, dtype=np.float32),
    }


def main():
    events = pd.read_csv(EVENTS_CSV)
    windows = pd.read_csv(WINDOWS_CSV)
    v2x_config = load_v2x_config()
    simulator = RSUSimulator(v2x_config)

    for split in ['train', 'cal', 'holdout']:
        data = process_split(events, windows, simulator, split)
        if data is not None:
            out_path = OUT_DIR / f'a3_{split}.npz'
            np.savez_compressed(out_path, **data)
            print(f'Saved {out_path} with {data["labels_pos"].shape[0]} examples')
        else:
            print(f'No examples for {split}')

    meta = {
        'top_k': TOP_K, 'v2v_range_m': V2V_RANGE_M, 'ablation': 'A3',
        'detection_uncertainty': {
            'missed_detection_prob': v2x_config.uncertainty.missed_detection_prob,
            'position_noise_std_m': v2x_config.uncertainty.position_noise_std_m,
            'stale_track_prob': v2x_config.uncertainty.stale_track_prob,
            'stale_track_lag_s': v2x_config.uncertainty.stale_track_lag_s,
        },
    }
    with open(OUT_DIR / 'a3_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    main()
