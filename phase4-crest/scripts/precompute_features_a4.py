"""Precompute A4 ablation features: ego time-series + simulated RSU CPMs.

Unlike A3 (V2V BSMs, naturally centered on the ego vehicle since it is one
communication endpoint), a Roadside Unit (RSU) Collective Perception Message
(CPM) comes from a FIXED infrastructure sensor. Coverage is therefore
centered on a fixed RSU position per corridor (dataset_source), not on ego.

RSU PLACEMENT:
    No RSU placement is published with NGSIM/MiTra (no real RSUs exist in
    these datasets). We place one RSU at the median (x_m, y_m) position of
    all vehicles observed in that dataset_source, approximating a roadside
    sensor near the middle of the recorded corridor. This is a documented
    simulation assumption, not real deployed hardware.

RADIUS SWEEP:
    Runs the frozen sweep from configs/base.yaml (v2x.rsu.coverage_radii_m =
    [150, 300, 500]). For each radius, a vehicle must satisfy BOTH:
      (a) be within RSU coverage_radius_m of the fixed RSU position, AND
      (b) be within MAX_RADIUS_M (300m, same as B1/A1-A3) of the ego vehicle,
    to be included as neighbor input -- a vehicle RSU can see far from ego is
    not relevant to ego's immediate hazard prediction.

DETECTION UNCERTAINTY:
    Same frozen defaults as A3 (configs/base.yaml v2x.detection_uncertainty),
    applied via RSUSimulator.observe() relative to the RSU position.

Ego and negative-sampling logic identical to scripts/precompute_features.py.
map_encoder input slot remains zero (A4 only modifies the neighbor tensor).

Outputs NPZ files in outputs/features/, one set per radius:
  a4_r150_{train,cal,holdout}.npz
  a4_r300_{train,cal,holdout}.npz
  a4_r500_{train,cal,holdout}.npz
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

RNG = np.random.default_rng(42)


def load_base_config():
    with open(BASE_YAML, encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_uncertainty_config(cfg) -> DetectionUncertaintyConfig:
    du = cfg['v2x']['detection_uncertainty']
    return DetectionUncertaintyConfig(
        missed_detection_prob=du['missed_detection_prob'],
        false_positive_ghost_prob=du['false_positive_ghost_prob'],
        position_noise_std_m=du['position_noise_std_m'],
        stale_track_prob=du['stale_track_prob'],
        stale_track_lag_s=du['stale_track_lag_s'],
    )


def compute_rsu_positions(sources, cache):
    """One fixed RSU per dataset_source, placed at the median vehicle position."""
    positions = {}
    for src in sources:
        df = cache[src]
        positions[src] = (float(df['x_m'].median()), float(df['y_m'].median()))
    return positions


def simulate_rsu_scene(scene: pd.DataFrame, ego_id: int, rsu_xy, simulator: RSUSimulator, rng) -> pd.DataFrame:
    """Return a scene where non-ego vehicles are replaced by RSU-observed detections
    that are also within MAX_RADIUS_M of ego."""
    ego_row = scene[scene['vehicle_id'] == ego_id].iloc[0]
    others = scene[scene['vehicle_id'] != ego_id]

    rsu_x, rsu_y = rsu_xy
    rsu_visible = simulator.observe(others, rsu_x, rsu_y, rng=rng)

    if rsu_visible.empty:
        return ego_row.to_frame().T

    dx = rsu_visible['x_m'] - float(ego_row['x_m'])
    dy = rsu_visible['y_m'] - float(ego_row['y_m'])
    within_ego_range = rsu_visible[dx * dx + dy * dy <= MAX_RADIUS_M ** 2]

    return pd.concat([ego_row.to_frame().T, within_ego_range], ignore_index=True)


def extract_features(scene, ego_id):
    ego, neigh, mask = build_snapshot_features(scene, ego_id, top_k=TOP_K, max_radius_m=MAX_RADIUS_M)
    return ego.astype(np.float32), neigh.astype(np.float32), mask.astype(bool)


def process_split(events_df, windows_df, simulator, rsu_positions, split_name):
    events = events_df[events_df['split'] == split_name].reset_index(drop=True)
    print(f'\n[A4] Processing {split_name}: {len(events)} positive events')
    if events.empty:
        return None

    features = []
    cache = {}
    sources = events['dataset_source'].unique()

    all_sources = set(sources)
    if split_name != 'holdout':
        all_sources.update(NEGATIVE_ONLY_SOURCES)
    for src in all_sources:
        df = load_source_df(src)
        df = df.sort_values('time_s').reset_index(drop=True)
        df['_time_idx'] = df['time_s']
        cache[src] = df.set_index('_time_idx', drop=False)

    for i, event in events.iterrows():
        if i % 500 == 0 and i > 0:
            print(f'    processed {i}/{len(events)}', flush=True)
        src = event['dataset_source']
        df_pos = cache[src]
        rsu_xy = rsu_positions[src]

        scene_pos = fast_scene(df_pos, float(event['start_time_s']))
        if scene_pos.empty:
            continue
        ego_pos_id = choose_ego_for_event(scene_pos, event)
        rsu_scene_pos = simulate_rsu_scene(scene_pos, ego_pos_id, rsu_xy, simulator, RNG)
        f_ego_pos, f_neigh_pos, mask_pos = extract_features(rsu_scene_pos, ego_pos_id)

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
        rsu_scene_neg = simulate_rsu_scene(scene_neg, ego_neg_id, rsu_positions[neg_src], simulator, RNG)
        f_ego_neg, f_neigh_neg, mask_neg = extract_features(rsu_scene_neg, ego_neg_id)

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
    base_cfg = load_base_config()
    uncertainty = load_uncertainty_config(base_cfg)
    radii = base_cfg['v2x']['rsu']['coverage_radii_m']

    # Compute one fixed RSU position per source — load one source at a time to save memory.
    all_sources = set(events['dataset_source'].unique()) | set(NEGATIVE_ONLY_SOURCES)
    rsu_positions = {}
    for src in all_sources:
        print(f'  Computing RSU position for {src} ...', flush=True)
        df = load_source_df(src)
        rsu_positions[src] = (float(df['x_m'].median()), float(df['y_m'].median()))
        del df

    for radius_m in radii:
        print(f'\n=== RSU coverage radius: {radius_m}m ===')
        config = RSUConfig(coverage_radius_m=float(radius_m), uncertainty=uncertainty)
        simulator = RSUSimulator(config)

        for split in ['train', 'cal', 'holdout']:
            data = process_split(events, windows, simulator, rsu_positions, split)
            if data is not None:
                out_path = OUT_DIR / f'a4_r{radius_m}_{split}.npz'
                np.savez_compressed(out_path, **data)
                print(f'Saved {out_path} with {data["labels_pos"].shape[0]} examples')
            else:
                print(f'No examples for {split} at radius {radius_m}m')

    meta = {
        'top_k': TOP_K, 'max_radius_m': MAX_RADIUS_M, 'coverage_radii_m': radii, 'ablation': 'A4',
        'rsu_placement': 'median vehicle position per dataset_source (simulated, not real hardware)',
        'detection_uncertainty': {
            'missed_detection_prob': uncertainty.missed_detection_prob,
            'position_noise_std_m': uncertainty.position_noise_std_m,
            'stale_track_prob': uncertainty.stale_track_prob,
            'stale_track_lag_s': uncertainty.stale_track_lag_s,
        },
    }
    with open(OUT_DIR / 'a4_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    main()
