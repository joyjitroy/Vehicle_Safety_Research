"""Precompute F_opt features: A2 (weather/traffic) + A4_r500 (RSU CPMs, 500m only).

F_opt drops A1 (map/elevation), A3 (V2V BSMs), and keeps only the two
best-performing single-source configs to test whether targeted fusion
outperforms the full F model (0.700 AUPRC).

Architecture:
    map_input_dim = 8  (A2 weather/traffic only — no A1 elevation, no concat)
    neighbor set  = RSU at 500m only (no V2V union)

Why this combination:
    A2  (0.751 AUPRC) — highest individual gain; weather/traffic density
        signals are strongly predictive of congestion onset.
    A4_r500 (0.722 AUPRC) — highest RSU result; monotonic gain with radius
        means the 500m radius captures the widest useful infrastructure view.
    Hypothesis: removing A1 noise and A3 simulation noise will allow the
    model to learn cleaner fusion weights and exceed F (0.700).

Outputs NPZ files in outputs/features/:
    fopt_train.npz, fopt_cal.npz, fopt_holdout.npz
    (map_pos/map_neg are 8-dim: weather/traffic context from A2)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from crest.datasets.crest_dataset import load_source_df
from crest.features.snapshot import build_snapshot_features
from crest.features.v2x_sim import RSUConfig, RSUSimulator

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
from precompute_features_a2 import (
    build_context_features as build_weather_context,
    load_weather_table,
)
from precompute_features_a4 import (
    compute_rsu_positions,
    simulate_rsu_scene,
)

EVENTS_CSV  = ROOT / 'outputs' / 'results' / 'label_events_split.csv'
WINDOWS_CSV = ROOT / 'outputs' / 'results' / 'label_windows_split.csv'
OUT_DIR     = ROOT / 'outputs' / 'features'
OUT_DIR.mkdir(parents=True, exist_ok=True)

RSU_RANGE_M = 500.0   # A4_r500 — fixed, not a sweep
RNG         = np.random.default_rng(42)
MAP_DIM     = 8       # A2 weather/traffic only


def extract_features(scene, ego_id):
    ego, neigh, mask = build_snapshot_features(
        scene, ego_id, top_k=TOP_K, max_radius_m=MAX_RADIUS_M
    )
    return ego.astype(np.float32), neigh.astype(np.float32), mask.astype(bool)


def process_split(events_df, windows_df, simulator, rsu_positions,
                  weather_table, split_name):
    events = events_df[events_df['split'] == split_name].reset_index(drop=True)
    print(f'\n[F_opt] Processing {split_name}: {len(events)} positive events')
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

        src    = event['dataset_source']
        df_pos = cache[src]
        rsu_xy = rsu_positions[src]

        # Positive snapshot
        scene_pos = fast_scene(df_pos, float(event['start_time_s']))
        if scene_pos.empty:
            continue
        ego_pos_id  = choose_ego_for_event(scene_pos, event)
        rsu_pos     = simulate_rsu_scene(scene_pos, ego_pos_id, rsu_xy, simulator, RNG)
        f_ego_pos, f_neigh_pos, mask_pos = extract_features(rsu_pos, ego_pos_id)
        map_pos = build_weather_context(
            weather_table, scene_pos, ego_pos_id, src,
            float(event['start_time_s'])
        ).astype(np.float32)

        # Negative snapshot
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
            print(f'Warning: no negative sample for event {i}; skipping')
            continue
        ego_neg_id  = choose_random_ego(scene_neg)
        rsu_neg     = simulate_rsu_scene(
            scene_neg, ego_neg_id, rsu_positions[neg_src], simulator, RNG
        )
        f_ego_neg, f_neigh_neg, mask_neg = extract_features(rsu_neg, ego_neg_id)
        map_neg = build_weather_context(
            weather_table, scene_neg, ego_neg_id, neg_src, float(t_neg)
        ).astype(np.float32)

        features.append({
            'ego_pos': f_ego_pos, 'neighbors_pos': f_neigh_pos, 'mask_pos': mask_pos,
            'map_pos': map_pos,
            'ego_neg': f_ego_neg, 'neighbors_neg': f_neigh_neg, 'mask_neg': mask_neg,
            'map_neg': map_neg,
        })

    N = len(features)
    if N == 0:
        return None

    return {
        'ego_pos':       np.stack([f['ego_pos']       for f in features]),
        'neighbors_pos': np.stack([f['neighbors_pos'] for f in features]),
        'mask_pos':      np.stack([f['mask_pos']      for f in features]),
        'map_pos':       np.stack([f['map_pos']       for f in features]),
        'ego_neg':       np.stack([f['ego_neg']       for f in features]),
        'neighbors_neg': np.stack([f['neighbors_neg'] for f in features]),
        'mask_neg':      np.stack([f['mask_neg']      for f in features]),
        'map_neg':       np.stack([f['map_neg']       for f in features]),
        'labels_pos':    np.full(N, 1.0, dtype=np.float32),
        'labels_neg':    np.full(N, 0.0, dtype=np.float32),
    }


def main():
    events  = pd.read_csv(EVENTS_CSV)
    windows = pd.read_csv(WINDOWS_CSV)

    simulator = RSUSimulator(RSUConfig(coverage_radius_m=RSU_RANGE_M))
    weather_tbl = load_weather_table()

    all_sources  = set(events['dataset_source'].unique()) | set(NEGATIVE_ONLY_SOURCES)
    print('Computing RSU positions ...')
    temp_cache = {}
    for src in all_sources:
        df = load_source_df(src)
        temp_cache[src] = df
    rsu_positions = compute_rsu_positions(all_sources, temp_cache)
    del temp_cache

    for split in ('train', 'cal', 'holdout'):
        result = process_split(
            events, windows, simulator, rsu_positions, weather_tbl, split
        )
        if result is None:
            print(f'  Skipped {split} (no events)')
            continue
        out_path = OUT_DIR / f'fopt_{split}.npz'
        np.savez(out_path, **result)
        print(f'  Saved {out_path}  ({result["ego_pos"].shape[0]} pairs)')


if __name__ == '__main__':
    main()
