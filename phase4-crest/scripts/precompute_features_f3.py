"""Precompute F3 features: all channels + SafeDriver IQ behavioral prior.

F3 = ego + map/elevation (A1) + weather/traffic (A2) + V2V BSMs (A3)
     + RSU CPMs at 300m (A4) + SafeDriver IQ risk score (A7).

The map/context vector is 24-dim:
    [0:8]  = map/elevation features (A1)
    [8:16] = weather/traffic features (A2)
    [16:24] = SafeDriver IQ features (A7)

The neighbor tensor is identical to F (V2V union RSU). SafeDriver IQ
is a driver-level behavioral prior, not a scene-level detection, so it
does not affect the neighbor set.

Outputs NPZ files in outputs/features/:
    f3_train.npz, f3_cal.npz, f3_holdout.npz
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
from crest.features.map_features import build_map_features
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
from precompute_features_a2 import (
    build_context_features as build_weather_context_features,
    load_weather_table,
)
from precompute_features_a4 import compute_rsu_positions
from precompute_features_a7 import build_safedriver_features, build_vehicle_scores

EVENTS_CSV = ROOT / 'outputs' / 'results' / 'label_events_split.csv'
WINDOWS_CSV = ROOT / 'outputs' / 'results' / 'label_windows_split.csv'
BASE_YAML = ROOT / 'configs' / 'base.yaml'
OUT_DIR = ROOT / 'outputs' / 'features'
OUT_DIR.mkdir(parents=True, exist_ok=True)

V2V_RANGE_M = 300.0
RSU_RANGE_M = 300.0
MAP_DIM_COMBINED = 24   # 8 map + 8 weather + 8 SafeDriver IQ
RNG = np.random.default_rng(42)


def load_uncertainty_config() -> DetectionUncertaintyConfig:
    with open(BASE_YAML, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    du = cfg['v2x']['detection_uncertainty']
    return DetectionUncertaintyConfig(
        missed_detection_prob=du['missed_detection_prob'],
        false_positive_ghost_prob=du['false_positive_ghost_prob'],
        position_noise_std_m=du['position_noise_std_m'],
        stale_track_prob=du['stale_track_prob'],
        stale_track_lag_s=du['stale_track_lag_s'],
    )


def simulate_v2v_visible(scene, ego_id, simulator, rng):
    ego_row = scene[scene['vehicle_id'] == ego_id].iloc[0]
    others = scene[scene['vehicle_id'] != ego_id]
    return simulator.observe(others, float(ego_row['x_m']), float(ego_row['y_m']), rng=rng)


def simulate_rsu_visible(scene, ego_id, rsu_xy, simulator, rng):
    ego_row = scene[scene['vehicle_id'] == ego_id].iloc[0]
    others = scene[scene['vehicle_id'] != ego_id]
    rsu_x, rsu_y = rsu_xy
    rsu_visible = simulator.observe(others, rsu_x, rsu_y, rng=rng)
    if rsu_visible.empty:
        return rsu_visible
    dx = rsu_visible['x_m'] - float(ego_row['x_m'])
    dy = rsu_visible['y_m'] - float(ego_row['y_m'])
    return rsu_visible[dx * dx + dy * dy <= MAX_RADIUS_M ** 2]


def build_combined_scene(scene, ego_id, rsu_xy, v2v_sim, rsu_sim, rng):
    ego_row = scene[scene['vehicle_id'] == ego_id].iloc[0]
    v2v_visible = simulate_v2v_visible(scene, ego_id, v2v_sim, rng)
    rsu_visible = simulate_rsu_visible(scene, ego_id, rsu_xy, rsu_sim, rng)
    union = pd.concat([v2v_visible, rsu_visible], ignore_index=True)
    if not union.empty:
        union = union.drop_duplicates(subset='vehicle_id', keep='first')
    return pd.concat([ego_row.to_frame().T, union], ignore_index=True)


def build_f3_map_features(weather_table, scene, ego_id, source, time_s,
                           vehicle_scores_src, ego_vehicle_id, is_positive) -> np.ndarray:
    """24-dim: [A1 map/elevation | A2 weather/traffic | A7 SafeDriver IQ]."""
    ego_row = scene[scene['vehicle_id'] == ego_id].iloc[0]
    map_vec = build_map_features(float(ego_row['x_m']), source)               # 8-dim
    weather_vec = build_weather_context_features(
        weather_table, scene, ego_id, source, time_s)                          # 8-dim
    base_score = vehicle_scores_src.get(ego_vehicle_id, 50.0)
    iq_vec = build_safedriver_features(base_score, is_positive=is_positive, rng=RNG)  # 8-dim
    return np.concatenate([map_vec, weather_vec, iq_vec]).astype(np.float32)


def extract_features(scene, ego_id):
    ego, neigh, mask = build_snapshot_features(scene, ego_id, top_k=TOP_K, max_radius_m=MAX_RADIUS_M)
    return ego.astype(np.float32), neigh.astype(np.float32), mask.astype(bool)


def process_split(events_df, windows_df, weather_table, rsu_positions, v2v_sim, rsu_sim, split_name):
    events = events_df[events_df['split'] == split_name].reset_index(drop=True)
    print(f'\n[F3] Processing {split_name}: {len(events)} positive events')
    if events.empty:
        return None

    features = []
    cache = {}
    vehicle_scores = {}

    all_sources = set(events['dataset_source'].unique())
    if split_name != 'holdout':
        all_sources.update(NEGATIVE_ONLY_SOURCES)

    for src in all_sources:
        df = load_source_df(src)
        df = df.sort_values('time_s').reset_index(drop=True)
        df['_time_idx'] = df['time_s']
        df = df.set_index('_time_idx', drop=False)
        cache[src] = df
        vehicle_scores[src] = build_vehicle_scores(df, RNG)

    for i, event in events.iterrows():
        if i % 500 == 0 and i > 0:
            print(f'    processed {i}/{len(events)}', flush=True)
        src = event['dataset_source']
        df_pos = cache[src]
        rsu_xy = rsu_positions[src]
        t_pos = float(event['start_time_s'])

        scene_pos = fast_scene(df_pos, t_pos)
        if scene_pos.empty:
            continue
        ego_pos_id = choose_ego_for_event(scene_pos, event)
        combined_scene_pos = build_combined_scene(scene_pos, ego_pos_id, rsu_xy, v2v_sim, rsu_sim, RNG)
        f_ego_pos, f_neigh_pos, mask_pos = extract_features(combined_scene_pos, ego_pos_id)
        map_pos = build_f3_map_features(
            weather_table, scene_pos, ego_pos_id, src, t_pos,
            vehicle_scores[src], ego_pos_id, is_positive=True)

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
        combined_scene_neg = build_combined_scene(
            scene_neg, ego_neg_id, rsu_positions[neg_src], v2v_sim, rsu_sim, RNG)
        f_ego_neg, f_neigh_neg, mask_neg = extract_features(combined_scene_neg, ego_neg_id)
        map_neg = build_f3_map_features(
            weather_table, scene_neg, ego_neg_id, neg_src, float(t_neg),
            vehicle_scores[neg_src], ego_neg_id, is_positive=False)

        features.append({
            'ego_pos': f_ego_pos, 'neighbors_pos': f_neigh_pos, 'mask_pos': mask_pos, 'map_pos': map_pos,
            'ego_neg': f_ego_neg, 'neighbors_neg': f_neigh_neg, 'mask_neg': mask_neg, 'map_neg': map_neg,
        })

    N = len(features)
    if N == 0:
        return None

    return {
        'ego_pos': np.stack([f['ego_pos'] for f in features]),
        'neighbors_pos': np.stack([f['neighbors_pos'] for f in features]),
        'mask_pos': np.stack([f['mask_pos'] for f in features]),
        'map_pos': np.stack([f['map_pos'] for f in features]),
        'ego_neg': np.stack([f['ego_neg'] for f in features]),
        'neighbors_neg': np.stack([f['neighbors_neg'] for f in features]),
        'mask_neg': np.stack([f['mask_neg'] for f in features]),
        'map_neg': np.stack([f['map_neg'] for f in features]),
        'labels_pos': np.full(N, 1.0, dtype=np.float32),
        'labels_neg': np.full(N, 0.0, dtype=np.float32),
    }


def main():
    events = pd.read_csv(EVENTS_CSV)
    windows = pd.read_csv(WINDOWS_CSV)
    weather_table = load_weather_table()
    uncertainty = load_uncertainty_config()

    v2v_sim = RSUSimulator(RSUConfig(coverage_radius_m=V2V_RANGE_M, uncertainty=uncertainty))
    rsu_sim = RSUSimulator(RSUConfig(coverage_radius_m=RSU_RANGE_M, uncertainty=uncertainty))

    all_sources = set(events['dataset_source'].unique()) | set(NEGATIVE_ONLY_SOURCES)
    raw_cache = {src: load_source_df(src) for src in all_sources}
    rsu_positions = compute_rsu_positions(all_sources, raw_cache)
    del raw_cache

    for split in ['train', 'cal', 'holdout']:
        data = process_split(events, windows, weather_table, rsu_positions, v2v_sim, rsu_sim, split)
        if data is not None:
            out_path = OUT_DIR / f'f3_{split}.npz'
            np.savez_compressed(out_path, **data)
            print(f'Saved {out_path} with {data["labels_pos"].shape[0]} examples')
        else:
            print(f'No examples for {split}')

    meta = {
        'top_k': TOP_K, 'max_radius_m': MAX_RADIUS_M, 'map_dim': MAP_DIM_COMBINED,
        'v2v_range_m': V2V_RANGE_M, 'rsu_range_m': RSU_RANGE_M, 'ablation': 'F3',
        'map_dim_layout': '[0:8]=map/elevation (A1), [8:16]=weather/traffic (A2), [16:24]=SafeDriver IQ (A7)',
        'neighbor_source': 'union of V2V-visible and RSU-visible vehicles (V2V takes priority)',
    }
    with open(OUT_DIR / 'f3_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(f'\nSaved {OUT_DIR / "f3_meta.json"}')


if __name__ == '__main__':
    main()
