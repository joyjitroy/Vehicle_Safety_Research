"""Precompute A1 ablation features: ego time-series + synthetic map/elevation.

Reuses the exact same event/window selection, ego snapshot extraction, and
negative sampling as scripts/precompute_features.py (B1), so that ego and
neighbor tensors are identical between B1 and A1. The only addition is a
non-zero 8-dim map/elevation feature vector per snapshot (see
src/crest/features/map_features.py), isolating the effect of adding
map/elevation context.

Outputs NPZ files in outputs/features/:
  a1_train.npz, a1_cal.npz, a1_holdout.npz
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from crest.datasets.crest_dataset import load_source_df
from crest.features.map_features import build_map_features
from crest.features.snapshot import build_snapshot_features

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
OUT_DIR = ROOT / 'outputs' / 'features'
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAP_DIM = 8


def extract_features(scene, ego_id):
    ego, neigh, mask = build_snapshot_features(scene, ego_id, top_k=TOP_K, max_radius_m=MAX_RADIUS_M)
    return ego.astype(np.float32), neigh.astype(np.float32), mask.astype(bool)


def ego_x_m(scene, ego_id) -> float:
    row = scene[scene['vehicle_id'] == ego_id].iloc[0]
    return float(row['x_m'])


def process_split(events_df, windows_df, split_name):
    events = events_df[events_df['split'] == split_name].reset_index(drop=True)
    print(f'\n[A1] Processing {split_name}: {len(events)} positive events')
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
        f_ego_pos, f_neigh_pos, mask_pos = extract_features(scene_pos, ego_pos_id)
        map_pos = build_map_features(ego_x_m(scene_pos, ego_pos_id), src)

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
        f_ego_neg, f_neigh_neg, mask_neg = extract_features(scene_neg, ego_neg_id)
        map_neg = build_map_features(ego_x_m(scene_neg, ego_neg_id), neg_src)

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
    for split in ['train', 'cal', 'holdout']:
        data = process_split(events, windows, split)
        if data is not None:
            out_path = OUT_DIR / f'a1_{split}.npz'
            np.savez_compressed(out_path, **data)
            print(f'Saved {out_path} with {data["labels_pos"].shape[0]} examples')
        else:
            print(f'No examples for {split}')

    meta = {'map_dim': MAP_DIM, 'top_k': TOP_K, 'max_radius_m': MAX_RADIUS_M, 'ablation': 'A1'}
    with open(OUT_DIR / 'a1_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    main()
