"""Precompute CREST feature tensors from labeled events for fast training.

Outputs NPZ files in outputs/features/:
  train.npz, cal.npz, holdout.npz
Each contains:
  ego (N, 6), neighbors (N, K, 6), mask (N, K), map_features (N, 8), labels (N,)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from crest.datasets.crest_dataset import load_source_df
from crest.features.snapshot import build_snapshot_features, get_scene_at_time

EVENTS_CSV = ROOT / 'outputs' / 'results' / 'label_events_split.csv'
WINDOWS_CSV = ROOT / 'outputs' / 'results' / 'label_windows_split.csv'
OUT_DIR = ROOT / 'outputs' / 'features'
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAP_DIM = 8
NEGATIVE_ONLY_SOURCES = [f'mitra_t{i}' for i in range(1, 4)]
TOP_K = 10
MAX_RADIUS_M = 300.0
RNG = np.random.default_rng(42)


def positive_ranges(windows_df, source):
    src = windows_df[windows_df['dataset_source'] == source]
    return [(r['window_start_s'], r['window_end_s']) for _, r in src.iterrows()]


def fast_scene(df, t):
    idx = df.index.searchsorted(t)
    return df.iloc[max(0, idx - 1):idx + 2]


def sample_negative_time(df, ranges, rng=RNG):
    t_min = df['time_s'].min()
    t_max = df['time_s'].max()
    for _ in range(2000):
        t = rng.uniform(t_min, t_max)
        if all(not (r[0] <= t <= r[1]) for r in ranges):
            scene = fast_scene(df, t)
            if len(scene) > 0:
                return t, scene
    return None, None


def choose_ego_for_event(scene, event):
    lane = int(event['lane_id'])
    lane_scene = scene[scene['lane_id'] == lane]
    if lane_scene.empty:
        lane_scene = scene
    if event['type'] == 'queue':
        row = lane_scene.loc[lane_scene['speed_ms'].idxmin()]
    else:
        row = lane_scene.loc[lane_scene['lon_acc_ms2'].idxmin()]
    vid = row['vehicle_id']
    return int(vid.iloc[0] if hasattr(vid, 'iloc') else vid)


def choose_random_ego(scene, rng=RNG):
    return int(scene.iloc[rng.integers(0, len(scene))]['vehicle_id'])


def extract_features(scene, ego_id):
    ego, neigh, mask = build_snapshot_features(scene, ego_id, top_k=TOP_K, max_radius_m=MAX_RADIUS_M)
    return ego.astype(np.float32), neigh.astype(np.float32), mask.astype(bool)


def process_split(events_df, windows_df, split_name):
    events = events_df[events_df['split'] == split_name].reset_index(drop=True)
    print(f'\nProcessing {split_name}: {len(events)} positive events')
    if events.empty:
        return None

    features = []
    cache = {}
    sources = events['dataset_source'].unique()

    # load all sources referenced in this split (including negative-only fallbacks)
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

        # positive snapshot at event onset
        scene_pos = fast_scene(df_pos, float(event['start_time_s']))
        if scene_pos.empty:
            continue
        ego_pos = choose_ego_for_event(scene_pos, event)
        f_ego_pos, f_neigh_pos, mask_pos = extract_features(scene_pos, ego_pos)

        # negative snapshot
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
        ego_neg = choose_random_ego(scene_neg)
        f_ego_neg, f_neigh_neg, mask_neg = extract_features(scene_neg, ego_neg)

        features.append({
            'ego_pos': f_ego_pos,
            'neighbors_pos': f_neigh_pos,
            'mask_pos': mask_pos,
            'ego_neg': f_ego_neg,
            'neighbors_neg': f_neigh_neg,
            'mask_neg': mask_neg,
            'label_pos': 1.0,
            'label_neg': 0.0,
        })

    N = len(features)
    if N == 0:
        return None
    ego_pos = np.stack([f['ego_pos'] for f in features])
    neighbors_pos = np.stack([f['neighbors_pos'] for f in features])
    mask_pos = np.stack([f['mask_pos'] for f in features])
    ego_neg = np.stack([f['ego_neg'] for f in features])
    neighbors_neg = np.stack([f['neighbors_neg'] for f in features])
    mask_neg = np.stack([f['mask_neg'] for f in features])
    labels_pos = np.full(N, 1.0, dtype=np.float32)
    labels_neg = np.full(N, 0.0, dtype=np.float32)

    return {
        'ego_pos': ego_pos,
        'neighbors_pos': neighbors_pos,
        'mask_pos': mask_pos,
        'ego_neg': ego_neg,
        'neighbors_neg': neighbors_neg,
        'mask_neg': mask_neg,
        'labels_pos': labels_pos,
        'labels_neg': labels_neg,
        'map_features': np.zeros((N, MAP_DIM), dtype=np.float32),
    }


def main():
    events = pd.read_csv(EVENTS_CSV)
    windows = pd.read_csv(WINDOWS_CSV)
    for split in ['train', 'cal', 'holdout']:
        data = process_split(events, windows, split)
        if data is not None:
            out_path = OUT_DIR / f'{split}.npz'
            np.savez_compressed(out_path, **data)
            print(f'Saved {out_path} with {data["labels_pos"].shape[0]} examples')
        else:
            print(f'No examples for {split}')

    meta = {
        'map_dim': MAP_DIM,
        'top_k': TOP_K,
        'max_radius_m': MAX_RADIUS_M,
        'negative_only_sources': NEGATIVE_ONLY_SOURCES,
    }
    with open(OUT_DIR / 'meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    main()
