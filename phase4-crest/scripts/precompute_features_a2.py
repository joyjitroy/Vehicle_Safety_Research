"""Precompute A2 ablation features: ego time-series + traffic/weather context.

Reuses the exact same event/window selection, ego snapshot extraction, and
negative sampling as scripts/precompute_features.py (B1), so ego and neighbor
tensors are identical between B1 and A2. The addition is a non-zero 8-dim
"map_encoder" input vector per snapshot combining:
  - static per-source weather (from data/external/weather.json)
  - dynamic per-snapshot local traffic density/speed (from the raw scene)

Feature vector (8-dim), matching configs/model.yaml map_encoder.input_dim=8:
    0: temperature_c        - Open-Meteo temperature_2m at the matched hour
    1: precipitation_mm     - Open-Meteo precipitation at the matched hour
    2: visibility_km        - Open-Meteo visibility / 1000
    3: windspeed_kmh        - Open-Meteo windspeed_10m
    4: weathercode          - Open-Meteo WMO weathercode (raw numeric code)
    5: is_precipitating     - 1.0 if precipitation_mm > 0 else 0.0
    6: local_traffic_density - vehicles per 100m in the ego's lane at this
                                 snapshot (dynamic, computed from the scene)
    7: local_avg_speed_ms   - mean speed of vehicles in the ego's lane at
                                 this snapshot (dynamic, computed from the scene)

WEATHER TIME MATCHING:
    NGSIM time_s is a real Unix epoch (Global_Time / 1000), so the snapshot's
    hour-of-day is computed exactly and matched to the nearest hourly record
    in data/external/weather.json.

    MiTra Time [s] is relative to each session's recording start with no
    published real-world clock offset, so an exact hour cannot be recovered.
    We use a fixed representative hour (12:00, local noon) for all MiTra
    snapshots, matching the representative dates already used in
    data/external/mitra_dates.json. This is documented as a limitation,
    consistent with the MiTra date-approximation note in
    docs/mitra_dates_methods_note.md.

Outputs NPZ files in outputs/features/:
  a2_train.npz, a2_cal.npz, a2_holdout.npz
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from crest.datasets.crest_dataset import load_source_df
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
WEATHER_JSON = ROOT / 'data' / 'external' / 'weather.json'
OUT_DIR = ROOT / 'outputs' / 'features'
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAP_DIM = 8
MITRA_REPRESENTATIVE_HOUR = 12  # local noon; MiTra has no absolute clock offset
TRAFFIC_LANE_WINDOW_M = 100.0   # +/- window along the road for density/speed


def load_weather_table():
    if not WEATHER_JSON.exists():
        raise FileNotFoundError(
            f'{WEATHER_JSON} not found. Run scripts/fetch_weather.py first.'
        )
    with open(WEATHER_JSON, encoding='utf-8') as f:
        return json.load(f)


def _closest_hourly_record(records, target_iso_prefix):
    for rec in records:
        if rec['time'].startswith(target_iso_prefix):
            return rec
    return records[0] if records else None


def weather_for_snapshot(weather_table, source: str, time_s: float) -> dict:
    records = weather_table.get(source)
    if not records or (isinstance(records, dict) and 'error' in records):
        return {'temperature_2m': 0.0, 'precipitation': 0.0, 'visibility': 0.0,
                'windspeed_10m': 0.0, 'weathercode': 0.0}

    if source.startswith('ngsim'):
        dt = datetime.fromtimestamp(time_s, tz=timezone.utc)
        hour_prefix = dt.strftime('%Y-%m-%dT%H')
        rec = _closest_hourly_record(records, hour_prefix)
        return rec if rec is not None else {
            'temperature_2m': 0.0, 'precipitation': 0.0, 'visibility': 0.0,
            'windspeed_10m': 0.0, 'weathercode': 0.0,
        }

    # MiTra: no absolute clock; use representative noon hour.
    rec = next((r for r in records if r['time'][11:13] == f'{MITRA_REPRESENTATIVE_HOUR:02d}'), None)
    if rec is not None:
        return rec
    return records[len(records) // 2] if records else {
        'temperature_2m': 0.0, 'precipitation': 0.0, 'visibility': 0.0,
        'windspeed_10m': 0.0, 'weathercode': 0.0,
    }


def local_traffic_stats(scene: pd.DataFrame, ego_row: pd.Series):
    """Dynamic traffic density/speed in the ego's lane within a fixed window."""
    lane_scene = scene[scene['lane_id'] == ego_row['lane_id']]
    nearby = lane_scene[(lane_scene['y_m'] - ego_row['y_m']).abs() <= TRAFFIC_LANE_WINDOW_M]
    if nearby.empty:
        return 0.0, float(ego_row['speed_ms'])
    density = len(nearby) / (2 * TRAFFIC_LANE_WINDOW_M / 100.0)  # vehicles per 100m
    avg_speed = float(nearby['speed_ms'].mean())
    return float(density), avg_speed


def build_context_features(weather_table, scene: pd.DataFrame, ego_id: int, source: str, time_s: float) -> np.ndarray:
    ego_row = scene[scene['vehicle_id'] == ego_id].iloc[0]
    w = weather_for_snapshot(weather_table, source, time_s)
    temperature_c = float(w.get('temperature_2m') or 0.0)
    precipitation_mm = float(w.get('precipitation') or 0.0)
    visibility_km = float(w.get('visibility') or 0.0) / 1000.0
    windspeed_kmh = float(w.get('windspeed_10m') or 0.0)
    weathercode = float(w.get('weathercode') or 0.0)
    is_precipitating = 1.0 if precipitation_mm > 0 else 0.0

    density, avg_speed = local_traffic_stats(scene, ego_row)

    return np.array([
        temperature_c, precipitation_mm, visibility_km, windspeed_kmh,
        weathercode, is_precipitating, density, avg_speed,
    ], dtype=np.float32)


def extract_features(scene, ego_id):
    ego, neigh, mask = build_snapshot_features(scene, ego_id, top_k=TOP_K, max_radius_m=MAX_RADIUS_M)
    return ego.astype(np.float32), neigh.astype(np.float32), mask.astype(bool)


def process_split(events_df, windows_df, weather_table, split_name):
    events = events_df[events_df['split'] == split_name].reset_index(drop=True)
    print(f'\n[A2] Processing {split_name}: {len(events)} positive events')
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
        ctx_pos = build_context_features(weather_table, scene_pos, ego_pos_id, src, float(event['start_time_s']))

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
        ctx_neg = build_context_features(weather_table, scene_neg, ego_neg_id, neg_src, float(t_neg))

        features.append({
            'ego_pos': f_ego_pos, 'neighbors_pos': f_neigh_pos, 'mask_pos': mask_pos, 'map_pos': ctx_pos,
            'ego_neg': f_ego_neg, 'neighbors_neg': f_neigh_neg, 'mask_neg': mask_neg, 'map_neg': ctx_neg,
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

    for split in ['train', 'cal', 'holdout']:
        data = process_split(events, windows, weather_table, split)
        if data is not None:
            out_path = OUT_DIR / f'a2_{split}.npz'
            np.savez_compressed(out_path, **data)
            print(f'Saved {out_path} with {data["labels_pos"].shape[0]} examples')
        else:
            print(f'No examples for {split}')

    meta = {
        'map_dim': MAP_DIM, 'top_k': TOP_K, 'max_radius_m': MAX_RADIUS_M, 'ablation': 'A2',
        'mitra_representative_hour': MITRA_REPRESENTATIVE_HOUR,
        'traffic_lane_window_m': TRAFFIC_LANE_WINDOW_M,
    }
    with open(OUT_DIR / 'a2_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    main()
