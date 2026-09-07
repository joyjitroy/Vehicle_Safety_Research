"""Precompute A6 ablation features: ego + simulated onboard perception detections.

A6 tests the value of onboard perception (camera/LiDAR/radar) as an inference-time
input to CREST. Since NGSIM and MiTra contain no sensor data, we simulate the
PERCEPTION OUTPUT (the detection list the vehicle's existing perception stack
would produce) directly from ground-truth trajectories by applying:

  1. Field-of-view filter  — front-facing cone ±60° from ego heading direction.
     Heading is approximated from consecutive position deltas (dx, dy over time).
     NGSIM and MiTra have no explicit heading field; we derive it from x_m/y_m.
  2. Range filter          — maximum detection range 100m (configurable).
  3. Detection uncertainty — per detected vehicle per frame:
       missed_detection_prob = 0.10  (higher than RSU; occlusion, weather)
       position_noise_std_m  = 0.30  (better than RSU; closer range, better sensor)
       speed_noise_std_ms    = 0.10
       false_positive_ghost_prob = 0.005 (lower than RSU; camera more precise)
       No stale track injection (onboard refreshes at sensor rate)

This is structurally identical to A3 (V2V simulation) — only the simulator
differs. The neighbor tensor is replaced by simulated perception detections;
the map_encoder slot remains zero (A6 isolates the perception channel).

Outputs NPZ files in outputs/features/:
  a6_train.npz, a6_cal.npz, a6_holdout.npz

Ablation chain position:
  B1 → A1 → A2 → A3 → A4 → F → A6 (perception, inference-only simulation)

Architecture note: At real inference time, the vehicle's existing perception
stack provides the detection list — no CREST retraining required. A6 estimates
the upper bound of this contribution using simulated perception uncertainty
on ground-truth trajectories.
"""
import json
import math
import sys
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
OUT_DIR = ROOT / 'outputs' / 'features'
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAP_DIM = 8
RNG = np.random.default_rng(42)

# ── Perception simulation parameters ──────────────────────────────────────────
PERCEPTION_RANGE_M = 100.0        # max detection range
FOV_HALF_ANGLE_DEG = 60.0         # ±60° front-facing cone
MISSED_DETECTION_PROB = 0.10      # per vehicle per frame
POSITION_NOISE_STD_M = 0.30       # lateral/longitudinal position noise
SPEED_NOISE_STD_MS = 0.10         # speed noise
FALSE_POSITIVE_PROB = 0.005       # ghost object per frame
# ──────────────────────────────────────────────────────────────────────────────


def estimate_ego_heading(df: pd.DataFrame, ego_id: int, t: float) -> float:
    """Estimate ego heading (radians) from consecutive position deltas.

    Looks for the two closest timestamps to t for the ego vehicle and computes
    the heading angle from the position change. Falls back to 0.0 (forward =
    +x direction) if fewer than two rows are available.
    """
    ego_rows = df[df['vehicle_id'] == ego_id].sort_values('time_s')
    if len(ego_rows) < 2:
        return 0.0
    idx = ego_rows['time_s'].searchsorted(t)
    idx = max(1, min(idx, len(ego_rows) - 1))
    r0 = ego_rows.iloc[idx - 1]
    r1 = ego_rows.iloc[idx]
    dx = float(r1['x_m']) - float(r0['x_m'])
    dy = float(r1['y_m']) - float(r0['y_m'])
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0
    return math.atan2(dy, dx)


def in_fov(ego_x: float, ego_y: float, heading_rad: float,
           veh_x: float, veh_y: float,
           fov_half_rad: float, max_range_m: float) -> bool:
    """Return True if vehicle (veh_x, veh_y) is within the ego perception cone."""
    dx = veh_x - ego_x
    dy = veh_y - ego_y
    dist = math.hypot(dx, dy)
    if dist > max_range_m or dist < 1e-3:
        return False
    angle_to_veh = math.atan2(dy, dx)
    diff = abs(math.atan2(math.sin(angle_to_veh - heading_rad),
                          math.cos(angle_to_veh - heading_rad)))
    return diff <= fov_half_rad


def simulate_perception_scene(
    scene: pd.DataFrame,
    ego_id: int,
    df_full: pd.DataFrame,
    t: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Return a scene where non-ego vehicles are replaced by simulated
    onboard perception detections within the ego's field of view.

    Steps:
      1. Estimate ego heading from consecutive position deltas.
      2. Filter others to FOV cone + range.
      3. Apply missed detection, position noise, false positives.
    """
    ego_row = scene[scene['vehicle_id'] == ego_id].iloc[0]
    ego_x = float(ego_row['x_m'])
    ego_y = float(ego_row['y_m'])
    heading = estimate_ego_heading(df_full, ego_id, t)
    fov_half_rad = math.radians(FOV_HALF_ANGLE_DEG)

    others = scene[scene['vehicle_id'] != ego_id].copy()

    # FOV + range filter
    in_cone = others.apply(
        lambda r: in_fov(ego_x, ego_y, heading,
                         float(r['x_m']), float(r['y_m']),
                         fov_half_rad, PERCEPTION_RANGE_M),
        axis=1,
    )
    visible = others[in_cone].copy()

    # Missed detection
    if not visible.empty:
        keep = rng.random(len(visible)) >= MISSED_DETECTION_PROB
        visible = visible[keep].copy()

    # Position noise
    if not visible.empty:
        visible['x_m'] = visible['x_m'] + rng.normal(0, POSITION_NOISE_STD_M, len(visible))
        visible['y_m'] = visible['y_m'] + rng.normal(0, POSITION_NOISE_STD_M, len(visible))
        visible['speed_ms'] = (visible['speed_ms']
                               + rng.normal(0, SPEED_NOISE_STD_MS, len(visible))).clip(lower=0.0)

    # False positive ghost objects (rare)
    if rng.random() < FALSE_POSITIVE_PROB:
        angle = rng.uniform(-fov_half_rad, fov_half_rad) + heading
        dist = rng.uniform(10.0, PERCEPTION_RANGE_M)
        ghost = ego_row.copy()
        ghost['vehicle_id'] = -1
        ghost['x_m'] = ego_x + dist * math.cos(angle)
        ghost['y_m'] = ego_y + dist * math.sin(angle)
        ghost['speed_ms'] = rng.uniform(0.0, 30.0)
        ghost['lon_acc_ms2'] = 0.0
        visible = pd.concat([visible, ghost.to_frame().T], ignore_index=True)

    return pd.concat([ego_row.to_frame().T, visible], ignore_index=True)


def extract_features(scene, ego_id):
    ego, neigh, mask = build_snapshot_features(scene, ego_id, top_k=TOP_K, max_radius_m=MAX_RADIUS_M)
    return ego.astype(np.float32), neigh.astype(np.float32), mask.astype(bool)


def process_split(events_df, windows_df, split_name):
    events = events_df[events_df['split'] == split_name].reset_index(drop=True)
    print(f'\n[A6] Processing {split_name}: {len(events)} positive events')
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
        df = load_source_df(src)
        df = df.sort_values('time_s').reset_index(drop=True)
        df['_time_idx'] = df['time_s']
        df = df.set_index('_time_idx', drop=False)
        cache[src] = df

    for i, event in events.iterrows():
        if i % 500 == 0 and i > 0:
            print(f'    processed {i}/{len(events)}', flush=True)
        src = event['dataset_source']
        df_pos = cache[src]

        t_pos = float(event['start_time_s'])
        scene_pos = fast_scene(df_pos, t_pos)
        if scene_pos.empty:
            continue
        ego_pos_id = choose_ego_for_event(scene_pos, event)
        perc_scene_pos = simulate_perception_scene(scene_pos, ego_pos_id, df_pos, t_pos, RNG)
        f_ego_pos, f_neigh_pos, mask_pos = extract_features(perc_scene_pos, ego_pos_id)

        ranges = positive_ranges(windows_df, src)
        t_neg, scene_neg = sample_negative_time(df_pos, ranges)
        if t_neg is None:
            for fallback in NEGATIVE_ONLY_SOURCES:
                t_neg, scene_neg = sample_negative_time(cache[fallback], [])
                if t_neg is not None:
                    break
        if t_neg is None:
            print(f'Warning: could not sample negative for event {i}; skipping')
            continue
        ego_neg_id = choose_random_ego(scene_neg)
        # Find which cached source the negative time came from; fall back to
        # the positive source's df if we can't determine it (rare edge case).
        _neg_src_key = None
        for _s, _df in cache.items():
            if hasattr(_df.index, '__contains__') and t_neg in _df.index:
                _neg_src_key = _s
                break
        df_neg = cache[_neg_src_key] if _neg_src_key is not None else df_pos
        perc_scene_neg = simulate_perception_scene(scene_neg, ego_neg_id, df_neg, t_neg, RNG)
        f_ego_neg, f_neigh_neg, mask_neg = extract_features(perc_scene_neg, ego_neg_id)

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

    for split in ['train', 'cal', 'holdout']:
        data = process_split(events, windows, split)
        if data is not None:
            out_path = OUT_DIR / f'a6_{split}.npz'
            np.savez_compressed(out_path, **data)
            print(f'Saved {out_path} with {data["labels_pos"].shape[0]} examples')
        else:
            print(f'No examples for {split}')

    meta = {
        'top_k': TOP_K,
        'perception_range_m': PERCEPTION_RANGE_M,
        'fov_half_angle_deg': FOV_HALF_ANGLE_DEG,
        'ablation': 'A6',
        'detection_uncertainty': {
            'missed_detection_prob': MISSED_DETECTION_PROB,
            'position_noise_std_m': POSITION_NOISE_STD_M,
            'speed_noise_std_ms': SPEED_NOISE_STD_MS,
            'false_positive_ghost_prob': FALSE_POSITIVE_PROB,
        },
        'note': (
            'Perception output simulated from ground-truth trajectories. '
            'At real inference time, the vehicle existing perception stack '
            'provides the detection list — no CREST retraining required.'
        ),
    }
    with open(OUT_DIR / 'a6_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(f'\nSaved {OUT_DIR / "a6_meta.json"}')


if __name__ == '__main__':
    main()
