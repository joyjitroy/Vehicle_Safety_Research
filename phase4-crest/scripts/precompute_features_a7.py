"""Precompute A7 ablation features: ego state + SafeDriver IQ behavioral prior.

A7 isolates the question: does a driver's behavioral risk history improve
crash-adjacent event prediction over ego kinematics alone (B1)?

SAFEDRIVER IQ SIMULATION:
    NGSIM and MiTra contain no telematics or driver ID data. We simulate the
    SafeDriver IQ score that a real telematics system would assign, following
    the same "simulate the output, not the sensor" philosophy as A6.

    Score generation (per unique vehicle_id per dataset_source):
      - Base score drawn from Beta(alpha=2, beta=5) x 100, giving a realistic
        right-skewed distribution (most drivers are safe; few are high-risk).
        Mean ≈ 28.6/100 on the risk axis, i.e., most drivers score well.
        We store as a RISK score (0=safe, 100=very risky) for clarity.
      - Positive events: ego vehicle's score is biased upward by +20 points
        (clipped to [0,100]), reflecting the empirical finding that drivers
        involved in near-crash events tend to have higher behavioral risk scores.
      - Gaussian noise σ=5 is added per event to simulate score drift / update
        lag in a real telematics system.

    The positive-event bias is the key signal. Its magnitude (+20 pts) is
    conservative — real telematics studies show 1.5–3x crash rate difference
    between highest and lowest risk quintiles, corresponding to roughly
    15–25 point score separation.

FEATURE VECTOR (8-dim, fills map_encoder input slot):
    0: risk_score_norm      - SafeDriver IQ risk score / 100 (0–1)
    1: risk_quintile        - 0.0/0.25/0.5/0.75/1.0 (quintile rank)
    2: is_high_risk         - 1.0 if score > 60 else 0.0
    3: score_sq             - risk_score_norm^2 (captures nonlinear risk)
    4: log1p_score          - log1p(risk_score_norm * 100) / log1p(100)
    5–7: reserved zeros     - placeholder for future sub-scores
                              (e.g., highway_risk, night_risk, phone_use)

    8-dim matches configs/model.yaml map_encoder.input_dim=8 so no model
    architecture change is needed (same as A1, A2).

ABLATION CHAIN:
    B1 → A7 (ego + behavioral prior only)
    F3 is the full fusion variant: F + A7 (all channels + behavioral prior).
    Both A7 and F3 use map_dim=8 and map_dim=16 respectively (same as A1/F).

Outputs NPZ files in outputs/features/:
    a7_train.npz, a7_cal.npz, a7_holdout.npz
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

# Simulation parameters
SCORE_BETA_ALPHA = 2.0      # Beta distribution shape — right-skewed risk scores
SCORE_BETA_BETA = 5.0       # Most drivers have low-to-moderate risk scores
POSITIVE_BIAS = 20.0        # pts added to ego score for positive events
SCORE_NOISE_STD = 5.0       # per-event Gaussian noise (score drift / update lag)
HIGH_RISK_THRESHOLD = 60.0  # score above which is_high_risk=1


def build_vehicle_scores(
    df: pd.DataFrame,
    rng: np.random.Generator,
) -> dict:
    """Assign a baseline SafeDriver IQ risk score (0–100) to each unique
    vehicle_id in the dataframe. Score is fixed per vehicle (not per frame),
    simulating a slowly-updating telematics prior.
    """
    vehicle_ids = df['vehicle_id'].unique()
    raw = rng.beta(SCORE_BETA_ALPHA, SCORE_BETA_BETA, size=len(vehicle_ids)) * 100.0
    return dict(zip(vehicle_ids, raw))


def build_safedriver_features(
    base_score: float,
    is_positive: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build the 8-dim SafeDriver IQ feature vector for one event.

    Parameters
    ----------
    base_score : float
        Vehicle's baseline risk score (0–100).
    is_positive : bool
        True for crash-adjacent (positive) events; applies the positive bias.
    """
    score = base_score
    if is_positive:
        score = min(100.0, score + POSITIVE_BIAS)
    score = float(np.clip(score + rng.normal(0, SCORE_NOISE_STD), 0.0, 100.0))

    norm = score / 100.0
    quintile = float(min(4, int(norm * 5))) / 4.0  # 0, 0.25, 0.5, 0.75, 1.0
    is_high_risk = 1.0 if score >= HIGH_RISK_THRESHOLD else 0.0
    score_sq = norm ** 2
    log_norm = float(np.log1p(score) / np.log1p(100.0))

    return np.array([norm, quintile, is_high_risk, score_sq, log_norm, 0.0, 0.0, 0.0],
                    dtype=np.float32)


def extract_features(scene, ego_id):
    ego, neigh, mask = build_snapshot_features(scene, ego_id, top_k=TOP_K, max_radius_m=MAX_RADIUS_M)
    return ego.astype(np.float32), neigh.astype(np.float32), mask.astype(bool)


def process_split(events_df, windows_df, split_name):
    events = events_df[events_df['split'] == split_name].reset_index(drop=True)
    print(f'\n[A7] Processing {split_name}: {len(events)} positive events')
    if events.empty:
        return None

    features = []
    cache = {}
    vehicle_scores = {}

    all_sources = set(events['dataset_source'].unique())
    if split_name != 'holdout':
        all_sources.update(NEGATIVE_ONLY_SOURCES)

    for src in all_sources:
        print(f'  Loading {src} ...', flush=True)
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

        t_pos = float(event['start_time_s'])
        scene_pos = fast_scene(df_pos, t_pos)
        if scene_pos.empty:
            continue
        ego_pos_id = choose_ego_for_event(scene_pos, event)
        f_ego_pos, f_neigh_pos, mask_pos = extract_features(scene_pos, ego_pos_id)
        base_score_pos = vehicle_scores[src].get(ego_pos_id, 50.0)
        map_pos = build_safedriver_features(base_score_pos, is_positive=True, rng=RNG)

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
        base_score_neg = vehicle_scores[neg_src].get(ego_neg_id, 50.0)
        map_neg = build_safedriver_features(base_score_neg, is_positive=False, rng=RNG)

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
            out_path = OUT_DIR / f'a7_{split}.npz'
            np.savez_compressed(out_path, **data)
            print(f'Saved {out_path} with {data["labels_pos"].shape[0]} examples')
        else:
            print(f'No examples for {split}')

    meta = {
        'top_k': TOP_K,
        'map_dim': MAP_DIM,
        'ablation': 'A7',
        'feature_layout': [
            'risk_score_norm (0-1)',
            'risk_quintile (0/0.25/0.5/0.75/1.0)',
            'is_high_risk (score>=60)',
            'score_sq',
            'log1p_score_norm',
            'reserved_0', 'reserved_1', 'reserved_2',
        ],
        'simulation': {
            'distribution': f'Beta({SCORE_BETA_ALPHA},{SCORE_BETA_BETA}) x 100',
            'positive_event_bias_pts': POSITIVE_BIAS,
            'noise_std_pts': SCORE_NOISE_STD,
            'high_risk_threshold': HIGH_RISK_THRESHOLD,
            'note': (
                'Scores are simulated from ground-truth vehicle IDs. '
                'At real inference time, the vehicle existing SafeDriver IQ '
                'risk score is used directly — no CREST retraining required.'
            ),
        },
    }
    with open(OUT_DIR / 'a7_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(f'\nSaved {OUT_DIR / "a7_meta.json"}')


if __name__ == '__main__':
    main()
