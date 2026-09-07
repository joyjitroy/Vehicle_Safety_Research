"""B0 baseline: analytical TTC / stopping-distance hazard classifier (no ML).

For every holdout event (positive) and every holdout negative snapshot used by
the B1 pipeline, this computes a physics-based hazard score from ego speed,
ego acceleration, and the gap to the leader vehicle (leader_id from the
canonical schema). No learned parameters are involved.

Assumptions (documented for the paper Methods section):
  - REACTION_TIME_S = 1.5 s   (typical driver perception-reaction time)
  - MAX_DECEL_MS2   = 8.0 m/s^2 (assumed max achievable braking deceleration,
                                  dry pavement, standard traffic-engineering value)
  - TTC_THRESHOLD_S = 2.0 s   (common forward-collision-warning threshold)
  - gap_m is approximated as the Euclidean distance between ego and leader
    positions (x_m, y_m). For NGSIM this is effectively longitudinal spacing
    since lateral offset within a lane is small; for MiTra (UTM coordinates)
    this is a straight-line approximation. This approximation is stated
    explicitly as a B0 baseline limitation, not used by the learned CREST model.

Hazard rule (binary):
  hazard = 1 if (TTC is defined and TTC < TTC_THRESHOLD_S) OR
                (gap_m < stopping_distance_m)
         else 0
  where stopping_distance_m = ego_speed_ms * REACTION_TIME_S
                               + ego_speed_ms**2 / (2 * MAX_DECEL_MS2)

Continuous risk score (for AUPRC / Brier / log-loss comparison with B1):
  risk_score = clip(0.5 * stopping_ratio + 0.5 * ttc_urgency, 0, 1)
  where stopping_ratio = clip(stopping_distance_m / max(gap_m, eps), 0, 1)
        ttc_urgency     = clip(TTC_THRESHOLD_S / max(TTC, eps), 0, 1) if TTC defined else 0

Reuses the identical event selection and negative-sampling logic as
scripts/precompute_features.py (choose_ego_for_event, fast_scene,
sample_negative_time) restricted to the holdout split, so B0 is evaluated on
the exact same examples as B1's outputs/features/holdout.npz.

Output: outputs/results/b0_baseline.json
  {
    "assumptions": {...},
    "metrics": {"auprc": ..., "brier": ..., "log_loss": ..., "n_samples": ...},
    "events": [ {event_id/index, dataset_source, time_s, ego_speed_ms, gap_m,
                 ttc_s, stopping_distance_m, hazard_flag, risk_score, label}, ... ]
  }
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from crest.datasets.crest_dataset import load_source_df

from precompute_features import (
    NEGATIVE_ONLY_SOURCES,
    choose_ego_for_event,
    choose_random_ego,
    fast_scene,
    positive_ranges,
    sample_negative_time,
)

EVENTS_CSV = ROOT / 'outputs' / 'results' / 'label_events_split.csv'
WINDOWS_CSV = ROOT / 'outputs' / 'results' / 'label_windows_split.csv'
OUT_PATH = ROOT / 'outputs' / 'results' / 'b0_baseline.json'
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

REACTION_TIME_S = 1.5
MAX_DECEL_MS2 = 8.0
TTC_THRESHOLD_S = 2.0
EPS = 1e-3


def stopping_distance_m(speed_ms: float) -> float:
    return speed_ms * REACTION_TIME_S + (speed_ms ** 2) / (2 * MAX_DECEL_MS2)


def compute_gap_and_ttc(scene: pd.DataFrame, ego_row: pd.Series):
    """Return (gap_m, ttc_s) for the ego's leader, or (None, None) if no leader in scene."""
    leader_id = int(ego_row['leader_id'])
    if leader_id <= 0:
        return None, None
    leader_rows = scene[scene['vehicle_id'] == leader_id]
    if leader_rows.empty:
        return None, None
    leader_row = leader_rows.iloc[0]

    dx = float(leader_row['x_m'] - ego_row['x_m'])
    dy = float(leader_row['y_m'] - ego_row['y_m'])
    gap_m = float(np.hypot(dx, dy))

    closing_speed = float(ego_row['speed_ms'] - leader_row['speed_ms'])
    if closing_speed > EPS:
        ttc_s = gap_m / closing_speed
    else:
        ttc_s = None
    return gap_m, ttc_s


def score_snapshot(scene: pd.DataFrame, ego_id: int):
    ego_row = scene[scene['vehicle_id'] == ego_id].iloc[0]
    ego_speed = float(ego_row['speed_ms'])
    gap_m, ttc_s = compute_gap_and_ttc(scene, ego_row)
    stop_dist = stopping_distance_m(ego_speed)

    if gap_m is None:
        # No leader observed in range; treat as no immediate forward hazard.
        return {
            'ego_speed_ms': ego_speed, 'gap_m': None, 'ttc_s': None,
            'stopping_distance_m': stop_dist, 'hazard_flag': 0, 'risk_score': 0.0,
        }

    stopping_ratio = float(np.clip(stop_dist / max(gap_m, EPS), 0.0, 1.0))
    ttc_urgency = float(np.clip(TTC_THRESHOLD_S / max(ttc_s, EPS), 0.0, 1.0)) if ttc_s is not None else 0.0
    risk_score = float(np.clip(0.5 * stopping_ratio + 0.5 * ttc_urgency, 0.0, 1.0))
    hazard_flag = int((ttc_s is not None and ttc_s < TTC_THRESHOLD_S) or (gap_m < stop_dist))

    return {
        'ego_speed_ms': ego_speed, 'gap_m': gap_m, 'ttc_s': ttc_s,
        'stopping_distance_m': stop_dist, 'hazard_flag': hazard_flag, 'risk_score': risk_score,
    }


def main():
    events_df = pd.read_csv(EVENTS_CSV)
    windows_df = pd.read_csv(WINDOWS_CSV)
    events = events_df[events_df['split'] == 'holdout'].reset_index(drop=True)
    print(f'Processing holdout: {len(events)} positive events')

    cache = {}
    sources = set(events['dataset_source'].unique())
    for src in sources:
        df = load_source_df(src)
        df = df.sort_values('time_s').reset_index(drop=True)
        df['_time_idx'] = df['time_s']
        cache[src] = df.set_index('_time_idx', drop=False)

    records = []
    for i, event in events.iterrows():
        src = event['dataset_source']
        df_pos = cache[src]

        scene_pos = fast_scene(df_pos, float(event['start_time_s']))
        if scene_pos.empty:
            continue
        ego_pos_id = choose_ego_for_event(scene_pos, event)
        result_pos = score_snapshot(scene_pos, ego_pos_id)
        result_pos.update({'dataset_source': src, 'time_s': float(event['start_time_s']), 'label': 1})
        records.append(result_pos)

        ranges = positive_ranges(windows_df, src)
        t_neg, scene_neg = sample_negative_time(df_pos, ranges)
        if t_neg is None:
            for fallback in NEGATIVE_ONLY_SOURCES:
                if fallback not in cache:
                    continue
                t_neg, scene_neg = sample_negative_time(cache[fallback], [])
                if t_neg is not None:
                    break
        if t_neg is None:
            continue
        ego_neg_id = choose_random_ego(scene_neg)
        result_neg = score_snapshot(scene_neg, ego_neg_id)
        result_neg.update({'dataset_source': src, 'time_s': float(t_neg), 'label': 0})
        records.append(result_neg)

    labels = np.array([r['label'] for r in records])
    scores = np.clip(np.array([r['risk_score'] for r in records]), 1e-6, 1 - 1e-6)

    metrics = {
        'n_samples': int(len(labels)),
        'n_positive': int(labels.sum()),
        'n_negative': int(len(labels) - labels.sum()),
        'auprc': float(average_precision_score(labels, scores)),
        'brier': float(brier_score_loss(labels, scores)),
        'log_loss': float(log_loss(labels, scores)),
    }

    output = {
        'assumptions': {
            'reaction_time_s': REACTION_TIME_S,
            'max_decel_ms2': MAX_DECEL_MS2,
            'ttc_threshold_s': TTC_THRESHOLD_S,
            'gap_approximation': 'Euclidean distance between ego and leader (x_m, y_m)',
        },
        'metrics': metrics,
        'events': records,
    }
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f'Saved {OUT_PATH}')


if __name__ == '__main__':
    main()
