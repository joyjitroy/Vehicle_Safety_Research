"""Extract hard-braking and queue-onset events from canonical trajectory DataFrames."""
from typing import List, Optional

import numpy as np
import pandas as pd


def _find_runs(mask: np.ndarray):
    """Return (start_idx, end_idx) inclusive for consecutive True runs."""
    if len(mask) == 0:
        return []
    m = mask.astype(int)
    diff = np.diff(m, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts, ends))


def extract_hard_brake_events(
    df: pd.DataFrame,
    acc_col: str = 'lon_acc_ms2',
    threshold_ms2: float = -3.9,
    min_duration_s: float = 0.5,
    fps: float = 10.0,
    smooth_window_s: Optional[float] = None,
    vehicle_col: str = 'vehicle_id',
    lane_col: str = 'lane_id',
    time_col: str = 'time_s',
    frame_col: Optional[str] = None,
) -> pd.DataFrame:
    """Extract independent hard-braking episodes.

    Smoothing window defaults to min_duration_s to match the event duration.
    """
    if smooth_window_s is None:
        smooth_window_s = min_duration_s
    min_frames = int(np.ceil(min_duration_s * fps))
    window_frames = max(1, int(np.round(smooth_window_s * fps)))
    if window_frames % 2 == 0:
        window_frames += 1  # centering works best with odd windows

    df = df.copy()
    df['_order'] = np.arange(len(df))
    df = df.sort_values([vehicle_col, time_col]).reset_index(drop=True)

    df['_acc_smooth'] = (
        df.groupby(vehicle_col, sort=False)[acc_col]
        .rolling(window=window_frames, min_periods=1, center=True)
        .mean()
        .reset_index(level=0, drop=True)
    )

    events = []
    for vid, g in df.groupby(vehicle_col, sort=False):
        g = g.sort_values(time_col).reset_index(drop=True)
        mask = g['_acc_smooth'].values <= threshold_ms2
        for s, e in _find_runs(mask):
            if e - s >= min_frames:
                seg = g.iloc[s:e]
                events.append({
                    vehicle_col: int(vid),
                    lane_col: int(seg[lane_col].iloc[0]),
                    'start_time_s': float(seg[time_col].iloc[0]),
                    'end_time_s': float(seg[time_col].iloc[-1]),
                    'duration_s': float(seg[time_col].iloc[-1] - seg[time_col].iloc[0]),
                    'min_acc_ms2': float(seg['_acc_smooth'].min()),
                    'type': 'hard_brake',
                })
    return pd.DataFrame(events) if events else pd.DataFrame(columns=[vehicle_col, lane_col, 'start_time_s', 'end_time_s', 'duration_s', 'min_acc_ms2', 'type'])


def extract_queue_events(
    df: pd.DataFrame,
    speed_threshold_ms: float = 2.0,
    min_vehicles: int = 3,
    max_gap_m: float = 5.0,
    min_duration_s: float = 5.0,
    fps: float = 10.0,
    merge_gap_s: float = 10.0,
    max_y_diff_m: float = 50.0,
    max_cluster_y_range_m: float = 30.0,
    time_col: str = 'time_s',
    lane_col: str = 'lane_id',
    y_col: str = 'y_m',
    length_col: str = 'vehicle_length_m',
    vehicle_col: str = 'vehicle_id',
) -> pd.DataFrame:
    """Extract independent queue-onset episodes from a canonical DataFrame.

    Fast implementation: per (frame, lane) checks whether there are at least
    ``min_vehicles`` slow vehicles whose longitudinal spread is within
    ``max_cluster_y_range_m``. Consecutive qualifying frames in the same lane
    are merged into an episode. This preserves the core contiguous-vehicle
    requirement while avoiding the slow per-vehicle clustering step on very
    large datasets.
    """
    min_frames = int(np.ceil(min_duration_s * fps))

    df = df[[time_col, lane_col, y_col, 'speed_ms']].copy()
    df['_slow'] = df['speed_ms'] < speed_threshold_ms
    df = df[df['_slow']].copy()

    if df.empty:
        return pd.DataFrame(columns=[lane_col, 'start_time_s', 'end_time_s', 'duration_s', 'max_vehicle_count', 'y_m', 'type'])

    # Per frame/lane: count slow vehicles and their y spread
    stats = (
        df.groupby([time_col, lane_col], sort=False)
        .agg(
            count=(y_col, 'size'),
            y_min=(y_col, 'min'),
            y_max=(y_col, 'max'),
        )
        .reset_index()
    )
    stats['y_range'] = stats['y_max'] - stats['y_min']
    stats['_qualifies'] = (stats['count'] >= min_vehicles) & (stats['y_range'] <= max_cluster_y_range_m)
    stats = stats[stats['_qualifies']].copy()
    stats['y_m'] = (stats['y_min'] + stats['y_max']) / 2.0

    if stats.empty:
        return pd.DataFrame(columns=[lane_col, 'start_time_s', 'end_time_s', 'duration_s', 'max_vehicle_count', 'y_m', 'type'])

    # Build episodes per lane by merging temporally/spatially close frames
    episodes = []
    for lane, g in stats.groupby(lane_col, sort=False):
        g = g.sort_values(time_col).reset_index(drop=True)
        current = None
        for _, row in g.iterrows():
            if current is None:
                current = {
                    'start_time_s': float(row[time_col]),
                    'end_time_s': float(row[time_col]),
                    'frames': 1,
                    'max_vehicle_count': int(row['count']),
                    'y_m': float(row['y_m']),
                }
                continue
            time_gap = float(row[time_col]) - current['end_time_s']
            y_gap = abs(float(row['y_m']) - current['y_m'])
            if time_gap <= merge_gap_s and y_gap <= max_y_diff_m:
                current['end_time_s'] = float(row[time_col])
                current['frames'] += 1
                current['max_vehicle_count'] = max(current['max_vehicle_count'], int(row['count']))
                current['y_m'] = 0.5 * (current['y_m'] + float(row['y_m']))
            else:
                if current['frames'] >= min_frames:
                    episodes.append({
                        lane_col: int(lane),
                        'start_time_s': current['start_time_s'],
                        'end_time_s': current['end_time_s'],
                        'duration_s': current['end_time_s'] - current['start_time_s'],
                        'max_vehicle_count': current['max_vehicle_count'],
                        'y_m': current['y_m'],
                        'type': 'queue',
                    })
                current = {
                    'start_time_s': float(row[time_col]),
                    'end_time_s': float(row[time_col]),
                    'frames': 1,
                    'max_vehicle_count': int(row['count']),
                    'y_m': float(row['y_m']),
                }
        if current is not None and current['frames'] >= min_frames:
            episodes.append({
                lane_col: int(lane),
                'start_time_s': current['start_time_s'],
                'end_time_s': current['end_time_s'],
                'duration_s': current['end_time_s'] - current['start_time_s'],
                'max_vehicle_count': current['max_vehicle_count'],
                'y_m': current['y_m'],
                'type': 'queue',
            })

    return pd.DataFrame(episodes) if episodes else pd.DataFrame(columns=[lane_col, 'start_time_s', 'end_time_s', 'duration_s', 'max_vehicle_count', 'y_m', 'type'])
