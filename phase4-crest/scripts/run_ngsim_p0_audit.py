"""NGSIM P0 audit: independent hard-braking and queue-onset events.

Uses frozen label definitions from the CREST research plan:
- Hard braking: lon_acc <= -3.9 m/s^2 for >= 0.5 s
- Queue onset: >= 3 contiguous vehicles with speed < 2 m/s for >= 5 s
- Merge temporally proximate events (within 10 s) in the same lane.
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw' / 'ngsim'
OUT_DIR = ROOT / 'outputs' / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / 'ngsim_p0_audit.csv'
OUT_JSON = OUT_DIR / 'ngsim_p0_audit_summary.json'

HARD_ACC_MS2 = -3.9
HARD_DURATION_S = 0.5
QUEUE_SPEED_MS = 2.0
QUEUE_DURATION_S = 5.0
MIN_QUEUE_VEHICLES = 3
MAX_QUEUE_GAP_M = 5.0
MERGE_GAP_S = 10.0

FT_TO_M = 0.3048
MS_TO_KMH = 3.6


def load_trajectory(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    # Convert units
    df['speed_ms'] = df['v_Vel'] * FT_TO_M
    df['lon_acc_ms2'] = df['v_Acc'] * FT_TO_M
    df['time_s'] = df['Global_Time'] / 1000.0
    df['length_m'] = df['v_Length'] * FT_TO_M
    df['x_m'] = df['Local_X'] * FT_TO_M
    df['y_m'] = df['Local_Y'] * FT_TO_M
    return df


def smooth_acceleration(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Apply a per-vehicle moving average to acceleration to reduce NGSIM spikes."""
    groups = []
    for vid, g in df.groupby('Vehicle_ID', sort=False):
        g = g.sort_values('Frame_ID')
        g['lon_acc_ms2_smooth'] = g['lon_acc_ms2'].rolling(window=window, min_periods=1, center=True).mean()
        groups.append(g)
    return pd.concat(groups, ignore_index=True)


def find_runs(mask: np.ndarray) -> list:
    """Return (start_idx, end_idx) inclusive for consecutive True runs."""
    if len(mask) == 0:
        return []
    m = mask.astype(int)
    diff = np.diff(m)
    starts = np.where((np.concatenate(([m[0]], diff)) == 1))[0]
    ends = np.where((np.concatenate((diff, [-m[-1]])) == -1))[0]
    return list(zip(starts, ends))


def extract_hard_brake_events(df: pd.DataFrame) -> pd.DataFrame:
    """Per-vehicle hard-braking episodes using smoothed acceleration."""
    events = []
    min_frames = int(np.ceil(HARD_DURATION_S * 10))  # NGSIM is 10 Hz
    for vid, g in df.groupby('Vehicle_ID', sort=False):
        g = g.sort_values('Frame_ID').reset_index(drop=True)
        mask = g['lon_acc_ms2_smooth'].values <= HARD_ACC_MS2
        for s, e in find_runs(mask):
            dur_frames = e - s + 1
            if dur_frames >= min_frames:
                seg = g.iloc[s:e+1]
                events.append({
                    'vehicle_id': int(vid),
                    'lane_id': int(seg['Lane_ID'].iloc[0]),
                    'start_time_s': float(seg['time_s'].iloc[0]),
                    'end_time_s': float(seg['time_s'].iloc[-1]),
                    'duration_s': float(seg['time_s'].iloc[-1] - seg['time_s'].iloc[0]),
                    'min_acc_ms2': float(seg['lon_acc_ms2_smooth'].min()),
                    'mean_speed_ms': float(seg['speed_ms'].mean()),
                    'y_m': float(seg['y_m'].mean()),
                    'source_file': Path(g.attrs.get('source', '')).name if hasattr(g, 'attrs') else '',
                })
    return pd.DataFrame(events)


def extract_queue_events(df: pd.DataFrame) -> pd.DataFrame:
    """Per-frame per-lane queue clusters, then merge into episodes."""
    min_frames = int(np.ceil(QUEUE_DURATION_S * 10))
    events = []
    # Work on sorted frame/lane
    df = df.sort_values(['Frame_ID', 'Lane_ID', 'y_m']).reset_index(drop=True)
    frames = sorted(df['Frame_ID'].unique())
    lane_ids = sorted(df['Lane_ID'].unique())

    active_clusters = []  # list of dicts: lane_id, ids set, start_frame, last_frame, y_anchor
    cluster_id = 0
    for frame in frames:
        frame_df = df[df['Frame_ID'] == frame]
        for lane in lane_ids:
            lane_df = frame_df[frame_df['Lane_ID'] == lane]
            if len(lane_df) < MIN_QUEUE_VEHICLES:
                continue
            slow = lane_df[lane_df['speed_ms'] < QUEUE_SPEED_MS]
            if len(slow) < MIN_QUEUE_VEHICLES:
                continue
            # sort by longitudinal position and cluster by gap
            slow = slow.sort_values('y_m').reset_index(drop=True)
            y = slow['y_m'].values
            length = slow['length_m'].values
            ids = slow['Vehicle_ID'].values
            if len(y) == 0:
                continue
            cluster_ids = [ids[0]]
            cluster_ys = [y[0]]
            for i in range(1, len(y)):
                gap = y[i] - y[i-1] - length[i-1]
                if gap <= MAX_QUEUE_GAP_M:
                    cluster_ids.append(ids[i])
                    cluster_ys.append(y[i])
                else:
                    if len(cluster_ids) >= MIN_QUEUE_VEHICLES:
                        active_clusters.append({
                            'cluster_id': cluster_id,
                            'lane_id': int(lane),
                            'frame': int(frame),
                            'time_s': float(slow['time_s'].iloc[0]),
                            'ids': set(cluster_ids),
                            'y_m': float(np.median(cluster_ys)),
                        })
                        cluster_id += 1
                    cluster_ids = [ids[i]]
                    cluster_ys = [y[i]]
            if len(cluster_ids) >= MIN_QUEUE_VEHICLES:
                active_clusters.append({
                    'cluster_id': cluster_id,
                    'lane_id': int(lane),
                    'frame': int(frame),
                    'time_s': float(slow['time_s'].iloc[0]),
                    'ids': set(cluster_ids),
                    'y_m': float(np.median(cluster_ys)),
                })
                cluster_id += 1

    # Merge consecutive frames/clusters into episodes in same lane with overlapping or contiguous ids
    episodes = []
    # group active clusters by lane
    by_lane = {}
    for cl in active_clusters:
        by_lane.setdefault(cl['lane_id'], []).append(cl)

    for lane, clusters in by_lane.items():
        clusters = sorted(clusters, key=lambda c: c['frame'])
        current = None
        for cl in clusters:
            if current is None:
                current = {
                    'lane_id': lane,
                    'start_frame': cl['frame'],
                    'end_frame': cl['frame'],
                    'start_time_s': cl['time_s'],
                    'end_time_s': cl['time_s'],
                    'ids': set(cl['ids']),
                    'y_m': cl['y_m'],
                }
                continue
            # extend if within 1 frame temporal gap and ids overlap or gap <= 10s
            if (cl['frame'] - current['end_frame'] <= 1) and not current['ids'].isdisjoint(cl['ids']):
                current['end_frame'] = cl['frame']
                current['end_time_s'] = cl['time_s']
                current['ids'].update(cl['ids'])
            elif (cl['time_s'] - current['end_time_s']) <= MERGE_GAP_S and abs(cl['y_m'] - current['y_m']) <= 50.0:
                # treat as same episode resumed after small gap
                current['end_frame'] = cl['frame']
                current['end_time_s'] = cl['time_s']
                current['ids'].update(cl['ids'])
            else:
                dur_frames = current['end_frame'] - current['start_frame'] + 1
                if dur_frames >= min_frames:
                    episodes.append({
                        'lane_id': int(current['lane_id']),
                        'start_time_s': float(current['start_time_s']),
                        'end_time_s': float(current['end_time_s']),
                        'duration_s': float(current['end_time_s'] - current['start_time_s']),
                        'vehicle_count': len(current['ids']),
                        'y_m': float(current['y_m']),
                    })
                current = {
                    'lane_id': lane,
                    'start_frame': cl['frame'],
                    'end_frame': cl['frame'],
                    'start_time_s': cl['time_s'],
                    'end_time_s': cl['time_s'],
                    'ids': set(cl['ids']),
                    'y_m': cl['y_m'],
                }
        if current is not None:
            dur_frames = current['end_frame'] - current['start_frame'] + 1
            if dur_frames >= min_frames:
                episodes.append({
                    'lane_id': int(current['lane_id']),
                    'start_time_s': float(current['start_time_s']),
                    'end_time_s': float(current['end_time_s']),
                    'duration_s': float(current['end_time_s'] - current['start_time_s']),
                    'vehicle_count': len(current['ids']),
                    'y_m': float(current['y_m']),
                })
    return pd.DataFrame(episodes)


def merge_temporal_events(events: pd.DataFrame, lane_col: str = 'lane_id') -> pd.DataFrame:
    """Merge events in the same lane whose time gaps are <= MERGE_GAP_S."""
    if events.empty:
        return events.copy()
    events = events.sort_values([lane_col, 'start_time_s']).reset_index(drop=True)
    merged = []
    current = None
    for _, row in events.iterrows():
        if current is None:
            current = row.to_dict()
            continue
        same_lane = row[lane_col] == current[lane_col]
        time_gap = row['start_time_s'] - current['end_time_s']
        if same_lane and time_gap <= MERGE_GAP_S:
            current['end_time_s'] = max(current['end_time_s'], row['end_time_s'])
            current['duration_s'] = current['end_time_s'] - current['start_time_s']
            # aggregate counts / min values as available
            if 'vehicle_count' in row and 'vehicle_count' in current:
                current['vehicle_count'] = max(current['vehicle_count'], row['vehicle_count'])
            if 'min_acc_ms2' in row and 'min_acc_ms2' in current:
                current['min_acc_ms2'] = min(current['min_acc_ms2'], row['min_acc_ms2'])
        else:
            merged.append(current)
            current = row.to_dict()
    if current is not None:
        merged.append(current)
    return pd.DataFrame(merged)


def audit_file(path: Path) -> dict:
    print(f'Auditing {path.relative_to(RAW)} ...', flush=True)
    t0 = time.time()
    df = load_trajectory(path)
    df = smooth_acceleration(df)
    file_tag = f"{path.parent.parent.parent.name}/{path.parent.name}/{path.name}"

    hb = extract_hard_brake_events(df)
    hb_merged = merge_temporal_events(hb)

    q = extract_queue_events(df)
    q_merged = merge_temporal_events(q)

    elapsed = time.time() - t0
    print(f'  hard_brake={len(hb_merged)}, queue={len(q_merged)}, time={elapsed:.1f}s', flush=True)
    return {
        'file': file_tag,
        'path': str(path),
        'total_rows': len(df),
        'unique_vehicles': int(df['Vehicle_ID'].nunique()),
        'hard_brake_events': len(hb_merged),
        'queue_events': len(q_merged),
        'raw_hard_brake_episodes': len(hb),
        'raw_queue_episodes': len(q),
        'hard_brake_details': hb_merged.to_dict('records'),
        'queue_details': q_merged.to_dict('records'),
    }


def main():
    files = sorted([
        p for p in RAW.rglob('vehicle-trajectory-data/**/*.csv')
        if 'RECONSTRUCTED' not in str(p).upper() and 'detector' not in str(p).lower()
    ])
    print(f'Found {len(files)} NGSIM trajectory files.')
    results = [audit_file(p) for p in files]

    # Write summary CSV
    rows = []
    for r in results:
        rows.append({
            'file': r['file'],
            'total_rows': r['total_rows'],
            'unique_vehicles': r['unique_vehicles'],
            'hard_brake_events': r['hard_brake_events'],
            'queue_events': r['queue_events'],
        })
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['file', 'total_rows', 'unique_vehicles', 'hard_brake_events', 'queue_events'])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        'files': len(results),
        'total_rows': sum(r['total_rows'] for r in results),
        'total_unique_vehicles': sum(r['unique_vehicles'] for r in results),
        'total_hard_brake_events': sum(r['hard_brake_events'] for r in results),
        'total_queue_events': sum(r['queue_events'] for r in results),
        'per_file': results,
    }
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print('\nDone.')
    print(f'CSV: {OUT_CSV}')
    print(f'JSON: {OUT_JSON}')
    print(f'Total hard-brake events: {summary["total_hard_brake_events"]}')
    print(f'Total queue events: {summary["total_queue_events"]}')


if __name__ == '__main__':
    main()
