"""Hard-braking threshold sensitivity on NGSIM with 5-frame moving-average v_Acc."""
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw' / 'ngsim'
CONFIG = ROOT / 'configs' / 'base.yaml'
OUT_CSV = ROOT / 'outputs' / 'results' / 'ngsim_hard_brake_threshold_sensitivity.csv'
OUT_JSON = ROOT / 'outputs' / 'results' / 'ngsim_hard_brake_threshold_sensitivity.json'

FT_TO_M = 0.3048
THRESHOLDS = [-3.0, -3.5, -3.9, -4.5]
MIN_FRAMES = 5
MERGE_GAP_S = 10.0


def load_smoothed(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df['lon_acc_ms2'] = df['v_Acc'] * FT_TO_M
    df['time_s'] = df['Global_Time'] / 1000.0
    df = df.sort_values(['Vehicle_ID', 'Frame_ID']).reset_index(drop=True)
    df['acc_smooth'] = (
        df.groupby('Vehicle_ID')['lon_acc_ms2']
        .rolling(window=MIN_FRAMES, min_periods=1, center=True)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return df


def find_runs(mask: np.ndarray):
    m = mask.astype(int)
    diff = np.diff(m, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return zip(starts, ends)


def extract_events(df: pd.DataFrame, threshold: float):
    events = []
    for vid, g in df.groupby('Vehicle_ID', sort=False):
        g = g.reset_index(drop=True)
        mask = g['acc_smooth'].values <= threshold
        for s, e in find_runs(mask):
            if e - s >= MIN_FRAMES:
                seg = g.iloc[s:e]
                events.append({
                    'vehicle_id': int(vid),
                    'lane_id': int(seg['Lane_ID'].iloc[0]),
                    'start_time_s': float(seg['time_s'].iloc[0]),
                    'end_time_s': float(seg['time_s'].iloc[-1]),
                    'min_acc_ms2': float(seg['acc_smooth'].min()),
                })
    return pd.DataFrame(events)


def merge_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    events = events.sort_values(['lane_id', 'start_time_s']).reset_index(drop=True)
    merged = []
    cur = None
    for _, row in events.iterrows():
        if cur is None:
            cur = row.to_dict()
            continue
        same_lane = row['lane_id'] == cur['lane_id']
        gap = row['start_time_s'] - cur['end_time_s']
        if same_lane and gap <= MERGE_GAP_S:
            cur['end_time_s'] = max(cur['end_time_s'], row['end_time_s'])
            cur['min_acc_ms2'] = min(cur['min_acc_ms2'], row['min_acc_ms2'])
        else:
            merged.append(cur)
            cur = row.to_dict()
    if cur is not None:
        merged.append(cur)
    return pd.DataFrame(merged)


def count_for_threshold(path: Path, threshold: float) -> dict:
    df = load_smoothed(path)
    raw = extract_events(df, threshold)
    merged = merge_events(raw)
    return {
        'file': path.name,
        'threshold_ms2': threshold,
        'raw_episodes': len(raw),
        'independent_events': len(merged),
    }


def main():
    files = sorted([
        p for p in RAW.rglob('vehicle-trajectory-data/**/*.csv')
        if 'RECONSTRUCTED' not in str(p).upper() and 'detector' not in str(p).lower()
    ])

    rows = []
    for threshold in THRESHOLDS:
        t0 = time.time()
        total_raw = 0
        total_merged = 0
        for path in files:
            res = count_for_threshold(path, threshold)
            total_raw += res['raw_episodes']
            total_merged += res['independent_events']
            rows.append(res)
        print(f"threshold={threshold}: raw={total_raw}, independent={total_merged}, time={time.time()-t0:.1f}s")

    summary = {
        'thresholds': THRESHOLDS,
        'per_file': rows,
        'totals': {
            str(t): sum(r['independent_events'] for r in rows if r['threshold_ms2'] == t)
            for t in THRESHOLDS
        },
    }

    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['file', 'threshold_ms2', 'raw_episodes', 'independent_events'])
        writer.writeheader()
        writer.writerows(rows)

    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print('\nTotals:')
    for t in THRESHOLDS:
        print(f"  {t:>5.1f} m/s²: {summary['totals'][str(t)]} independent events")
    print(f'\nSaved: {OUT_CSV}')
    print(f'Saved: {OUT_JSON}')


if __name__ == '__main__':
    main()
