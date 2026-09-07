"""Fast NGSIM hard-braking recount using vectorized 5-frame rolling mean.

Label: smoothed lon_acc <= -3.9 m/s^2 for >= 5 consecutive frames (0.5 s at 10 Hz).
Events within 10 s and the same lane are merged into independent episodes.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw' / 'ngsim'
OUT = ROOT / 'outputs' / 'results' / 'ngsim_hard_brake_quick.json'
OUT.parent.mkdir(parents=True, exist_ok=True)

FT_TO_M = 0.3048
HARD_ACC_MS2 = -3.9
MIN_FRAMES = 5
MERGE_GAP_S = 10.0


def extract_hard_brake_events(df: pd.DataFrame):
    df = df.copy()
    df['lon_acc_ms2'] = df['v_Acc'] * FT_TO_M
    df['time_s'] = df['Global_Time'] / 1000.0
    df = df.sort_values(['Vehicle_ID', 'Frame_ID']).reset_index(drop=True)

    # 5-frame centered moving average per vehicle
    df['acc_smooth'] = (
        df.groupby('Vehicle_ID')['lon_acc_ms2']
        .rolling(window=MIN_FRAMES, min_periods=1, center=True)
        .mean()
        .reset_index(level=0, drop=True)
    )

    mask = df['acc_smooth'].values <= HARD_ACC_MS2
    vid = df['Vehicle_ID'].values
    frame = df['Frame_ID'].values

    # Identify run starts/ends where mask is True
    mask_int = mask.astype(int)
    diff = np.diff(mask_int, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    events = []
    for s, e in zip(starts, ends):
        length = e - s
        if length >= MIN_FRAMES:
            seg = df.iloc[s:e]
            events.append({
                'vehicle_id': int(seg['Vehicle_ID'].iloc[0]),
                'lane_id': int(seg['Lane_ID'].iloc[0]),
                'start_time_s': float(seg['time_s'].iloc[0]),
                'end_time_s': float(seg['time_s'].iloc[-1]),
                'duration_s': float(seg['time_s'].iloc[-1] - seg['time_s'].iloc[0]),
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
            cur['duration_s'] = cur['end_time_s'] - cur['start_time_s']
            cur['min_acc_ms2'] = min(cur['min_acc_ms2'], row['min_acc_ms2'])
        else:
            merged.append(cur)
            cur = row.to_dict()
    if cur is not None:
        merged.append(cur)
    return pd.DataFrame(merged)


def audit_file(path: Path):
    t0 = time.time()
    df = pd.read_csv(path, low_memory=False)
    raw = extract_hard_brake_events(df)
    merged = merge_events(raw)
    elapsed = time.time() - t0
    print(f"{path.name}: raw={len(raw)}, merged={len(merged)}, time={elapsed:.1f}s")
    return {
        'file': str(path.relative_to(RAW)),
        'rows': len(df),
        'raw_episodes': len(raw),
        'independent_events': len(merged),
    }


def main():
    files = sorted([
        p for p in RAW.rglob('vehicle-trajectory-data/**/*.csv')
        if 'RECONSTRUCTED' not in str(p).upper() and 'detector' not in str(p).lower()
    ])
    results = [audit_file(p) for p in files]
    summary = {
        'files': len(results),
        'total_independent_hard_brake_events': sum(r['independent_events'] for r in results),
        'per_file': results,
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"\nTotal independent hard-brake events: {summary['total_independent_hard_brake_events']}")
    print(f"Saved: {OUT}")


if __name__ == '__main__':
    main()
