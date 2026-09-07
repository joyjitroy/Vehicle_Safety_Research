"""Prediction-horizon T sensitivity on NGSIM labels.

For each horizon T in {5, 10, 15, 20}s, define a positive prediction interval
[onset - T, onset] for each independent hazard event. Merge overlapping intervals.
Report the number of independent positive-label windows per horizon.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVENT_JSON = ROOT / 'outputs' / 'results' / 'ngsim_p0_audit_summary.json'
OUT_CSV = ROOT / 'outputs' / 'results' / 'ngsim_label_horizon_sensitivity.csv'
OUT_JSON = ROOT / 'outputs' / 'results' / 'ngsim_label_horizon_sensitivity.json'

HORIZONS = [5, 10, 15, 20]


def load_events(path: Path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    events = []
    for r in data.get('per_file', []):
        for ev in r.get('hard_brake_details', []):
            events.append({'onset_s': ev['start_time_s'], 'kind': 'hard_brake'})
        for ev in r.get('queue_details', []):
            events.append({'onset_s': ev['start_time_s'], 'kind': 'queue'})
    return pd.DataFrame(events)


def merge_intervals(intervals):
    """Merge sorted (start, end) intervals. Overlap or touching intervals merged."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def count_independent_windows(events: pd.DataFrame, horizon: int) -> dict:
    intervals = [(row['onset_s'] - horizon, row['onset_s']) for _, row in events.iterrows()]
    merged = merge_intervals(intervals)
    return {
        'horizon_s': horizon,
        'num_events': len(events),
        'independent_positive_windows': len(merged),
    }


def main():
    events = load_events(EVENT_JSON)
    print(f'Loaded {len(events)} independent NGSIM events ({len(events[events["kind"] == "queue"])} queue, {len(events[events["kind"] == "hard_brake"])} hard brake).')

    rows = [count_independent_windows(events, t) for t in HORIZONS]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    df.to_csv(OUT_CSV, index=False)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2)

    print(f'\nSaved: {OUT_CSV}')
    print(f'Saved: {OUT_JSON}')


if __name__ == '__main__':
    main()
