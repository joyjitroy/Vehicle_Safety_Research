"""Build frozen CREST labels from NGSIM and MiTra T4-T9.

Outputs:
- outputs/results/label_events.csv
- outputs/results/label_windows.csv
- outputs/results/label_summary.json
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from crest.adapters.mitra import load_exclusion_ids, load_mitra_csv
from crest.adapters.ngsim import load_ngsim_csv
from crest.labels.event_extraction import extract_hard_brake_events, extract_queue_events

OUT_DIR = ROOT / 'outputs' / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)
EVENTS_CSV = OUT_DIR / 'label_events.csv'
WINDOWS_CSV = OUT_DIR / 'label_windows.csv'
SUMMARY_JSON = OUT_DIR / 'label_summary.json'

NGSIM_RAW = ROOT / 'data' / 'raw' / 'ngsim'
MITRA_RAW = ROOT / 'data' / 'raw' / 'mitra'
MITRA_EXCLUSION = MITRA_RAW / 'Teleporting_jumps_Dall.csv'

HORIZON_S = 10.0
HARD_BRAKE_ACC_MS2 = -3.9
HARD_BRAKE_MIN_DURATION_S = 0.5
QUEUE_SPEED_MS = 2.0
QUEUE_MIN_VEHICLES = 3
QUEUE_MAX_GAP_M = 5.0
QUEUE_MIN_DURATION_S = 5.0


def load_config(path: Path = ROOT / 'configs' / 'base.yaml'):
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def process_ngsim_file(path: Path, location: str, period: str, cfg: dict) -> pd.DataFrame:
    print(f'  Loading {path.relative_to(ROOT)} ...', flush=True)
    df = load_ngsim_csv(path, location, period)
    fps = cfg['sampling']['ngsim_hz']
    print(f'  Extracting queue events ...', flush=True)
    q = extract_queue_events(
        df,
        speed_threshold_ms=QUEUE_SPEED_MS,
        min_vehicles=QUEUE_MIN_VEHICLES,
        max_gap_m=QUEUE_MAX_GAP_M,
        min_duration_s=QUEUE_MIN_DURATION_S,
        fps=fps,
        merge_gap_s=cfg['merge']['temporal_gap_s'],
    )
    q['dataset_source'] = f'{location}_{period}'
    q['source_file'] = str(path.relative_to(ROOT))

    print(f'  Extracting hard-brake events ...', flush=True)
    hb = extract_hard_brake_events(
        df,
        threshold_ms2=HARD_BRAKE_ACC_MS2,
        min_duration_s=HARD_BRAKE_MIN_DURATION_S,
        fps=fps,
    )
    hb['dataset_source'] = f'{location}_{period}'
    hb['source_file'] = str(path.relative_to(ROOT))

    events = pd.concat([q, hb], ignore_index=True)
    return events


def process_mitra_session(session: str, cfg: dict, exclude_ids: set) -> pd.DataFrame:
    suffix = session.split('_')[-1]  # Data_T6 -> T6
    path = MITRA_RAW / session / f'{suffix}_DAll.csv'  # Data_T6 -> T6_DAll.csv
    print(f'Processing {session} ...', flush=True)
    print(f'  Loading {path.relative_to(ROOT)} ...', flush=True)
    df = load_mitra_csv(path, session.replace('Data_', 'mitra_').lower(), exclude_vehicle_ids=exclude_ids)
    fps = cfg['sampling']['mitra_hz']
    print(f'  Extracting queue events ...', flush=True)
    q = extract_queue_events(
        df,
        speed_threshold_ms=QUEUE_SPEED_MS,
        min_vehicles=QUEUE_MIN_VEHICLES,
        max_gap_m=QUEUE_MAX_GAP_M,
        min_duration_s=QUEUE_MIN_DURATION_S,
        fps=fps,
        merge_gap_s=cfg['merge']['temporal_gap_s'],
    )
    q['dataset_source'] = session.replace('Data_', 'mitra_').lower()
    q['source_file'] = str(path.relative_to(ROOT))

    print(f'  Extracting hard-brake events ...', flush=True)
    hb = extract_hard_brake_events(
        df,
        threshold_ms2=HARD_BRAKE_ACC_MS2,
        min_duration_s=HARD_BRAKE_MIN_DURATION_S,
        fps=fps,
    )
    hb['dataset_source'] = session.replace('Data_', 'mitra_').lower()
    hb['source_file'] = str(path.relative_to(ROOT))

    events = pd.concat([q, hb], ignore_index=True)
    return events


def build_windows(events: pd.DataFrame, horizon_s: float) -> pd.DataFrame:
    windows = []
    for i, row in events.iterrows():
        onset = row['start_time_s']
        windows.append({
            'event_id': i,
            'dataset_source': row['dataset_source'],
            'source_file': row['source_file'],
            'event_type': row['type'],
            'lane_id': row.get('lane_id', -1),
            'event_onset_s': onset,
            'window_start_s': onset - horizon_s,
            'window_end_s': onset,
            'event_duration_s': row['duration_s'],
        })
    return pd.DataFrame(windows)


def main():
    cfg = load_config()
    exclude_ids = load_exclusion_ids(MITRA_EXCLUSION)
    if exclude_ids:
        print(f'Excluding {len(exclude_ids)} MiTra vehicle IDs.')

    all_events = []

    # NGSIM US-101 and I-80 (training corpus and internal holdout)
    for path in sorted(NGSIM_RAW.rglob('vehicle-trajectory-data/**/*.csv')):
        if 'RECONSTRUCTED' in str(path).upper() or 'detector' in str(path).lower():
            continue
        # path like .../ngsim/US-101/vehicle-trajectory-data/0750am-0805am/trajectories-0750am-0805am.csv
        parts = path.relative_to(NGSIM_RAW).parts
        location = f'ngsim_{parts[0].lower()}'
        period = parts[2]
        t0 = time.time()
        ev = process_ngsim_file(path, location, period, cfg)
        print(f'  -> {len(ev)} events ({len(ev[ev["type"]=="queue"])} queue, {len(ev[ev["type"]=="hard_brake"])} hard_brake) in {time.time()-t0:.1f}s\n', flush=True)
        all_events.append(ev)

    # MiTra T4-T9
    for session in [f'Data_T{i}' for i in range(4, 10)]:
        t0 = time.time()
        ev = process_mitra_session(session, cfg, exclude_ids)
        print(f'  -> {len(ev)} events ({len(ev[ev["type"]=="queue"])} queue, {len(ev[ev["type"]=="hard_brake"])} hard_brake) in {time.time()-t0:.1f}s\n', flush=True)
        all_events.append(ev)

    events = pd.concat(all_events, ignore_index=True)
    events.insert(0, 'event_id', events.index)
    windows = build_windows(events, HORIZON_S)

    events.to_csv(EVENTS_CSV, index=False)
    windows.to_csv(WINDOWS_CSV, index=False)

    summary = {
        'total_events': len(events),
        'queue_events': int((events['type'] == 'queue').sum()),
        'hard_brake_events': int((events['type'] == 'hard_brake').sum()),
        'prediction_horizon_s': HORIZON_S,
        'total_positive_windows': len(windows),
        'by_dataset': events.groupby('dataset_source').size().to_dict(),
    }
    with open(SUMMARY_JSON, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print('Done.')
    print(f'  Total events: {summary["total_events"]}')
    print(f'  Queue events: {summary["queue_events"]}')
    print(f'  Hard-brake events: {summary["hard_brake_events"]}')
    print(f'  Positive windows: {summary["total_positive_windows"]}')
    print(f'  Events CSV: {EVENTS_CSV}')
    print(f'  Windows CSV: {WINDOWS_CSV}')
    print(f'  Summary JSON: {SUMMARY_JSON}')


if __name__ == '__main__':
    main()
