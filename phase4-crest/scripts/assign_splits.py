"""Assign train / calibration / holdout splits to label events and windows."""
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

EVENTS_IN = ROOT / 'outputs' / 'results' / 'label_events.csv'
WINDOWS_IN = ROOT / 'outputs' / 'results' / 'label_windows.csv'
SPLITS_YAML = ROOT / 'configs' / 'splits.yaml'

EVENTS_OUT = ROOT / 'outputs' / 'results' / 'label_events_split.csv'
WINDOWS_OUT = ROOT / 'outputs' / 'results' / 'label_windows_split.csv'


def load_splits(path: Path):
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def assign_split(row: pd.Series, splits: dict, time_col: str = 'start_time_s') -> str:
    src: str = row['dataset_source']

    # ── Holdout: NGSIM I-80 (cross-site US) + MiTra T9 (cross-session European)
    if src.startswith('ngsim_i-80'):
        return 'holdout'
    if src == 'mitra_t9':
        return 'holdout'

    # ── Calibration: MiTra T8 + last 30% of NGSIM US-101 by time
    if src == 'mitra_t8':
        return 'cal'
    if src.startswith('ngsim_us-101'):
        t = float(row[time_col])
        cal_start = splits['calibration']['ngsim_us101']['calibration_start_global_time']
        return 'cal' if t >= cal_start else 'train'

    # ── Train: MiTra T4-T7 + NGSIM US-101 early portion (handled above)
    if src in {'mitra_t4', 'mitra_t5', 'mitra_t6', 'mitra_t7'}:
        return 'train'

    return 'unknown'


def main():
    splits = load_splits(SPLITS_YAML)

    events = pd.read_csv(EVENTS_IN)
    events['split'] = events.apply(lambda row: assign_split(row, splits, time_col='start_time_s'), axis=1)
    events.to_csv(EVENTS_OUT, index=False)

    windows = pd.read_csv(WINDOWS_IN)
    windows['split'] = windows.apply(lambda row: assign_split(row, splits, time_col='window_start_s'), axis=1)
    windows.to_csv(WINDOWS_OUT, index=False)

    print('Split counts:')
    print(events['split'].value_counts().to_string())
    print(f'\nSaved: {EVENTS_OUT}')
    print(f'Saved: {WINDOWS_OUT}')


if __name__ == '__main__':
    main()
