"""Designate the calibration split: last 20% of NGSIM US-101 by time + last 20% of MiTra T6 by time."""
import json
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT_YAML = ROOT / 'configs' / 'splits.yaml'
OUT_JSON = ROOT / 'outputs' / 'results' / 'calibration_split_info.json'
OUT_YAML.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

NGSIM_US101_DIR = ROOT / 'data' / 'raw' / 'ngsim' / 'US-101' / 'vehicle-trajectory-data'
MITRA_T6 = ROOT / 'data' / 'raw' / 'mitra' / 'Data_T6' / 'T6_DAll.csv'
CALIBRATION_FRACTION = 0.20


def ngsim_time_range(paths):
    t_min = float('inf')
    t_max = float('-inf')
    for p in paths:
        df = pd.read_csv(p, usecols=['Global_Time'], low_memory=False)
        if df.empty:
            continue
        t_min = min(t_min, df['Global_Time'].min())
        t_max = max(t_max, df['Global_Time'].max())
    return t_min, t_max


def mitra_time_range(path):
    df = pd.read_csv(path, usecols=['Time [s]'], low_memory=False)
    return df['Time [s]'].min(), df['Time [s]'].max()


def main():
    ngsim_paths = sorted(NGSIM_US101_DIR.rglob('trajectories-*.csv'))
    ngsim_paths = [p for p in ngsim_paths if 'RECONSTRUCTED' not in str(p).upper()]
    ngsim_t0, ngsim_t1 = ngsim_time_range(ngsim_paths)
    ngsim_duration_ms = ngsim_t1 - ngsim_t0
    ngsim_cal_start_ms = ngsim_t1 - CALIBRATION_FRACTION * ngsim_duration_ms

    mitra_t0, mitra_t1 = mitra_time_range(MITRA_T6)
    mitra_duration_s = mitra_t1 - mitra_t0
    mitra_cal_start_s = mitra_t1 - CALIBRATION_FRACTION * mitra_duration_s

    splits = {
        'calibration': {
            'description': 'Last 20% by time; reserved for Platt scaling only.',
            'ngsim_us101': {
                'time_unit': 'ms',
                'global_time_min': int(ngsim_t0),
                'global_time_max': int(ngsim_t1),
                'calibration_start_global_time': int(ngsim_cal_start_ms),
            },
            'mitra_t6': {
                'time_unit': 's',
                'time_min': float(mitra_t0),
                'time_max': float(mitra_t1),
                'calibration_start_time': float(mitra_cal_start_s),
            },
        },
        'train': {
            'ngsim_us101': 'global_time < calibration_start_global_time',
            'mitra_t6': 'time_s < calibration_start_time',
        },
    }

    with open(OUT_YAML, 'w', encoding='utf-8') as f:
        yaml.safe_dump(splits, f, sort_keys=False)

    info = {
        'calibration_fraction': CALIBRATION_FRACTION,
        'ngsim_us101': {
            'duration_ms': int(ngsim_duration_ms),
            'calibration_start_global_time': int(ngsim_cal_start_ms),
            'train_end_global_time': int(ngsim_cal_start_ms) - 1,
        },
        'mitra_t6': {
            'duration_s': float(mitra_duration_s),
            'calibration_start_time_s': float(mitra_cal_start_s),
            'train_end_time_s': float(mitra_cal_start_s) - 0.001,
        },
    }
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)

    print('Calibration split designated.')
    print(f'  NGSIM US-101: {info["ngsim_us101"]["duration_ms"]} ms total; calibration starts at {info["ngsim_us101"]["calibration_start_global_time"]} ms')
    print(f'  MiTra T6: {info["mitra_t6"]["duration_s"]:.1f} s total; calibration starts at {info["mitra_t6"]["calibration_start_time_s"]:.3f} s')
    print(f'\nSaved: {OUT_YAML}')
    print(f'Saved: {OUT_JSON}')


if __name__ == '__main__':
    main()
