"""Verify coordinate and unit conversions for NGSIM and MiTra."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import pandas as pd
from crest.adapters.mitra import load_mitra_csv
from crest.adapters.ngsim import load_ngsim_csv

NGSIM_PATH = ROOT / 'data' / 'raw' / 'ngsim' / 'US-101' / 'vehicle-trajectory-data' / '0750am-0805am' / 'trajectories-0750am-0805am.csv'
MITRA_PATH = ROOT / 'data' / 'raw' / 'mitra' / 'Data_T9' / 'T9_DAll.csv'

FT_TO_M = 0.3048


def verify_ngsim():
    raw = pd.read_csv(NGSIM_PATH, nrows=1)
    canonical = load_ngsim_csv(NGSIM_PATH, 'ngsim_us101', '0750am-0805am').head(1)
    row_raw = raw.iloc[0]
    row_can = canonical.iloc[0]

    checks = {
        'x_m (Local_X * 0.3048)': (row_raw['Local_X'] * FT_TO_M, row_can['x_m']),
        'y_m (Local_Y * 0.3048)': (row_raw['Local_Y'] * FT_TO_M, row_can['y_m']),
        'speed_ms (v_Vel * 0.3048)': (row_raw['v_Vel'] * FT_TO_M, row_can['speed_ms']),
        'lon_acc_ms2 (v_Acc * 0.3048)': (row_raw['v_Acc'] * FT_TO_M, row_can['lon_acc_ms2']),
        'time_s (Global_Time / 1000)': (row_raw['Global_Time'] / 1000.0, row_can['time_s']),
    }
    print('NGSIM conversion checks:')
    for name, (expected, actual) in checks.items():
        ok = abs(expected - actual) < 1e-6
        print(f'  {name}: expected={expected:.6f}, actual={actual:.6f}, ok={ok}')
        if not ok:
            raise AssertionError(f'NGSIM conversion failed: {name}')
    print('NGSIM: all conversions OK\n')


def verify_mitra():
    raw = pd.read_csv(MITRA_PATH, nrows=1)
    canonical = load_mitra_csv(MITRA_PATH, 'mitra_t9').head(1)
    row_raw = raw.iloc[0]
    row_can = canonical.iloc[0]

    checks = {
        'x_m (UTM meters, direct use)': (row_raw['x [m]'], row_can['x_m']),
        'y_m (UTM meters, direct use)': (row_raw['y [m]'], row_can['y_m']),
        'speed_ms (km/h / 3.6)': (row_raw['Speed [km/h]'] / 3.6, row_can['speed_ms']),
        'lon_acc_ms2 (direct use)': (row_raw['Lon. Acc. [ms-2]'], row_can['lon_acc_ms2']),
        'time_s (direct use)': (row_raw['Time [s]'], row_can['time_s']),
    }
    print('MiTra conversion checks:')
    for name, (expected, actual) in checks.items():
        ok = abs(expected - actual) < 1e-6
        print(f'  {name}: expected={expected:.6f}, actual={actual:.6f}, ok={ok}')
        if not ok:
            raise AssertionError(f'MiTra conversion failed: {name}')
    print('MiTra: all conversions OK\n')


def main():
    verify_ngsim()
    verify_mitra()
    print('Checkpoint #6: coordinate conversion verification passed.')


if __name__ == '__main__':
    main()
