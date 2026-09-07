"""Verify canonical adapters on sample rows from NGSIM and MiTra."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from crest.adapters.mitra import load_mitra_csv
from crest.adapters.ngsim import load_ngsim_csv
from crest.adapters.schema import CANONICAL_COLUMNS

NGSIM_PATH = ROOT / 'data' / 'raw' / 'ngsim' / 'US-101' / 'vehicle-trajectory-data' / '0750am-0805am' / 'trajectories-0750am-0805am.csv'
MITRA_PATH = ROOT / 'data' / 'raw' / 'mitra' / 'Data_T9' / 'T9_DAll.csv'


def main():
    print('Canonical columns:', CANONICAL_COLUMNS)
    print()

    print('NGSIM sample (US-101 0750am-0805am):')
    ngsim = load_ngsim_csv(NGSIM_PATH, 'ngsim_us101', '0750am-0805am')
    print(ngsim.head(3).to_string(index=False))
    print(f'  rows={len(ngsim)}, cols={list(ngsim.columns)}')
    print(f'  speed_ms range: {ngsim["speed_ms"].min():.2f} - {ngsim["speed_ms"].max():.2f}')
    print(f'  lon_acc_ms2 range: {ngsim["lon_acc_ms2"].min():.2f} - {ngsim["lon_acc_ms2"].max():.2f}')
    print()

    print('MiTra sample (T9_DAll):')
    mitra = load_mitra_csv(MITRA_PATH, 'mitra_t9')
    print(mitra.head(3).to_string(index=False))
    print(f'  rows={len(mitra)}, cols={list(mitra.columns)}')
    print(f'  speed_ms range: {mitra["speed_ms"].min():.2f} - {mitra["speed_ms"].max():.2f}')
    print(f'  lon_acc_ms2 range: {mitra["lon_acc_ms2"].min():.2f} - {mitra["lon_acc_ms2"].max():.2f}')
    print()

    print('Verification passed.')


if __name__ == '__main__':
    main()
