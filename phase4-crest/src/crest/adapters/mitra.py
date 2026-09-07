"""MiTra adapter to canonical CREST schema."""
from pathlib import Path

import pandas as pd

from .schema import CANONICAL_COLUMNS, assert_canonical

KM_H_TO_M_S = 1.0 / 3.6


def _map_vehicle_type(value: str) -> str:
    value = str(value).strip().lower()
    if value in {'car'}:
        return 'car'
    if value in {'motorcycle'}:
        return 'motorcycle'
    if value in {'truck'}:
        return 'truck'
    if value in {'heavy vehicle'}:
        return 'heavy_vehicle'
    if value in {'medium vehicle'}:
        return 'medium_vehicle'
    if value in {'bus'}:
        return 'bus'
    return 'unknown'


def load_mitra_csv(path: Path, session: str, exclude_vehicle_ids=None) -> pd.DataFrame:
    """Load a MiTra T*_DAll.csv file and convert to canonical schema.

    Parameters
    ----------
    path : Path
        Path to the DAll CSV file.
    session : str
        Session tag, e.g. 'mitra_t6'.
    exclude_vehicle_ids : set or None
        Vehicle IDs to drop (e.g., teleporting jumps).
    """
    df = pd.read_csv(path, low_memory=False)

    if exclude_vehicle_ids is not None and len(exclude_vehicle_ids) > 0:
        df = df[~df['Vehicle_ID'].isin(exclude_vehicle_ids)]

    df_out = pd.DataFrame({
        'vehicle_id': df['Vehicle_ID'].astype('int64'),
        'dataset_source': session,
        'time_s': df['Time [s]'].astype('float64'),
        'x_m': df['x [m]'].astype('float64'),
        'y_m': df['y [m]'].astype('float64'),
        'speed_ms': df['Speed [km/h]'] * KM_H_TO_M_S,
        'lon_acc_ms2': df['Lon. Acc. [ms-2]'].astype('float64'),
        'lat_acc_ms2': df['Lat. Acc. [ms-2]'].astype('float64'),
        'lane_id': df['Lane'].astype('int64'),
        'vehicle_type': df['Vehicle_type'].apply(_map_vehicle_type),
        'leader_id': df['Leader_ID'].fillna(-1).astype('int64'),
        'follower_id': df['Follower_ID'].fillna(-1).astype('int64'),
        'vehicle_length_m': df['Vehicle_length [m]'].astype('float64'),
    })

    df_out = df_out[CANONICAL_COLUMNS]
    return assert_canonical(df_out)


def load_exclusion_ids(path: Path, column: str = 'Vehicle_ID') -> set:
    """Load a CSV of excluded vehicle IDs and return a set of ints."""
    if not path.exists():
        return set()
    df = pd.read_csv(path, low_memory=False)
    if column not in df.columns:
        return set()
    return set(df[column].dropna().astype(int).unique())
