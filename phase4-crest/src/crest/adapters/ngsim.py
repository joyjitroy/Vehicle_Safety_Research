"""NGSIM adapter to canonical CREST schema."""
from pathlib import Path

import pandas as pd

from .schema import CANONICAL_COLUMNS, assert_canonical

FT_TO_M = 0.3048


def _map_vehicle_type(vclass: int) -> str:
    # NGSIM v_Class: 1=motorcycle, 2=car, 3=truck
    mapping = {1: 'motorcycle', 2: 'car', 3: 'truck'}
    return mapping.get(int(vclass), 'unknown')


def load_ngsim_csv(path: Path, location: str, period: str) -> pd.DataFrame:
    """Load an NGSIM trajectory CSV and convert to canonical schema.

    Parameters
    ----------
    path : Path
        Path to the NGSIM trajectory CSV.
    location : str
        Dataset location tag, e.g. 'ngsim_us101' or 'ngsim_i80'.
    period : str
        Time-period tag, e.g. '0750am-0805am'.
    """
    df = pd.read_csv(path, low_memory=False)

    # NGSIM files vary in the spelling of the predecessor column.
    preceding_col = 'Preceeding' if 'Preceeding' in df.columns else 'Preceding'

    # Convert units and rename
    df_out = pd.DataFrame({
        'vehicle_id': df['Vehicle_ID'].astype('int64'),
        'dataset_source': f"{location}_{period}",
        'time_s': df['Global_Time'] / 1000.0,
        'x_m': df['Local_X'] * FT_TO_M,
        'y_m': df['Local_Y'] * FT_TO_M,
        'speed_ms': df['v_Vel'] * FT_TO_M,
        'lon_acc_ms2': df['v_Acc'] * FT_TO_M,
        'lat_acc_ms2': pd.NA,  # NGSIM does not provide lateral acceleration
        'lane_id': df['Lane_ID'].astype('int64'),
        'vehicle_type': df['v_Class'].apply(_map_vehicle_type),
        'leader_id': df[preceding_col].fillna(0).astype('int64'),
        'follower_id': df['Following'].fillna(0).astype('int64'),
        'vehicle_length_m': df['v_Length'] * FT_TO_M,
    })

    df_out = df_out[CANONICAL_COLUMNS]
    return assert_canonical(df_out)
