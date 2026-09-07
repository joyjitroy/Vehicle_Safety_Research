"""Canonical trajectory schema for CREST."""
from typing import List

CANONICAL_COLUMNS: List[str] = [
    'vehicle_id',
    'dataset_source',
    'time_s',
    'x_m',
    'y_m',
    'speed_ms',
    'lon_acc_ms2',
    'lat_acc_ms2',
    'lane_id',
    'vehicle_type',
    'leader_id',
    'follower_id',
    'vehicle_length_m',
]

CANONICAL_DTYPES = {
    'vehicle_id': 'int64',
    'dataset_source': 'object',
    'time_s': 'float64',
    'x_m': 'float64',
    'y_m': 'float64',
    'speed_ms': 'float64',
    'lon_acc_ms2': 'float64',
    'lat_acc_ms2': 'float64',
    'lane_id': 'int64',
    'vehicle_type': 'object',
    'leader_id': 'int64',
    'follower_id': 'int64',
    'vehicle_length_m': 'float64',
}

VEHICLE_TYPE_CATEGORIES = {'motorcycle', 'car', 'truck', 'bus', 'medium_vehicle', 'heavy_vehicle'}


def assert_canonical(df):
    missing = set(CANONICAL_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Canonical DataFrame missing columns: {missing}")
    return df
