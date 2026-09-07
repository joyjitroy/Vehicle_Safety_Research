"""Feature extraction from a single traffic snapshot for CREST."""
from typing import Tuple

import numpy as np
import pandas as pd


def ego_features(vehicle: pd.Series) -> np.ndarray:
    """Return 6-dim ego feature vector."""
    return np.array([
        float(vehicle['speed_ms']),
        float(vehicle['lon_acc_ms2']),
        float(vehicle.get('lat_acc_ms2', 0.0) if pd.notna(vehicle.get('lat_acc_ms2')) else 0.0),
        0.0,  # x relative to ego
        0.0,  # y relative to ego
        0.0,  # distance to ego
    ], dtype=np.float32)


def neighbor_features(ego: pd.Series, neighbor: pd.Series) -> np.ndarray:
    """Return 6-dim neighbor feature vector relative to ego."""
    dx = float(neighbor['x_m'] - ego['x_m'])
    dy = float(neighbor['y_m'] - ego['y_m'])
    dist = float(np.hypot(dx, dy))
    return np.array([
        float(neighbor['speed_ms']),
        float(neighbor['lon_acc_ms2']),
        float(neighbor.get('lat_acc_ms2', 0.0) if pd.notna(neighbor.get('lat_acc_ms2')) else 0.0),
        dx,
        dy,
        dist,
    ], dtype=np.float32)


def build_snapshot_features(
    scene_df: pd.DataFrame,
    ego_vehicle_id: int,
    top_k: int = 10,
    max_radius_m: float = 300.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ego features, neighbor features, and neighbor mask for one snapshot.

    Returns
    -------
    ego : np.ndarray, shape (6,)
    neighbors : np.ndarray, shape (top_k, 6)
    mask : np.ndarray, shape (top_k,)
    """
    ego_row = scene_df[scene_df['vehicle_id'] == ego_vehicle_id].iloc[0]
    ego_feat = ego_features(ego_row)
    ego_x = float(ego_row['x_m'])
    ego_y = float(ego_row['y_m'])

    others = scene_df[scene_df['vehicle_id'] != ego_vehicle_id].copy()
    if others.empty:
        return ego_feat, np.zeros((top_k, 6), dtype=np.float32), np.zeros(top_k, dtype=bool)

    others['dx'] = others['x_m'].astype(float) - ego_x
    others['dy'] = others['y_m'].astype(float) - ego_y
    others['dist'] = np.hypot(others['dx'].values, others['dy'].values)
    others = others[others['dist'] <= max_radius_m].sort_values('dist').head(top_k)

    neighbor_feats = []
    for _, row in others.iterrows():
        neighbor_feats.append(neighbor_features(ego_row, row))

    n = len(neighbor_feats)
    pad = top_k - n
    if pad > 0:
        neighbor_feats.extend([np.zeros(6, dtype=np.float32)] * pad)

    neighbors = np.stack(neighbor_feats, axis=0)
    mask = np.zeros(top_k, dtype=bool)
    mask[:n] = True
    return ego_feat, neighbors, mask


def find_closest_time(df: pd.DataFrame, target_time: float) -> float:
    """Return the actual time in the dataframe closest to the target time."""
    return df.loc[(df['time_s'] - target_time).abs().idxmin(), 'time_s']


def get_scene_at_time(df: pd.DataFrame, time_s: float, tol: float = 0.05) -> pd.DataFrame:
    """Return all rows at the timestamp closest to ``time_s``."""
    closest = find_closest_time(df, time_s)
    scene = df[df['time_s'] == closest].copy()
    if len(scene) == 0:
        # fallback to tolerance window
        scene = df[(df['time_s'] >= time_s - tol) & (df['time_s'] <= time_s + tol)].copy()
    return scene
