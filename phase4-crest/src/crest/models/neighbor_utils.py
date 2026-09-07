"""Utility functions for neighbor selection in CREST V2X feature construction."""
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def topk_neighbors(
    scene_df: pd.DataFrame,
    ego_vehicle_id: int,
    top_k: int,
    max_radius_m: Optional[float] = None,
    ego_x_col: str = 'x_m',
    ego_y_col: str = 'y_m',
    vehicle_id_col: str = 'vehicle_id',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return top-K nearest neighbors for each vehicle in scene_df.

    Parameters
    ----------
    scene_df : pd.DataFrame
        Single-timeframe scene with canonical columns.
    ego_vehicle_id : int
        Vehicle whose neighbors are selected (kept for API consistency).
    top_k : int
        Number of neighbors to return.
    max_radius_m : float or None
        If given, neighbors beyond this radius are excluded.

    Returns
    -------
    neighbor_ids : ndarray of shape (n_vehicles, top_k)
    distances : ndarray of shape (n_vehicles, top_k)
    mask : ndarray of shape (n_vehicles, top_k)
        True for valid neighbor positions.
    """
    coords = scene_df[[ego_x_col, ego_y_col]].values
    vehicle_ids = scene_df[vehicle_id_col].values
    n = len(coords)

    diffs = coords[:, None, :] - coords[None, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=2))
    # self-distance = inf so a vehicle is never its own neighbor
    np.fill_diagonal(dists, np.inf)

    if max_radius_m is not None:
        dists = np.where(dists <= max_radius_m, dists, np.inf)

    # sort and take top_k
    order = np.argsort(dists, axis=1)[:, :top_k]
    neighbor_ids = vehicle_ids[order]
    distances = np.take_along_axis(dists, order, axis=1)
    mask = np.isfinite(distances)

    return neighbor_ids, distances, mask
