"""CREST PyTorch Dataset: positive and negative snapshots from labeled events."""
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))

from crest.adapters.mitra import load_mitra_csv
from crest.adapters.ngsim import load_ngsim_csv
from crest.features.snapshot import build_snapshot_features, get_scene_at_time

MAP_DIM = 8


def _parse_ngsim_path(source: str) -> Tuple[Path, str, str]:
    # source looks like ngsim_i-80_0500pm-0515pm or ngsim_us-101_0750am-0805am
    parts = source.split('_')
    location = '_'.join(parts[1:2])  # i-80 or us-101
    period = parts[2]
    path = (
        ROOT
        / 'data'
        / 'raw'
        / 'ngsim'
        / location.upper()
        / 'vehicle-trajectory-data'
        / period
    )
    # find the trajectory CSV inside the period folder
    csvs = list(path.glob('trajectories-*.csv'))
    csvs = [c for c in csvs if 'RECONSTRUCTED' not in str(c).upper()]
    if not csvs:
        raise FileNotFoundError(f'No trajectory CSV found for {source}')
    return csvs[0], location, period


def _parse_mitra_path(source: str) -> Tuple[Path, str]:
    # source looks like mitra_t6
    suffix = source.split('_')[-1].upper()  # t6 -> T6
    session = f'Data_{suffix}'
    path = ROOT / 'data' / 'raw' / 'mitra' / session / f'{suffix}_DAll.csv'
    return path, session


def load_source_df(source: str) -> pd.DataFrame:
    if source.startswith('ngsim_'):
        path, location, period = _parse_ngsim_path(source)
        return load_ngsim_csv(path, f'ngsim_{location}', period)
    elif source.startswith('mitra_'):
        path, session = _parse_mitra_path(source)
        return load_mitra_csv(path, source)
    raise ValueError(f'Unknown source: {source}')


class CRESTSnapshotDataset(Dataset):
    """Build snapshots for training or calibration.

    Positive examples come from labeled event windows. Negative examples are
    sampled from the same source files outside any positive window; if those
    files are fully saturated (common in dense MiTra sessions), we fall back to
    negative-only sources such as MiTra T1-T3 free-flow sessions.
    """

    NEGATIVE_ONLY_SOURCES = [f'mitra_t{i}' for i in range(1, 4)]

    def __init__(
        self,
        events_csv: Path,
        windows_csv: Path,
        split: str,
        top_k: int = 10,
        max_radius_m: float = 300.0,
        seed: int = 42,
    ):
        self.events = pd.read_csv(events_csv)
        self.windows = pd.read_csv(windows_csv)
        self.events = self.events[self.events['split'] == split].reset_index(drop=True)
        self.windows = self.windows[self.windows['split'] == split].reset_index(drop=True)
        self.split = split
        self.top_k = top_k
        self.max_radius_m = max_radius_m
        self.rng = np.random.default_rng(seed)

        self.sources = self.events['dataset_source'].unique()
        self._cache: dict = {}

        # Build a per-source list of positive windows for negative sampling
        self._positive_ranges = {src: [] for src in self.NEGATIVE_ONLY_SOURCES}
        for src in self.sources:
            src_windows = self.windows[self.windows['dataset_source'] == src]
            ranges = [(r['window_start_s'], r['window_end_s']) for _, r in src_windows.iterrows()]
            self._positive_ranges[src] = ranges

    def _load_df(self, source: str) -> pd.DataFrame:
        if source not in self._cache:
            print(f'  Loading source {source} ...', flush=True)
            self._cache[source] = load_source_df(source)
        return self._cache[source]

    def _sample_negative_in_df(self, df: pd.DataFrame, ranges: list) -> Tuple[float, pd.DataFrame]:
        t_min = df['time_s'].min()
        t_max = df['time_s'].max()
        for _ in range(1000):
            t = self.rng.uniform(t_min, t_max)
            if all(not (r[0] <= t <= r[1]) for r in ranges):
                scene = get_scene_at_time(df, t)
                if len(scene) > 0:
                    return t, scene
        return None, None

    def _sample_negative(self, source: str, df: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
        ranges = self._positive_ranges.get(source, [])
        result = self._sample_negative_in_df(df, ranges)
        if result[0] is not None:
            return result

        # Positive source is saturated; fall back to a negative-only source.
        for neg_src in self.NEGATIVE_ONLY_SOURCES:
            neg_df = self._load_df(neg_src)
            result = self._sample_negative_in_df(neg_df, self._positive_ranges[neg_src])
            if result[0] is not None:
                return result

        raise RuntimeError(f'Could not sample a valid negative time for {source} or fallbacks')

    def _choose_ego_for_event(self, scene: pd.DataFrame, event: pd.Series) -> int:
        """Choose ego vehicle for a positive event snapshot."""
        lane = int(event['lane_id'])
        lane_scene = scene[scene['lane_id'] == lane]
        if lane_scene.empty:
            lane_scene = scene
        if event['type'] == 'queue':
            # slowest vehicle in lane
            return int(lane_scene.loc[lane_scene['speed_ms'].idxmin(), 'vehicle_id'])
        else:  # hard_brake
            # most negative longitudinal acceleration in lane
            return int(lane_scene.loc[lane_scene['lon_acc_ms2'].idxmin(), 'vehicle_id'])

    def _choose_random_ego(self, scene: pd.DataFrame) -> int:
        row = scene.sample(n=1, random_state=int(self.rng.integers(0, 1_000_000))).iloc[0]
        return int(row['vehicle_id'])

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, idx: int):
        event = self.events.iloc[idx]
        source = event['dataset_source']
        df = self._load_df(source)

        # Positive snapshot at event onset
        t_pos = float(event['start_time_s'])
        scene_pos = get_scene_at_time(df, t_pos)
        ego_id_pos = self._choose_ego_for_event(scene_pos, event)
        ego_pos, neigh_pos, mask_pos = build_snapshot_features(
            scene_pos, ego_id_pos, self.top_k, self.max_radius_m
        )

        # Negative snapshot from the same source
        t_neg, scene_neg = self._sample_negative(source, df)
        ego_id_neg = self._choose_random_ego(scene_neg)
        ego_neg, neigh_neg, mask_neg = build_snapshot_features(
            scene_neg, ego_id_neg, self.top_k, self.max_radius_m
        )

        return {
            'ego_pos': torch.from_numpy(ego_pos),
            'neighbors_pos': torch.from_numpy(neigh_pos),
            'mask_pos': torch.from_numpy(mask_pos),
            'ego_neg': torch.from_numpy(ego_neg),
            'neighbors_neg': torch.from_numpy(neigh_neg),
            'mask_neg': torch.from_numpy(mask_neg),
            'label_pos': torch.tensor(1.0, dtype=torch.float32),
            'label_neg': torch.tensor(0.0, dtype=torch.float32),
        }


if __name__ == '__main__':
    events_path = ROOT / 'outputs' / 'results' / 'label_events_split.csv'
    windows_path = ROOT / 'outputs' / 'results' / 'label_windows_split.csv'
    ds = CRESTSnapshotDataset(events_path, windows_path, split='train')
    print('Train dataset size:', len(ds))
    sample = ds[0]
    for k, v in sample.items():
        print(k, v.shape if hasattr(v, 'shape') else v)
