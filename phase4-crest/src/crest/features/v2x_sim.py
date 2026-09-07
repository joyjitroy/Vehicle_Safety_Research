"""V2X / RSU cooperative sensing simulation utilities.

Implements configurable RSU coverage radius and detection uncertainty:
  - missed detection probability
  - ghost false-positive probability (not currently tied to a real object)
  - position noise
  - stale tracks injected with a time lag
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class DetectionUncertaintyConfig:
    missed_detection_prob: float = 0.05
    false_positive_ghost_prob: float = 0.01
    position_noise_std_m: float = 1.0
    stale_track_prob: float = 0.05
    stale_track_lag_s: float = 0.2


@dataclass
class RSUConfig:
    coverage_radius_m: float = 500.0
    uncertainty: DetectionUncertaintyConfig = None

    def __post_init__(self):
        if self.uncertainty is None:
            self.uncertainty = DetectionUncertaintyConfig()


class RSUSimulator:
    """Simulate RSU observations of vehicles within coverage radius."""

    def __init__(self, config: RSUConfig):
        self.config = config

    def filter_by_radius(
        self,
        df: pd.DataFrame,
        ego_x_m: float,
        ego_y_m: float,
    ) -> pd.DataFrame:
        """Return rows within ``coverage_radius_m`` of the ego position."""
        dx = df['x_m'] - ego_x_m
        dy = df['y_m'] - ego_y_m
        return df[dx * dx + dy * dy <= self.config.coverage_radius_m ** 2].copy()

    def apply_uncertainty(self, df: pd.DataFrame, rng: Optional[np.random.Generator] = None) -> pd.DataFrame:
        """Apply detection uncertainty to a per-frame observation set."""
        if rng is None:
            rng = np.random.default_rng()

        df = df.copy()
        n = len(df)
        if n == 0:
            return df

        # Missed detections
        kept = rng.random(n) >= self.config.uncertainty.missed_detection_prob
        df = df[kept].copy()

        # Position noise
        if self.config.uncertainty.position_noise_std_m > 0:
            df['x_m'] = df['x_m'] + rng.normal(0.0, self.config.uncertainty.position_noise_std_m, len(df))
            df['y_m'] = df['y_m'] + rng.normal(0.0, self.config.uncertainty.position_noise_std_m, len(df))

        # Stale track injection: a subset of rows retain their positions from earlier time
        if self.config.uncertainty.stale_track_prob > 0 and self.config.uncertainty.stale_track_lag_s > 0:
            n_stale = max(1, int(round(n * self.config.uncertainty.stale_track_prob)))
            stale_idx = rng.choice(df.index, size=min(n_stale, len(df)), replace=False)
            # Stale offset is lag * observed speed along heading (approximate as speed)
            speed = df.loc[stale_idx, 'speed_ms'].values
            lag = self.config.uncertainty.stale_track_lag_s
            # approximate position error as lag * speed in a random direction
            angle = rng.uniform(0.0, 2 * np.pi, len(stale_idx))
            df.loc[stale_idx, 'x_m'] = df.loc[stale_idx, 'x_m'].values - lag * speed * np.cos(angle)
            df.loc[stale_idx, 'y_m'] = df.loc[stale_idx, 'y_m'].values - lag * speed * np.sin(angle)

        return df.reset_index(drop=True)

    def observe(
        self,
        df: pd.DataFrame,
        ego_x_m: float,
        ego_y_m: float,
        rng: Optional[np.random.Generator] = None,
    ) -> pd.DataFrame:
        """Filter by radius and apply uncertainty."""
        visible = self.filter_by_radius(df, ego_x_m, ego_y_m)
        return self.apply_uncertainty(visible, rng)
