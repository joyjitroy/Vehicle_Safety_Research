"""Synthetic map/elevation feature generator for ablation A1.

NGSIM and MiTra raw trajectory data do not contain real elevation, grade,
curvature, or ramp-proximity information (both are flat urban freeway
segments captured by fixed cameras/drones). Since A1 evaluates the value of
adding map/elevation context to the ego-only B1 baseline, we generate a
deterministic synthetic road-context profile as a function of longitudinal
position (x_m) and dataset source. The profile is fixed (seeded per source)
so that every snapshot at the same x_m always yields the same map features,
which is required for interpretable ablation results.

Feature vector (8-dim), matching configs/model.yaml map_encoder.input_dim=8:
    0: elevation_m       - synthetic sinusoidal elevation profile
    1: grade_pct         - analytic derivative of the elevation profile (%)
    2: curvature_1_per_m - synthetic secondary sinusoid (low magnitude)
    3: is_curve_flag     - 1.0 if abs(curvature) exceeds a fixed threshold
    4: num_lanes         - lane count observed in the source (approx, static)
    5: speed_limit_ms    - fixed nominal freeway speed limit for the source
    6: dist_to_ramp_m    - distance to nearest synthetic on/off-ramp marker
    7: source_type_flag  - 0.0 for NGSIM, 1.0 for MiTra
"""
import hashlib

import numpy as np

RAMP_SPACING_M = 500.0
CURVE_THRESHOLD = 0.0015


def _source_seed(source: str) -> int:
    return int(hashlib.sha256(source.encode('utf-8')).hexdigest(), 16) % (2 ** 32)


def _source_params(source: str):
    rng = np.random.default_rng(_source_seed(source))
    amplitude = rng.uniform(2.0, 6.0)       # meters
    wavelength = rng.uniform(400.0, 900.0)  # meters
    phase = rng.uniform(0.0, 2 * np.pi)
    curve_amp = rng.uniform(0.0005, 0.002)
    curve_wavelength = rng.uniform(150.0, 350.0)
    curve_phase = rng.uniform(0.0, 2 * np.pi)
    return amplitude, wavelength, phase, curve_amp, curve_wavelength, curve_phase


def _static_context(source: str):
    is_mitra = source.startswith('mitra')
    num_lanes = 4.0 if is_mitra else 5.0
    speed_limit_ms = 20.0 if is_mitra else 29.06  # ~45 mph MiTra urban, ~65 mph NGSIM freeway
    source_type_flag = 1.0 if is_mitra else 0.0
    return num_lanes, speed_limit_ms, source_type_flag


def build_map_features(x_m: float, source: str) -> np.ndarray:
    """Return an 8-dim synthetic map/elevation feature vector for one snapshot."""
    amplitude, wavelength, phase, curve_amp, curve_wavelength, curve_phase = _source_params(source)
    num_lanes, speed_limit_ms, source_type_flag = _static_context(source)

    x = float(x_m)
    k = 2 * np.pi / wavelength
    elevation_m = amplitude * np.sin(k * x + phase)
    grade_pct = 100.0 * amplitude * k * np.cos(k * x + phase)

    kc = 2 * np.pi / curve_wavelength
    curvature = curve_amp * np.sin(kc * x + curve_phase)
    is_curve_flag = 1.0 if abs(curvature) > CURVE_THRESHOLD else 0.0

    dist_to_ramp_m = float(RAMP_SPACING_M - (x % RAMP_SPACING_M))

    return np.array([
        elevation_m,
        grade_pct,
        curvature,
        is_curve_flag,
        num_lanes,
        speed_limit_ms,
        dist_to_ramp_m,
        source_type_flag,
    ], dtype=np.float32)
