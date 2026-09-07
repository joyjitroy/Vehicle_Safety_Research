# CREST: Calibrated Risk Estimation with Source-Aware Temporal Fusion

Phase 4 of the Vehicle Safety Research program. CREST extends the inverse crash-probability foundation of SafeDriver-IQ and the multi-model risk engine of PRISM into cooperative freeway hazard prediction. It combines Platt-calibrated hazard probabilities with systematic, per-source ablation of simulated V2V and RSU sensing channels, and supports inference-time behavioral and perception plugins without retraining.

## Abstract

Road traffic crashes cause approximately 1.19 million deaths globally each year, including over 39,000 in the United States in 2024. Freeway rear-end collisions, often caused by sudden queue formation and hard braking, are the focus of this work. Most hazard-prediction models rely solely on the vehicle's own onboard state (ego-only sensing) or produce uncalibrated risk scores that do not support actionable alert thresholds. Prior work using cooperative sensing, such as vehicle-to-vehicle (V2V) or roadside unit (RSU) communication, has not isolated the individual contribution of each source to prediction accuracy.

CREST introduces a framework that combines Platt-calibrated hazard probabilities with systematic per-source ablation across simulated V2V and RSU sensing channels, modeled on current communication standards. CREST also supports adding behavioral and perception plugins at inference time without retraining. Evaluated on real freeway trajectory data, the most effective individual cooperative source increases AUPRC to 0.751 versus 0.678 for ego-only sensing. Combining all sources without a behavioral prior reduces performance to 0.700, while adding a pre-computed behavioral risk score at inference raises AUPRC to 0.808, the highest configuration. At a 5% false-alarm rate, the calibrated model delivers a median warning of 2.97 seconds before hazard onset. These results are maintained on an independent freeway dataset (I-24 MOTION) without fine-tuning.

## Key Results

- **Ego-only (B1):** AUPRC 0.678
- **Best single cooperative source (A2 - weather / traffic):** AUPRC 0.751
- **Full cooperative fusion (F):** AUPRC 0.700
- **With behavioral plugin (A7):** AUPRC 0.808, Brier 0.166
- **Lead time at 5% false-alarm rate:** 2.97 s median warning
- **Generalization:** I-24 MOTION (Tennessee) without retraining

## Paper

Submitted to *IEEE Transactions on Vehicular Technology*.

Paper 1 repository for cooperative freeway hazard prediction.

## Folder structure

```
phase4-crest/
├── configs/                # Hydra / YAML experiment configs
├── data/
│   ├── raw/                # Downloaded datasets (NGSIM, MiTra, highD, ...)
│   ├── processed/          # Canonical-schema parquet files per dataset
│   ├── interim/            # Label windows, feature caches, splits
│   └── external/           # External holdout data (highD when available)
├── docs/                   # Paper-related notes and audit reports
├── notebooks/              # Exploratory / analysis notebooks
├── outputs/
│   ├── models/             # Trained model checkpoints
│   ├── figures/            # Generated figures
│   └── results/            # CSVs, metrics, calibration artifacts
├── scripts/                # One-off runnable scripts (audits, dataset prep)
├── src/crest/              # Main Python package
│   ├── adapters/           # Dataset-specific canonical-schema loaders
│   ├── labels/             # Hazard label extraction and windowing
│   ├── features/           # Feature builders, V2X simulation
│   ├── models/             # PyTorch models (no model code before checkpoints)
│   ├── evaluation/         # Metrics, calibration, statistical tests
│   └── utils/              # Logging, paths, constants
└── tests/                  # Unit and integration tests
```

## Pre-model-code checkpoints

Before the first model file is written, the following must be completed and confirmed:

1. NGSIM P0 audit: independent hard-braking and queue-onset event counts.
2. MiTra session usage confirmed and teleporting-jump vehicle IDs excluded.
3. Canonical schema implemented and verified on sample rows from NGSIM and MiTra.
4. Prediction-horizon T sensitivity (5s, 10s, 15s, 20s) and freeze T = 10s.
5. Hard-braking threshold sensitivity (-3.0, -3.5, -3.9, -4.5 m/s²) and freeze -3.9.
6. Coordinate conversions verified (NGSIM feet → m; MiTra km/h → m/s).
7. RSU coverage-radius and detection-uncertainty parameters configured.
8. Calibration split reserved: last 20% of NGSIM US-101 + last 20% of MiTra T6 by time.
9. SafeDriver-IQ prior artifact loadability verified (optional A5).
10. Framework: PyTorch; V2V encoder as sum-pooled MLP over top-K neighbors.

All label thresholds and evaluation protocols are frozen in `configs/base.yaml`.

## Quick commands

```bash
# Run NGSIM P0 audit
python scripts/run_ngsim_p0_audit.py

# Verify canonical adapters
python scripts/verify_canonical_schema.py
```
