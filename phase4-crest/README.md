# Phase 4: CREST - Calibrated Risk Estimation with Source-Aware Temporal Fusion

Phase 4 of the Vehicle Safety Research program. CREST extends the inverse crash-probability foundation of SafeDriver-IQ (Phase 1) and the multi-model risk engine of PRISM (Phase 2) into cooperative freeway hazard prediction. It combines Platt-calibrated hazard probabilities with systematic, per-source ablation of simulated V2V and RSU sensing channels, and supports inference-time behavioral and perception plugins without retraining. Submitted to **IEEE Transactions on Vehicular Technology**.

## Authors

Joyjit Roy, Sushanta Das, Samaresh Kumar Singh

## Abstract

Road traffic crashes cause approximately 1.19 million deaths globally each year, including over 39,000 in the United States in 2024. Freeway rear-end collisions, often caused by sudden queue formation and hard braking, are the focus of this paper. Most hazard-prediction models rely solely on the vehicle's own onboard state (ego-only sensing) or produce uncalibrated risk scores that do not support actionable alert thresholds. Prior research using cooperative sensing, such as vehicle-to-vehicle (V2V) or roadside unit (RSU) communication, has not isolated the individual contribution of each source to prediction accuracy. This paper introduces CREST (Calibrated Risk Estimation with Source-Aware Temporal Fusion), the first framework to combine Platt-calibrated hazard probabilities with systematic per-source ablation across simulated V2V and RSU sensing channels, modeled on current communication standards. CREST also supports adding behavioral and perception plugins at inference time without retraining. Evaluated on real freeway trajectory data, the most effective individual cooperative source increases detection accuracy (AUPRC) by over seven points compared to ego-only sensing (0.751 vs. 0.678). However, combining all sources reduces performance to 0.700, indicating that indiscriminate fusion diminishes predictive value. Introducing a pre-computed behavioral risk score at inference time reverses this trend, raising AUPRC to 0.808, the highest among all configurations. At a 5% false-alarm rate, the calibrated model delivers a median warning of 2.97 seconds before hazard onset, providing sufficient time for automated braking at highway speeds. These results are maintained on an independent freeway dataset without fine-tuning, and demonstrate that cooperative sensing sources offer unequal and non-additive value, and that source-aware calibration, rather than indiscriminate fusion, is essential for effective real-world hazard prediction.

## 1. Introduction

Freeway rear-end collisions are caused by sudden queue formation triggered by hard braking that propagates upstream faster than drivers can react. A vehicle traveling at 25 m/s requires over 50 meters to stop under emergency braking. Production ADAS relies on ego-only sensing, which cannot detect a queue forming beyond onboard sensor range. Prior cooperative-perception work (V2VNet, OPV2V, DiscoNet, Where2comm, CoBEVT, V2X-ViT, HEAL) focuses on object detection or BEV segmentation, not calibrated hazard probability, and none isolate the per-source predictive contribution needed for infrastructure investment decisions. CREST closes this gap with (1) per-source ablation across V2V, RSU, weather/traffic, and map-geometry channels, (2) Platt calibration for actionable alert thresholds, and (3) an inference-time plugin architecture for behavioral and perception priors.

## 2. System Architecture

![CREST Architecture](docs/images/F1_CREST_Architecture.png)

CREST is a multi-branch encoder-fusion network:

| Component | Role |
|---|---|
| **Ego encoder** | Encodes host-vehicle kinematic state (position, velocity, acceleration) over the observation window |
| **Neighbor encoder** | Top-K neighbor selection with a sum-pooled MLP; ingests simulated V2V BSMs (A3) or RSU CPMs (A4) when active |
| **Map encoder** | Encodes road geometry/elevation (A1) and weather/local-traffic context (A2) when active |
| **Fusion MLP** | Concatenates active encoder outputs into a single hazard logit |
| **Platt calibration layer** | Frozen post-hoc sigmoid mapping raw logit to a calibrated hazard probability, fit on the calibration split |
| **Inference-time plugin slot** | Accepts a pre-computed perception score (A6) or behavioral risk score (A7, SafeDriver-IQ) without retraining the base network |

The frozen model can operate on any subset of the five input sources (ego, V2V, RSU, map, weather/traffic) without architectural modification, so inference continues with the remaining sources if a cooperative channel becomes unavailable.

### V2X Simulation

![V2X Simulation Schematic](docs/images/F3_V2X_Simulation_Schematic.png)

V2V Basic Safety Messages (BSMs) and RSU Cooperative Perception Messages (CPMs) are simulated on top of ground-truth trajectories, including communication range limits, detection uncertainty (missed detections, position noise, stale tracks), and an RSU coverage-radius sweep (150 m, 300 m, 500 m).

![V2X Coverage](docs/images/F15_V2X_Coverage.png)

### Real-Time Inference Pipeline

![CREST Inference Flow](docs/images/F12_CREST_Inference_Flow.png)

At each time step, an input quality/consistency stage validates timestamp alignment, source freshness, and missing data before any source contributes to the scene representation. The frozen CREST model produces a raw hazard logit, and the frozen Platt layer converts it to a calibrated probability. This probability, plus the lead time at the chosen FAR operating point, is the actionable output delivered to driver alerts, ADAS decision support, fleet monitoring, and infrastructure planning.

![CREST HMI Alert](docs/images/F13_CREST_HMI_Alert.png)
![CREST Warning Sequence](docs/images/F14_CREST_Warning_Sequence.png)

## 3. Datasets

CREST is trained and evaluated on NGSIM and MiTra, two open-access freeway trajectory datasets covering different countries, sensor modalities, and traffic conditions. I-24 MOTION (Nashville, USA) is an independent additional US validation site, evaluated without fine-tuning.

### Dataset Characteristics

| Property | NGSIM | MiTra | I-24 MOTION |
|---|---|---|---|
| Method | Ground camera | Drone | Ground camera |
| Location | California, USA | Milan, Italy | Nashville, USA |
| Freeway | US-101, I-80 | A50 | I-24 |
| Duration | ~45 min/site | 135 min (9 sessions) | 3 incident days (Nov-Dec 2022) |
| Traffic | Mixed | Free-flow to congestion | Incident-driven |
| Labels | Queue onset | Hard braking, queue onset | Queue onset |
| Role | Train + holdout | Train + holdout | US validation |
| Source / Publication | FHWA, ITS DataHub, released 2016 [19] | Chaudhari et al., *Scientific Data*, 2025 [20] | Gloudemans et al., *Transp. Res. Part C*, 2023 [21] |

### Data Split Summary

![Data Split Diagram](docs/images/F2_Data_Split_Diagram.png)

| Split | Source | Events |
|---|---|---|
| Training | MiTra T4-T7 + NGSIM US-101 (first 70%) | 10,640 (5,320 positive + 5,320 negative, balanced) |
| Calibration | MiTra T8 + NGSIM US-101 (last 30%) | 1,284 |
| Holdout | MiTra T9 + NGSIM I-80 | 838 |

### Hazard Label Definitions

| Label Type | Definition | Source |
|---|---|---|
| Hard braking | Deceleration <= -3.9 m/s^2 for at least 0.5 s | MiTra only (NGSIM camera-stitching suppresses extreme deceleration) |
| Queue onset | 3+ adjacent vehicles below 2 m/s for at least 5 s | NGSIM and MiTra |

An observation window is labeled y=1 if either hazard type onset falls within the lookahead horizon T=10 s.

## 4. CREST Input Configurations (Ablation Design)

| ID | Configuration | Active Sources | Family |
|---|---|---|---|
| B0 | TTC Baseline | Analytical only | Baseline |
| B1 | Ego-Only | Ego kinematics | Baseline |
| A1 | Ego + Map Context | Ego + map geometry | Single-source |
| A2 | Ego + Weather | Ego + weather/traffic context | Single-source |
| A3 | Ego + V2V | Ego + V2V BSMs | Single-source |
| A4-150 | Ego + RSU-150 | Ego + RSU (150 m radius) | Infrastructure sweep |
| A4-300 | Ego + RSU-300 | Ego + RSU (300 m radius) | Infrastructure sweep |
| A4-500 | Ego + RSU-500 | Ego + RSU (500 m radius) | Infrastructure sweep |
| F | Full Fusion | All sources | Fusion |
| Fopt | Selective Fusion | Ego + Weather + RSU-500 | Fusion |
| A6 | Ego + Perception Plugin | Ego + onboard perception (inference-time) | Plugin |
| A7 | Ego + Behavioral Prior | Ego + SafeDriver-IQ score (inference-time) | Plugin |

## 5. Results

### Holdout Results by Input Configuration

![Ablation Bar Chart](docs/images/F5_Ablation_Bar_Chart.png)

| ID | Configuration | AUPRC | Brier | delta pp vs. B1 |
|---|---|---|---|---|
| B0 | TTC Baseline | 0.500 | 0.500 | - |
| B1 | Ego-Only | 0.678 | 0.230 | Ref. |
| A1 | Ego + Map Context | 0.664 | 0.231 | -1.4 |
| A2 | Ego + Weather | **0.751** | 0.218 | **+7.3** |
| A3 | Ego + V2V | 0.674 | 0.230 | -0.4 |
| A4-150 | Ego + RSU-150 | 0.672 | 0.230 | -0.6 |
| A4-300 | Ego + RSU-300 | 0.687 | 0.231 | +0.9 |
| A4-500 | Ego + RSU-500 | 0.722 | 0.229 | +4.4 |
| F | Full Fusion | 0.700 | 0.233 | +2.2 |
| Fopt | Selective Fusion | 0.688 | 0.229 | +1.0 |
| A6 | Ego + Perception Plugin† | 0.694 | 0.226 | +1.6 |
| A7 | Ego + Behavioral Prior† | **0.808** | **0.166** | **+13.0** |

†Inference-time plugins; base model not retrained. Delta pp = signed difference in percentage points from Ego-Only AUPRC (0.678).

**Key findings:**
- Weather/traffic context (A2) is the strongest single cooperative source; map geometry (A1) slightly degrades performance (static features add noise, not signal).
- B1 exceeds B0 by 17.8 pp AUPRC, confirming temporal learning over ego kinematics substantially beats a non-learned TTC heuristic.
- RSU coverage radius has a monotonic effect: 150 m underperforms baseline, 300 m and 500 m improve, with 500 m reaching +4.4 pp.
- **Negative transfer:** Full Fusion (0.700) and Selective Fusion (0.688) both underperform the best single sources (A2 0.751, A4-500 0.722). The MLP fusion head fails to suppress cross-source interference within the current training budget.
- Inference-time plugins avoid this: the behavioral prior (A7) yields the best overall result (+13.0 pp) without retraining the base model.

### Precision-Recall Curves

![PR Curve](docs/images/F6_PR_Curve.png)

### RSU Coverage Sweep

![RSU Sweep](docs/images/F7_RSU_Sweep_Plot.png)

### Training Dynamics

![Learning Curves](docs/images/F8_Learning_Curves.png)

Both B1 and A2 converge within 3 epochs with no overfitting on the calibration split.

### Cross-Site Generalization (Split Comparison)

![Split Comparison](docs/images/F9_Split_Comparison.png)

CREST (B1) evaluated on I-24 MOTION without fine-tuning achieves AUPRC 0.666 (-1.2 pp vs. primary holdout 0.678). Holdout AUPRC tracks calibration-split performance consistently across all configurations. Retraining Fopt on I-24 MOTION increases AUPRC to 0.688, still below the best single-source results on the primary holdout, confirming the negative-transfer pattern is not site-specific.

### Lead Time Analysis

![Lead Time](docs/images/F10_Lead_Time.png)

Lead time is the interval between a CREST alert and hazard onset, evaluated at pre-specified false-alarm rate (FAR) operating points.

| FAR | Detected | Rate | Median Lead Time | Mean Lead Time |
|---|---|---|---|---|
| 1% | 21/838 | 2.5% | 1.27 s | 2.48 s |
| **5%** | **96/838** | **11.5%** | **2.97 s** | **3.52 s** |
| 10% | 198/838 | 23.6% | 2.37 s | 3.34 s |

At 5% FAR (primary operating point), 50% of detections provide at least 3 s of warning and 35.4% provide at least 5 s. Approximately 15% of detections fall below 0.5 s (flagged only marginally before onset).

## 6. Repository Structure

```
phase4-crest/
├── configs/                        # Frozen experiment configuration (YAML)
│   ├── base.yaml                   # Label thresholds, horizon, splits, V2X sim params
│   ├── model.yaml                  # CREST architecture (encoder dims, fusion, dropout)
│   └── splits.yaml                 # Train/calibration/holdout assignment
├── docs/
│   └── images/                     # Paper figures (F1-F15)
├── src/crest/                      # Main Python package
│   ├── adapters/
│   │   ├── schema.py                # Canonical trajectory schema + validation
│   │   ├── ngsim.py                 # NGSIM CSV -> canonical schema adapter
│   │   └── mitra.py                 # MiTra CSV -> canonical schema adapter
│   ├── labels/
│   │   └── event_extraction.py      # Hard-braking + queue-onset label extraction
│   ├── features/
│   │   ├── snapshot.py              # Per-timestep feature snapshot extraction
│   │   ├── map_features.py          # Synthetic map/elevation feature generator (A1)
│   │   └── v2x_sim.py                # V2V/RSU simulation + detection uncertainty (A3/A4)
│   ├── datasets/
│   │   └── crest_dataset.py         # PyTorch Dataset over canonical trajectory data
│   ├── models/
│   │   ├── architecture.py          # CRESTConfig + CRESTModel (ego/neighbor/map encoders, fusion)
│   │   └── neighbor_utils.py        # Top-K neighbor selection utilities
│   └── evaluation/                  # Metrics, calibration, statistical tests
└── scripts/                         # Runnable pipeline scripts (see table below)
```

### Scripts Reference

| Script | Purpose |
|---|---|
| `run_ngsim_p0_audit.py` | P0 audit: independent hard-braking / queue-onset event counts on NGSIM |
| `run_ngsim_hard_brake_quick.py` | Quick NGSIM hard-braking sanity check |
| `verify_canonical_schema.py` | Verifies canonical schema on sample NGSIM/MiTra rows |
| `verify_coordinate_conversions.py` | Verifies unit conversions (NGSIM ft->m; MiTra km/h->m/s) |
| `verify_safedriver_iq_prior.py` | Verifies SafeDriver-IQ prior artifact loadability (A7) |
| `label_threshold_sensitivity.py` | Hard-braking threshold sensitivity sweep (-3.0/-3.5/-3.9/-4.5 m/s^2) |
| `label_horizon_sensitivity.py` | Prediction-horizon sensitivity sweep (5/10/15/20 s) |
| `build_labels.py` | Extracts hazard event windows and labels from canonical trajectories |
| `assign_splits.py` | Assigns train/calibration/holdout splits per `configs/splits.yaml` |
| `designate_calibration_split.py` | Reserves the calibration split for Platt scaling |
| `precompute_features.py` | Precomputes B1 (ego-only) feature tensors to NPZ |
| `precompute_features_a1.py` | Precomputes A1 features (ego + synthetic map/elevation) |
| `precompute_features_a2.py` | Precomputes A2 features (ego + weather/traffic context) |
| `precompute_features_a3.py` | Precomputes A3 features (ego + simulated V2V BSMs) |
| `precompute_features_a4.py` | Precomputes A4 features (ego + simulated RSU CPMs, radius sweep) |
| `precompute_features_a6.py` | Precomputes A6 features (ego + onboard perception plugin) |
| `precompute_features_a7.py` | Precomputes A7 features (ego + SafeDriver-IQ behavioral prior plugin) |
| `precompute_features_f.py` | Precomputes F features (all channels combined) |
| `precompute_features_f2.py` / `_f3.py` | Alternate full-fusion feature variants |
| `precompute_features_fopt.py` | Precomputes Fopt features (ego + weather + RSU-500) |
| `fetch_weather.py` | Fetches historical hourly weather (Open-Meteo) for NGSIM/MiTra sessions |
| `fetch_elevation.py` | Fetches elevation + OSM road features for map/elevation channel |
| `extract_i24_windows.py` | Extracts I-24 MOTION observation windows for independent validation |
| `compute_b0_baseline.py` | Computes the analytical TTC/stopping-distance B0 baseline |
| `train_baseline.py` | Trains the B1 ego-only CREST model |
| `train_fopt.py` | Trains the Fopt selective-fusion model |
| `run_ablations.py` | Trains/evaluates all ablation configurations (A1-A4, F, A6, A7) |
| `eval_holdout.py` | Evaluates a trained checkpoint on the holdout split |
| `infer_i24_holdout.py` | Runs inference on the I-24 MOTION holdout without fine-tuning |
| `compute_lead_time.py` | Computes lead time at FAR operating points (1%, 5%, 10%) |
| `generate_figures.py` | Generates all publication figures from saved results JSON/NPZ |

### Configuration Files

| File | Contents |
|---|---|
| `configs/base.yaml` | Prediction horizon (T=10s), label thresholds, sampling rate, V2X simulation params (RSU coverage radii, detection uncertainty) |
| `configs/model.yaml` | CREST architecture: input features, ego/neighbor/map encoder dims, fusion MLP, dropout |
| `configs/splits.yaml` | Train/calibration/holdout split assignment (session/time-based) |

## 7. Pre-Model-Code Checkpoints

Before the first model file was written, the following were completed and confirmed:

1. NGSIM P0 audit: independent hard-braking and queue-onset event counts.
2. MiTra session usage confirmed and teleporting-jump vehicle IDs excluded.
3. Canonical schema implemented and verified on sample rows from NGSIM and MiTra.
4. Prediction-horizon T sensitivity (5s, 10s, 15s, 20s) and freeze T = 10s.
5. Hard-braking threshold sensitivity (-3.0, -3.5, -3.9, -4.5 m/s^2) and freeze -3.9.
6. Coordinate conversions verified (NGSIM feet -> m; MiTra km/h -> m/s).
7. RSU coverage-radius and detection-uncertainty parameters configured.
8. Calibration split reserved: last 20% of NGSIM US-101 + last 20% of MiTra T6 by time.
9. SafeDriver-IQ prior artifact loadability verified (optional A5/A7).
10. Framework: PyTorch; V2V encoder as sum-pooled MLP over top-K neighbors.

All label thresholds and evaluation protocols are frozen in `configs/base.yaml`.

## 8. Quick Commands

```bash
# 1. Audit and verification
python scripts/run_ngsim_p0_audit.py
python scripts/verify_canonical_schema.py
python scripts/verify_coordinate_conversions.py

# 2. Labels and splits
python scripts/build_labels.py
python scripts/assign_splits.py

# 3. Precompute features per configuration
python scripts/precompute_features.py        # B1
python scripts/precompute_features_a1.py      # A1
python scripts/precompute_features_a2.py      # A2
python scripts/precompute_features_a3.py      # A3
python scripts/precompute_features_a4.py      # A4 (150/300/500)
python scripts/precompute_features_f.py       # F
python scripts/precompute_features_fopt.py    # Fopt
python scripts/precompute_features_a6.py      # A6
python scripts/precompute_features_a7.py      # A7

# 4. Baselines and training
python scripts/compute_b0_baseline.py
python scripts/train_baseline.py              # B1
python scripts/run_ablations.py               # A1-A4, F, A6, A7
python scripts/train_fopt.py                  # Fopt

# 5. Evaluation
python scripts/eval_holdout.py
python scripts/infer_i24_holdout.py
python scripts/compute_lead_time.py

# 6. Figures
python scripts/generate_figures.py
```

## 9. Limitations

- Simulated V2V and RSU observations are derived from ground-truth trajectories; real communication delay, packet loss, and sensor noise are not fully represented.
- The behavioral (A7) and perception (A6) plugins are pre-computed and not jointly trained with the hazard model.
- Fusion negative transfer (F, Fopt underperforming best single sources) indicates the current MLP fusion head does not suppress cross-source interference within the training budget used.
- Channel-loss robustness (operating on a reduced source subset) is architecturally supported but not evaluated during a live channel-loss event.

## 10. Citation

```bibtex
@article{crest2026,
  author  = {Roy, Joyjit and Das, Sushanta and Singh, Samaresh Kumar},
  title   = {CREST: Calibrated Risk Estimation with Source-Aware Temporal Fusion for Cooperative Freeway Hazard Prediction},
  journal = {IEEE Transactions on Vehicular Technology},
  year    = {2026},
  note    = {Submitted}
}
```
