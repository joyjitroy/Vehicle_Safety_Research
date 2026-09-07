# Phase 4: CREST - Calibrated Risk Estimation with Source-Aware Temporal Fusion

Phase 4 of the Vehicle Safety Research program. CREST extends the inverse crash-probability foundation of SafeDriver-IQ (Phase 1) and the multi-model risk engine of PRISM (Phase 2) into cooperative freeway hazard prediction. It combines Platt-calibrated hazard probabilities with systematic, per-source ablation of simulated V2V and RSU sensing channels, and supports inference-time behavioral and perception plugins without retraining. Submitted to **IEEE Transactions on Vehicular Technology**.

## Authors

Joyjit Roy, Sushanta Das, Samaresh Kumar Singh

## Abstract

Road traffic crashes cause approximately 1.19 million deaths globally each year, including over 39,000 in the United States in 2024. Freeway rear-end collisions, often caused by sudden queue formation and hard braking, are the focus of this paper. Most hazard-prediction models rely solely on the vehicle's own onboard state (ego-only sensing) or produce uncalibrated risk scores that do not support actionable alert thresholds. Prior research using cooperative sensing, such as vehicle-to-vehicle (V2V) or roadside unit (RSU) communication, has not isolated the individual contribution of each source to prediction accuracy. This paper introduces CREST (Calibrated Risk Estimation with Source-Aware Temporal Fusion), the first framework to combine Platt-calibrated hazard probabilities with systematic per-source ablation across simulated V2V and RSU sensing channels, modeled on current communication standards. CREST also supports adding behavioral and perception plugins at inference time without retraining. Evaluated on real freeway trajectory data, the most effective individual cooperative source increases detection accuracy (AUPRC) by over seven points compared to ego-only sensing (0.751 vs. 0.678). However, combining all sources reduces performance to 0.700, indicating that indiscriminate fusion diminishes predictive value. Introducing a pre-computed behavioral risk score at inference time reverses this trend, raising AUPRC to 0.808, the highest among all configurations. At a 5% false-alarm rate, the calibrated model delivers a median warning of 2.97 seconds before hazard onset, providing sufficient time for automated braking at highway speeds. These results are maintained on an independent freeway dataset without fine-tuning, and demonstrate that cooperative sensing sources offer unequal and non-additive value, and that source-aware calibration, rather than indiscriminate fusion, is essential for effective real-world hazard prediction.

## 1. Introduction

Freeway rear-end collisions are governed by unforgiving physics: a vehicle traveling at 25 m/s requires over 50 meters to stop under emergency braking, and reaction time adds another 25 to 30 meters, so a sensor detecting a stopped queue at 60 to 80 meters has already placed the vehicle inside the crash zone. Analytical time-to-collision and stopping-distance methods operate only at the edge of that zone and cannot provide meaningful early warning; effective intervention requires estimating hazard probability several seconds before it enters any single vehicle's sensor range, which by definition requires information from beyond that vehicle.

Two prior frameworks in this research program establish the foundation for CREST. SafeDriver-IQ converts binary crash classifiers trained on 213,003 NHTSA CRSS records into continuous 0-100 behavioral risk scores through inverse crash-probability modeling, but operates purely on single-vehicle driving history with no scene-level or cooperative-sensing signal. PRISM extends risk estimation to the scene level by fusing environmental, trajectory-kinematic, and VRU-interaction models through a deep Q-network agentic policy, but neither framework accepts V2X input or produces a calibrated hazard probability suitable for a fixed alert threshold.

Existing freeway hazard prediction and cooperative-perception work leaves three gaps unaddressed simultaneously: hazard prediction methods rely almost exclusively on ego-vehicle state; cooperative-perception systems such as V2VNet, OPV2V, DiscoNet, Where2comm, CoBEVT, and V2X-ViT fuse multi-agent observations for object detection or BEV segmentation but never isolate each sensing source's individual predictive contribution; and existing risk-scoring models typically output uncalibrated scores rather than probabilities that reflect true empirical hazard rate. CREST closes all three gaps on real freeway trajectory data through (1) the first systematic per-source cooperative-sensing ablation on real freeway data, spanning ego-only through V2V, three RSU coverage radii, and full fusion; (2) a Platt-calibrated hazard probability delivering a 2.97 s median lead time at 5% false-alarm rate; (3) an empirical RSU coverage-radius sweep (150/300/500 m yielding AUPRC 0.672/0.687/0.722), the first quantified estimate of prediction gain per unit of infrastructure coverage; (4) an inference-time plugin architecture, where a pre-computed behavioral prior raises AUPRC from 0.678 to 0.808; and (5) cross-continental generalization across US and Italian freeway trajectories with no fine-tuning.

## 2. System Architecture

![CREST Architecture](docs/images/F1_CREST_Architecture.png)

CREST is organized as a four-layer pipeline that converts heterogeneous multi-source freeway data into a calibrated, actionable hazard probability.

**Layer 1: Heterogeneous Input Sources.** Five source blocks are available at each observation window: ego kinematic state (speed, longitudinal/lateral acceleration, relative gap to the lead vehicle), V2V cooperative messages, RSU cooperative perception, static map/road geometry (elevation, lane count, speed limit, ramp proximity), and dynamic weather/traffic context (precipitation, visibility, weather code, traffic density). An input configuration `C` is any subset of these five blocks; a source not part of `C` is excluded entirely rather than zero-padded, keeping the per-source ablation clean.

**Layer 2: Source-Specific Encoders.** Four parallel MLP encoders process each active block independently. The Ego MLP is a per-timestep snapshot encoder over six scalar features, with no temporal recurrence across windows. The Neighbor MLP encodes cooperative neighbor vehicles, whether reported via V2V BSM or RSU CPM, through an identical per-vehicle path; outputs for up to K=10 nearest neighbors within 300 m are combined by sum-pooling into a fixed-length embedding regardless of neighbor count, so the observation source (V2V vs. RSU) is a configuration choice, not an architectural one. The Map MLP and Weather MLP share a unified implementation but are kept as separate ablation arms. The full network totals 20,289 parameters across all encoders and the fusion head.

**Layer 3: Fusion, Prediction, and Calibration.** Active encoder outputs are concatenated and passed through a two-layer MLP fusion head to a scalar logit `l = f_fuse([e; v; m])`, converted to a calibrated hazard probability by a frozen Platt sigmoid `p = sigmoid(a * l + b)`. Parameters `(a, b)` are estimated by maximum likelihood on a dedicated 1,284-event calibration split and frozen before any holdout evaluation, so calibration is a training-pipeline design constraint rather than a post-hoc correction.

**Layer 4: Applications and Plugins.** The calibrated probability, lead time, and false-alarm operating point are exposed to downstream driver alerts, ADAS decision support, fleet monitoring, and infrastructure planning. Two optional plugins extend the frozen base model at inference without retraining: the **behavioral prior plugin** injects a pre-computed SafeDriver-IQ risk score into the fusion vector with no gradient flow back through the plugin input, and the **onboard perception plugin** routes camera/LiDAR/radar detections through the same Neighbor MLP path used for V2V and RSU observations.

### V2X Feature Simulation

No public freeway dataset includes live V2X broadcasts, so V2V and RSU channels are simulated from trajectory replay while ego state, map geometry, and weather context come directly from trajectory records and external static datasets. V2V Basic Safety Messages follow SAE J2735: each vehicle broadcasts position, speed, heading, and acceleration at 10 Hz within a 500 m radius (the brake-event flag is deliberately excluded to avoid leaking the hazard label, which is itself derived from the same deceleration threshold). RSU Collective Perception Messages follow ETSI EN 302 637-2 and report detected objects within an experimentally varied coverage radius of 150 m, 300 m, or 500 m.

![V2X Simulation Schematic](docs/images/F3_V2X_Simulation_Schematic.png)

![V2X Coverage](docs/images/F15_V2X_Coverage.png)

Cooperative sensing geometry on a freeway corridor: V2V BSMs broadcast within a 500 m radius (cyan) and RSU CPMs are simulated with a 300 m coverage radius (green), both aggregated by the ego vehicle for CREST inference.

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

## 5. Results and Discussion

CREST was evaluated across eleven input configurations on the primary holdout (838 events spanning MiTra T9 and NGSIM I-80), summarized below.

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

†Inference-time plugins; base model not retrained.

![Ablation Bar Chart](docs/images/F5_Ablation_Bar_Chart.png)

**Source Attribution and Cooperative Value.** The ablation reveals that cooperative sources contribute unequal, non-additive value. Weather and traffic context (A2) is the strongest single source at 0.751 AUPRC, a 7.3 pp gain over ego-only, showing that dynamic environmental state carries substantial signal beyond ego kinematics; static map geometry (A1), by contrast, slightly degrades performance, indicating that fixed corridor features add noise rather than signal on these datasets. RSU coverage improves monotonically with radius: A4-150 (0.672) falls marginally below the ego-only baseline because 150 m captures too few upstream vehicles for meaningful advance context, A4-300 improves to 0.687, and A4-500 reaches 0.722, a 4.4 pp gain, while V2V BSMs (A3) provide only a marginal +0.4 pp lift under simulated channel conditions, since the kinematic overlap between V2V and ego state limits its added value.

![PR Curve](docs/images/F6_PR_Curve.png)

The precision-recall comparison confirms that learned temporal modeling matters on its own: B1 (ego-only CREST) exceeds B0 (the non-learned TTC/stopping-distance baseline) by 17.8 percentage points in AUPRC, showing a trained model substantially outperforms an analytical heuristic even without any cooperative input.

![RSU Sweep](docs/images/F7_RSU_Sweep_Plot.png)

**Negative Transfer in Fusion.** Full Fusion (F, 0.700) and Selective Fusion (Fopt, 0.688, combining only the two strongest individual sources: weather and RSU-500) both underperform the best single-source results, A2 (0.751) and A4-500 (0.722). Because Fopt still underperforms either of its two constituent sources in isolation, the negative transfer is not caused by including weak sources in the fusion set; instead, the MLP fusion head fails to suppress cross-source interference within the current training budget. Cooperative sensing architectures should not assume that adding more sources monotonically improves prediction; source selection and weighting, not indiscriminate combination, are the more effective design strategy for systems with heterogeneous V2X availability.

![Learning Curves](docs/images/F8_Learning_Curves.png)

Both B1 and A2 converge within 3 epochs with no sign of overfitting on the calibration split, and the validation-loss gap between the two stabilizes by epoch 2, showing that the weather encoder's contribution is established early and does not require extended training.

![Split Comparison](docs/images/F9_Split_Comparison.png)

**Cross-Site Generalization.** CREST (B1), evaluated on I-24 MOTION without fine-tuning, achieves an AUPRC of 0.666, only 1.2 pp below the primary holdout (0.678), and holdout AUPRC tracks calibration-split performance consistently across all configurations, confirming stable generalization rather than overfitting to a single corridor. Retraining Fopt on I-24 MOTION raises its AUPRC to 0.688, still below the best single-source results on the primary holdout, confirming the negative-transfer pattern is a property of the fusion mechanism itself, not an artifact of one training distribution.

![Lead Time](docs/images/F10_Lead_Time.png)

**Lead-Time Analysis.** At a 5% false-alarm rate, the primary reporting point, CREST detects 96 of 838 holdout events (11.5%), with a median lead time of 2.97 s and mean of 3.52 s; roughly half of detections provide at least 3 s of warning and 35.4% provide at least 5 s, though about 15% fall below 0.5 s (flagged only marginally before onset). At a stricter 1% FAR, only 21 events (2.5%) are detected with a shorter median lead time of 1.27 s, since the highest-confidence alerts are not necessarily the earliest; loosening to 10% FAR raises detection to 198 events (23.6%) but pulls the median lead time down to 2.37 s as more marginal, shorter-warning detections enter the pool. The 5% operating point was fixed before evaluation and never adjusted afterward.

**Calibration as a Design Constraint.** Platt scaling parameters are estimated on the dedicated calibration split and frozen before holdout evaluation, so calibration is built into the training pipeline rather than applied as a post-hoc adjustment. In a safety-critical alert system, overconfident probability estimates can either suppress necessary alerts or trigger unnecessary interventions; by producing probabilities that reflect empirical hazard frequency rather than raw model confidence, CREST makes the lead-time and false-alarm operating points above operationally meaningful rather than arbitrary thresholds on an uncalibrated score.

**Plugin Architecture and Post-Deployment Extensibility.** The behavioral prior plugin (A7), which injects a pre-computed SafeDriver-IQ score, reaches 0.808 AUPRC, the best of all eleven configurations and a 13.0 pp gain over ego-only, reflecting a strong separation between SafeDriver-IQ score distributions for hazard and non-hazard events. The onboard perception plugin (A6) contributes a more modest +1.6 pp by adding camera/LiDAR detection context. Because both plugins operate purely at inference time, fleet operators or ADAS integrators can add new signal sources after deployment without retraining or recertifying the base model, which is especially valuable under regulatory regimes that require re-validation for any model update.

## 6. Real-Time Inference and Deployment

At each time step t, CREST validates timestamp alignment, source freshness, and missing-data conditions across the five potential input sources before constructing the current driving scene using only information available up to t (causal ordering, consistent with training). Each active source is routed through its independent encoder, so the frozen model can run on any subset of sources without architectural modification, and inference continues on the remaining sources if a cooperative channel drops out (not evaluated during a live channel-loss event). The formal per-timestep procedure is given in Algorithm 1.

![CREST Inference Flow](docs/images/F12_CREST_Inference_Flow.png)

```
Algorithm 1 - CREST Real-Time Hazard Inference
Require: Ego state x_ego in R^6; V2X message set M; map/weather context c in R^8;
         frozen Platt parameters (a, b); FAR threshold tau
Ensure:  Calibrated hazard probability p; alert decision d

1: N   <- { m in M : ||m.pos - x_ego.pos||_2 <= 300 }        # neighbors within 300 m
2: N_K <- TopK(N, k=10, key=distance)                        # top-10 nearest neighbors
3: e   <- f_ego(x_ego)                                       # ego encoder, e in R^64
4: v   <- mean_{j in N_K} f_nbr(x_j)                          # neighbor encoder, mean-pooled
5: m   <- f_map(c)                                            # map/weather encoder
6: l   <- f_fuse([e; v; m])                                   # fusion MLP, raw logit
7: p   <- sigmoid(a * l + b)                                  # frozen Platt calibration
8: if p > tau then d <- 1 else d <- 0                         # alert decision
9: return p, d
```

The alert issuance operating point is fixed at 5% FAR, established before evaluation from the precision-recall and ROC curves and never adjusted afterward. On CPU hardware, a single forward pass through this pipeline runs in 61.7 microseconds, comfortably inside the 100 ms V2X broadcast cycle and well within real-time budgets for driver alerts or ADAS decision support.

![CREST HMI Alert](docs/images/F13_CREST_HMI_Alert.png)

An illustrative HMI display showing a hazard probability of 0.83 and a lead-time estimate of 2.9 s alongside a queue-ahead advisory; the display is indicative only, and actual interface standards and integration are outside the scope of the paper.

![CREST Warning Sequence](docs/images/F14_CREST_Warning_Sequence.png)

The warning-to-stop sequence at the median lead time: at t = -2.97 s CREST issues an alert (hazard probability 0.83, queue-ahead advisory), braking is initiated at t = -1.5 s, and the vehicle reaches a safe stop at t = 0 s. The 1.47 s interval between the alert and the start of braking is the driver-or-ADAS response window available at the median operating point.

Full cooperative deployment depends on the availability of roadside V2X infrastructure and vehicle-side onboard units, neither of which is evaluated in this paper; latency budgets, communication reliability under congestion, regulatory certification, and fleet penetration rates remain open engineering and policy questions for operational rollout.

## 7. Application Examples

The calibrated hazard probability and its accompanying lead time are designed to plug directly into graduated, threshold-based alert systems. For **ADAS decision support**, the probability can be thresholded at the same 5% FAR operating point used in evaluation to trigger a driver alert with a known false-alarm rate and a median 2.97 s warning window, sufficient for automated emergency braking at highway speeds. For **fleet safety monitoring**, the same calibrated probability stream, aggregated across trips, can flag corridors or time windows with elevated hazard rates without requiring an actual crash to have occurred. For **infrastructure planning**, the empirical RSU coverage-radius sweep (0.672 at 150 m, 0.687 at 300 m, 0.722 at 500 m) gives planners a quantified estimate of predictive gain per unit of roadside sensor coverage, supporting evidence-based deployment-density decisions.

## 8. Repository Structure

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

## 9. Pre-Model-Code Checkpoints

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

## 10. Quick Commands

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

## 11. Limitations

**Simulated V2X Communications.** V2V Basic Safety Messages and RSU Collective Perception Messages are simulated from trajectory replay rather than captured from live broadcasts. Channel impairments such as packet loss, latency, and interference are modeled parametrically, but this does not fully represent the variability of real-world V2X deployments, so performance under live channel conditions may differ from the results reported here.

**Snapshot Encoding.** CREST's encoders process each observation window independently, with no temporal recurrence across consecutive windows. Sequential dependencies across windows are not modeled; recurrent or attention-based temporal encoding may capture additional predictive structure in freeway traffic dynamics that the current snapshot architecture cannot.

**Detection Rate at the Primary Operating Point.** At 5% FAR, CREST detects only 96 of 838 holdout events (11.5%); the majority of hazard events in the holdout set are not detected at this threshold. This highlights both the inherent difficulty of early freeway hazard prediction from trajectory data and the operational cost of a conservative false-alarm constraint, and it limits the system's coverage at the reported operating point.

**Geographic and Environmental Scope.** Evaluation is conducted entirely on US and European freeway datasets (NGSIM, MiTra, I-24 MOTION). Performance on urban arterials, signalized intersections, or road networks with different traffic characteristics has not been assessed, and generalization beyond structured freeways requires separate evaluation.

**Simulated Behavioral Prior.** The behavioral prior plugin (A7) uses a SafeDriver-IQ risk score computed from the same trajectory data used for training and evaluation. In live deployment this score would instead be computed from a separate longitudinal driving history, so the plugin's 0.808 AUPRC result should be treated as an upper bound on the contribution of driver behavioral context under ideal data conditions, not a guaranteed field result.

## 12. Conclusion and Future Directions

CREST establishes source-aware ablation as a design principle for cooperative freeway hazard prediction: it isolates each sensing channel's independent contribution and pairs those attributions with Platt-calibrated probabilities on real trajectory data, and thereby supports evidence-based infrastructure investment decisions. Across eleven configurations, cooperative sources prove non-additive; the best single source (A2, 0.751 AUPRC) outperforms full fusion (F, 0.700), while injecting a behavioral prior at inference (A7) yields the strongest result overall (0.808 AUPRC, 0.166 Brier). The model generalizes to an independent Tennessee freeway dataset with zero fine-tuning, and the calibrated model delivers a 2.97 s median warning lead time at 5% FAR while running in 61.7 microseconds on CPU hardware, well inside the 100 ms V2X broadcast cycle, sufficient distance at highway speeds for automated emergency braking to prevent or substantially mitigate rear-end collisions.

These properties position CREST as a deployable upstream sensing module for threshold-based intervention systems: its calibrated output can integrate directly with graduated alert architectures such as PRISM without per-site recalibration. As connected-vehicle infrastructure adoption increases, CREST's RSU coverage-sensitivity results give infrastructure planners empirical guidance on deployment density. Future work will validate the framework on live V2X field data and assess real-time behavioral score streams on connected-vehicle testbeds, alongside extending the fusion head to resolve the negative-transfer pattern above and jointly training the inference-time plugins with the base hazard model.

## 13. Publication Status

Submitted to *IEEE Transactions on Vehicular Technology*. Not yet available on arXiv or IEEE Xplore; citation details will be added once published.
