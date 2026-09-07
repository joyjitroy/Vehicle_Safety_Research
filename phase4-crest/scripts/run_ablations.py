"""Run all CREST ablation experiments (B1, A1, A2, A3, A4x3, F) per the V3.1 spec.

Config      Inputs                                              Feature files (outputs/features/)
B1          Ego time-series only                                train.npz / cal.npz / holdout.npz
A1          Ego + map/elevation features                        a1_{split}.npz
A2          Ego + traffic/weather features                      a2_{split}.npz
A3          Ego + simulated V2V BSMs (neighbor messages)         a3_{split}.npz
A4_r150     Ego + simulated RSU CPMs, 150m coverage              a4_r150_{split}.npz
A4_r300     Ego + simulated RSU CPMs, 300m coverage              a4_r300_{split}.npz
A4_r500     Ego + simulated RSU CPMs, 500m coverage              a4_r500_{split}.npz
F           All inputs combined (map+weather 16-dim, V2V+RSU)    f_{split}.npz

Each ablation reuses the frozen CREST architecture (src/crest/models/architecture.py)
and training/evaluation protocol (Adam, BCE loss, 5 epochs, Platt scaling on cal
split). Configs whose NPZ has no "map_pos"/"map_neg" field (B1, A3, A4x3) use a
zero map vector of the architecture's frozen map_input_dim=8. F uses map_input_dim=16
(see scripts/precompute_features_f.py docstring for why this deviates from the
frozen configs/model.yaml value, applied only to F's own CRESTConfig instance).

B1 is NOT retrained if outputs/checkpoints/baseline.pt already exists (it was
trained and approved in an earlier session); this script simply reloads it and
re-evaluates on holdout for a consistent entry in ablations.json. Pass
--retrain-b1 to force retraining from scratch.

Results are written to outputs/results/ablations.json, one key per config.
Existing entries for OTHER configs are preserved (this script updates one or
more keys per invocation, controlled by CLI args).

Usage:
  python run_ablations.py                 # runs all configs
  python run_ablations.py A1 A2           # runs only the listed configs
  python run_ablations.py --retrain-b1    # forces B1 retraining too
"""
import argparse
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.calibration import _sigmoid_calibration
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from crest.models.architecture import CRESTConfig, CRESTModel, count_parameters

MODEL_YAML = ROOT / 'configs' / 'model.yaml'
FEATURES_DIR = ROOT / 'outputs' / 'features'
CKPT_DIR = ROOT / 'outputs' / 'checkpoints'
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = ROOT / 'outputs' / 'results' / 'ablations.json'
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3

# name -> (npz_prefix or None for B1's unprefixed files, map_dim, checkpoint filename)
CONFIGS = {
    'B1':      (None,      8,  'baseline.pt'),
    'A1':      ('a1',      8,  'a1_map_elevation.pt'),
    'A2':      ('a2',      8,  'a2_weather_traffic.pt'),
    'A3':      ('a3',      8,  'a3_v2v_bsm.pt'),
    'A4_r150': ('a4_r150', 8,  'a4_r150_rsu_cpm.pt'),
    'A4_r300': ('a4_r300', 8,  'a4_r300_rsu_cpm.pt'),
    'A4_r500': ('a4_r500', 8,  'a4_r500_rsu_cpm.pt'),
    'F':       ('f',       16, 'f_combined.pt'),
    # A6: ego + simulated onboard perception detections (FOV cone, no map)
    'A6':      ('a6',      8,  'a6_perception.pt'),
    # F2: all sources + onboard perception (ego + map + weather + V2V + RSU + perception)
    'F2':      ('f2',      16, 'f2_full_perception.pt'),
    # A7: ego + SafeDriver IQ behavioral prior (no neighbors, no map/weather/V2X)
    'A7':      ('a7',      8,  'a7_safedriver_iq.pt'),
    # F3: all sources + SafeDriver IQ behavioral prior (map_dim=24: 8 map + 8 weather + 8 IQ)
    'F3':      ('f3',      24, 'f3_full_safedriver.pt'),
}


def load_model_config(path: Path, map_input_dim_override: int = None) -> CRESTConfig:
    with open(path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    arch = cfg['architecture']
    input_dim = len(arch['input_features'])
    map_input_dim = map_input_dim_override if map_input_dim_override is not None else arch['map_encoder']['input_dim']
    return CRESTConfig(
        ego_input_dim=input_dim,
        ego_hidden_dims=arch['ego_encoder']['hidden_dims'],
        ego_output_dim=arch['ego_encoder']['output_dim'],
        neighbor_input_dim=input_dim,
        neighbor_hidden_dims=arch['neighbor_encoder']['hidden_dims'],
        neighbor_output_dim=arch['neighbor_encoder']['output_dim'],
        map_input_dim=map_input_dim,
        map_hidden_dims=arch['map_encoder']['hidden_dims'],
        map_output_dim=arch['map_encoder']['output_dim'],
        fusion_hidden_dims=arch['fusion']['hidden_dims'],
        dropout=arch.get('dropout', 0.1),
    )


def load_split_tensors(prefix, split: str, map_dim: int):
    filename = f'{split}.npz' if prefix is None else f'{prefix}_{split}.npz'
    data = np.load(FEATURES_DIR / filename)
    ego = np.concatenate([data['ego_pos'], data['ego_neg']], axis=0)
    neighbors = np.concatenate([data['neighbors_pos'], data['neighbors_neg']], axis=0)
    mask = np.concatenate([data['mask_pos'], data['mask_neg']], axis=0)
    labels = np.concatenate([data['labels_pos'], data['labels_neg']], axis=0)

    if 'map_pos' in data.files:
        map_feat = np.concatenate([data['map_pos'], data['map_neg']], axis=0).astype(np.float32)
    else:
        map_feat = np.zeros((len(labels), map_dim), dtype=np.float32)

    return (
        torch.from_numpy(ego),
        torch.from_numpy(neighbors),
        torch.from_numpy(mask),
        torch.from_numpy(map_feat),
        torch.from_numpy(labels),
    )


def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for ego, neigh, mask, mapf, labels in loader:
            out = model(ego.to(device), neigh.to(device), mask.to(device), mapf.to(device))
            all_probs.extend(out.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
    all_probs = np.clip(np.array(all_probs), 1e-6, 1.0 - 1e-6)
    all_labels = np.array(all_labels)
    return {
        'auprc': float(average_precision_score(all_labels, all_probs)),
        'brier': float(brier_score_loss(all_labels, all_probs)),
        'log_loss': float(log_loss(all_labels, all_probs)),
        'mean_prob': float(all_probs.mean()),
    }


def fit_platt(logits, labels):
    logits = np.asarray(logits).reshape(-1, 1)
    labels = np.asarray(labels)
    return _sigmoid_calibration(logits.ravel(), labels)


def train_config(name: str, prefix, map_dim: int, ckpt_name: str):
    device = torch.device('cpu')
    cfg = load_model_config(MODEL_YAML, map_input_dim_override=map_dim)
    model = CRESTModel(cfg).to(device)
    print(f'[{name}] Model parameters: {count_parameters(model):,}')

    train_ds = TensorDataset(*load_split_tensors(prefix, 'train', map_dim))
    cal_ds = TensorDataset(*load_split_tensors(prefix, 'cal', map_dim))
    holdout_ds = TensorDataset(*load_split_tensors(prefix, 'holdout', map_dim))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    cal_loader = DataLoader(cal_ds, batch_size=BATCH_SIZE, shuffle=False)
    holdout_loader = DataLoader(holdout_ds, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()

    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        t0 = time.time()
        for ego, neigh, mask, mapf, labels in train_loader:
            optimizer.zero_grad()
            out = model(ego.to(device), neigh.to(device), mask.to(device), mapf.to(device))
            loss = criterion(out, labels.to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_time = time.time() - t0
        train_metrics = evaluate(model, train_loader, device)
        cal_metrics = evaluate(model, cal_loader, device)
        print(f'[{name}] Epoch {epoch}/{EPOCHS} — loss={epoch_loss/n_batches:.4f}, '
              f'train_auprc={train_metrics["auprc"]:.4f}, cal_auprc={cal_metrics["auprc"]:.4f} ({train_time:.1f}s)')
        history.append({'epoch': epoch, 'train': train_metrics, 'cal': cal_metrics, 'loss': epoch_loss / n_batches})

    ckpt_path = CKPT_DIR / ckpt_name
    torch.save(model.state_dict(), ckpt_path)

    model.eval()
    cal_logits, cal_labels = [], []
    with torch.no_grad():
        for ego, neigh, mask, mapf, labels in cal_loader:
            out = model(ego.to(device), neigh.to(device), mask.to(device), mapf.to(device)).cpu().numpy()
            eps = 1e-6
            probs = np.clip(out.ravel(), eps, 1 - eps)
            cal_logits.extend(np.log(probs / (1 - probs)).tolist())
            cal_labels.extend(labels.numpy().tolist())
    platt_a, platt_b = fit_platt(np.array(cal_logits), np.array(cal_labels))

    holdout_metrics = evaluate(model, holdout_loader, device)

    return {
        'config': name, 'map_input_dim': map_dim, 'checkpoint': str(ckpt_path),
        'history': history, 'platt': {'a': float(platt_a), 'b': float(platt_b)},
        'holdout': holdout_metrics,
    }


def load_existing_b1(map_dim: int, ckpt_name: str):
    """Reload the already-trained B1 checkpoint and re-evaluate (no retraining)."""
    device = torch.device('cpu')
    cfg = load_model_config(MODEL_YAML, map_input_dim_override=map_dim)
    model = CRESTModel(cfg).to(device)
    model.load_state_dict(torch.load(CKPT_DIR / ckpt_name, map_location=device, weights_only=True))

    holdout_ds = TensorDataset(*load_split_tensors(None, 'holdout', map_dim))
    holdout_loader = DataLoader(holdout_ds, batch_size=BATCH_SIZE, shuffle=False)
    holdout_metrics = evaluate(model, holdout_loader, device)

    platt_path = CKPT_DIR / 'platt.json'
    platt = json.load(open(platt_path, encoding='utf-8')) if platt_path.exists() else None
    history_path = CKPT_DIR / 'history.json'
    history = json.load(open(history_path, encoding='utf-8')) if history_path.exists() else []

    return {
        'config': 'B1', 'map_input_dim': map_dim, 'checkpoint': str(CKPT_DIR / ckpt_name),
        'history': history, 'platt': platt, 'holdout': holdout_metrics,
        'note': 'Reloaded existing checkpoint; not retrained.',
    }


def load_results():
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)


def main():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    parser = argparse.ArgumentParser()
    parser.add_argument('configs', nargs='*', default=None, help='Subset of configs to run (default: all)')
    parser.add_argument('--retrain-b1', action='store_true', help='Force B1 retraining instead of reusing checkpoint')
    args = parser.parse_args()

    to_run = args.configs if args.configs else list(CONFIGS.keys())
    for name in to_run:
        if name not in CONFIGS:
            raise ValueError(f'Unknown config: {name}. Valid: {list(CONFIGS.keys())}')

    results = load_results()

    for name in to_run:
        prefix, map_dim, ckpt_name = CONFIGS[name]
        print(f'\n=== Running {name} ===')

        if name == 'B1' and (CKPT_DIR / ckpt_name).exists() and not args.retrain_b1:
            result = load_existing_b1(map_dim, ckpt_name)
        else:
            result = train_config(name, prefix, map_dim, ckpt_name)

        results[name] = result
        save_results(results)
        print(f'[{name}] holdout metrics: {result["holdout"]}')

    print('\nDONE')


if __name__ == '__main__':
    main()
