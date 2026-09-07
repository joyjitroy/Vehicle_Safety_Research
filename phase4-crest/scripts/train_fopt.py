"""Train F_opt = A2 (weather/traffic, 8-dim map) + A4_r500 (RSU 500m, no V2V).

Follows the identical protocol as run_ablations.py:
    Adam, LR=1e-3, BCE loss, 5 epochs, Platt scaling on cal split.
    map_input_dim=8  (same as model.yaml default — no architecture override needed)

Prerequisite: run precompute_features_fopt.py first to generate:
    outputs/features/fopt_{train,cal,holdout}.npz

Output:
    outputs/checkpoints/fopt.pt
    outputs/results/ablations.json  ← entry added/updated at key "F_opt"

Usage (from project root, with .venv active):
    python scripts/train_fopt.py
"""
import json
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

MODEL_YAML   = ROOT / 'configs' / 'model.yaml'
FEATURES_DIR = ROOT / 'outputs' / 'features'
CKPT_DIR     = ROOT / 'outputs' / 'checkpoints'
RESULTS_PATH = ROOT / 'outputs' / 'results' / 'ablations.json'
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 64
EPOCHS     = 5
LR         = 1e-3
MAP_DIM    = 8    # A2 weather/traffic only
NPZ_PREFIX = 'fopt'
CKPT_NAME  = 'fopt.pt'
CONFIG_KEY = 'F_opt'


def load_model_config(map_input_dim=MAP_DIM):
    with open(MODEL_YAML, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    arch = cfg['architecture']
    input_dim = len(arch['input_features'])
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


def load_split(split: str):
    path = FEATURES_DIR / f'{NPZ_PREFIX}_{split}.npz'
    if not path.exists():
        sys.exit(
            f'\n[ERROR] Missing feature file: {path}\n'
            f'Run: python scripts/precompute_features_fopt.py\n'
        )
    data = np.load(path)
    ego      = np.concatenate([data['ego_pos'],       data['ego_neg']],       axis=0)
    neighbors= np.concatenate([data['neighbors_pos'], data['neighbors_neg']], axis=0)
    mask     = np.concatenate([data['mask_pos'],      data['mask_neg']],      axis=0)
    map_feat = np.concatenate([data['map_pos'],       data['map_neg']],       axis=0).astype(np.float32)
    labels   = np.concatenate([data['labels_pos'],    data['labels_neg']],    axis=0)
    return (
        torch.from_numpy(ego.astype(np.float32)),
        torch.from_numpy(neighbors.astype(np.float32)),
        torch.from_numpy(mask),
        torch.from_numpy(map_feat),
        torch.from_numpy(labels.astype(np.float32)),
    )


def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for ego, neigh, mask, mapf, labels in loader:
            out = model(
                ego.to(device), neigh.to(device), mask.to(device), mapf.to(device)
            )
            all_probs.extend(out.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
    all_probs  = np.clip(np.array(all_probs), 1e-6, 1.0 - 1e-6)
    all_labels = np.array(all_labels)
    return {
        'auprc':     float(average_precision_score(all_labels, all_probs)),
        'brier':     float(brier_score_loss(all_labels, all_probs)),
        'log_loss':  float(log_loss(all_labels, all_probs)),
        'mean_prob': float(all_probs.mean()),
    }


def main():
    device = torch.device('cpu')
    cfg    = load_model_config()
    model  = CRESTModel(cfg).to(device)
    print(f'[F_opt] map_input_dim={MAP_DIM}  parameters={count_parameters(model):,}')

    train_ds   = TensorDataset(*load_split('train'))
    cal_ds     = TensorDataset(*load_split('cal'))
    holdout_ds = TensorDataset(*load_split('holdout'))

    train_loader   = DataLoader(train_ds,   batch_size=BATCH_SIZE, shuffle=True)
    cal_loader     = DataLoader(cal_ds,     batch_size=BATCH_SIZE, shuffle=False)
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
            out  = model(ego.to(device), neigh.to(device), mask.to(device), mapf.to(device))
            loss = criterion(out, labels.to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1
        elapsed = time.time() - t0
        train_m = evaluate(model, train_loader, device)
        cal_m   = evaluate(model, cal_loader,   device)
        print(
            f'[F_opt] Epoch {epoch}/{EPOCHS}  loss={epoch_loss/n_batches:.4f}  '
            f'train_auprc={train_m["auprc"]:.4f}  cal_auprc={cal_m["auprc"]:.4f}  '
            f'({elapsed:.1f}s)'
        )
        history.append({
            'epoch': epoch,
            'train': train_m,
            'cal':   cal_m,
            'loss':  epoch_loss / n_batches,
        })

    # Save checkpoint
    ckpt_path = CKPT_DIR / CKPT_NAME
    torch.save(model.state_dict(), ckpt_path)
    print(f'\n[F_opt] Checkpoint saved → {ckpt_path}')

    # Platt scaling (logistic calibration on cal split)
    model.eval()
    cal_logits, cal_labels = [], []
    with torch.no_grad():
        for ego, neigh, mask, mapf, labels in cal_loader:
            probs = model(
                ego.to(device), neigh.to(device), mask.to(device), mapf.to(device)
            ).cpu().numpy().ravel()
            probs = np.clip(probs, 1e-6, 1 - 1e-6)
            cal_logits.extend(np.log(probs / (1 - probs)).tolist())
            cal_labels.extend(labels.numpy().tolist())
    platt_a, platt_b = _sigmoid_calibration(np.array(cal_logits), np.array(cal_labels))
    print(f'[F_opt] Platt scaling: a={platt_a:.4f}  b={platt_b:.4f}')

    # Holdout evaluation
    holdout_m = evaluate(model, holdout_loader, device)
    print(f'\n[F_opt] HOLDOUT  AUPRC={holdout_m["auprc"]:.4f}  Brier={holdout_m["brier"]:.4f}')

    # Merge into ablations.json
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, encoding='utf-8') as f:
            results = json.load(f)
    else:
        results = {}

    results[CONFIG_KEY] = {
        'config':        CONFIG_KEY,
        'description':   'F_opt: A2 weather/traffic (8-dim map) + A4_r500 RSU CPMs, no V2V, no A1 elevation',
        'map_input_dim': MAP_DIM,
        'checkpoint':    str(ckpt_path),
        'history':       history,
        'platt':         {'a': float(platt_a), 'b': float(platt_b)},
        'holdout':       holdout_m,
    }

    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f'[F_opt] Results written → {RESULTS_PATH}  (key: "{CONFIG_KEY}")')

    # Summary table
    print('\n' + '='*55)
    print('F_opt TRAINING COMPLETE')
    print('='*55)
    print(f'  AUPRC  : {holdout_m["auprc"]:.4f}')
    print(f'  Brier  : {holdout_m["brier"]:.4f}')
    print(f'  LogLoss: {holdout_m["log_loss"]:.4f}')
    compare = {
        'B1': 0.678, 'A2': 0.751, 'A4_r500': 0.722, 'F': 0.700,
    }
    print('\n  Context (holdout AUPRC):')
    for k, v in compare.items():
        print(f'    {k:<10s}: {v:.3f}')
    print(f'    {"F_opt":<10s}: {holdout_m["auprc"]:.3f}  ← this run')


if __name__ == '__main__':
    main()
