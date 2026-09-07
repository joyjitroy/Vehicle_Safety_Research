"""Train a CREST B1 baseline (ego + neighbors; map features zeroed)."""
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
OUT_DIR = ROOT / 'outputs' / 'checkpoints'
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_PATH = OUT_DIR / 'baseline.pt'

BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3
MAP_DIM = 8


def load_model_config(path: Path):
    with open(path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    arch = cfg['architecture']
    return CRESTConfig(
        ego_input_dim=len(arch['input_features']),
        ego_hidden_dims=arch['ego_encoder']['hidden_dims'],
        ego_output_dim=arch['ego_encoder']['output_dim'],
        neighbor_input_dim=len(arch['input_features']),
        neighbor_top_k=arch['neighbor_encoder']['top_k'],
        neighbor_hidden_dims=arch['neighbor_encoder']['hidden_dims'],
        neighbor_output_dim=arch['neighbor_encoder']['output_dim'],
        map_input_dim=arch['map_encoder']['input_dim'],
        map_hidden_dims=arch['map_encoder']['hidden_dims'],
        map_output_dim=arch['map_encoder']['output_dim'],
        fusion_hidden_dims=arch['fusion']['hidden_dims'],
        dropout=arch.get('dropout', 0.1),
    )


def load_split_tensors(split: str):
    path = FEATURES_DIR / f'{split}.npz'
    data = np.load(path)
    ego = np.concatenate([data['ego_pos'], data['ego_neg']], axis=0)
    neighbors = np.concatenate([data['neighbors_pos'], data['neighbors_neg']], axis=0)
    mask = np.concatenate([data['mask_pos'], data['mask_neg']], axis=0)
    labels = np.concatenate([data['labels_pos'], data['labels_neg']], axis=0)
    map_feat = np.zeros((len(labels), MAP_DIM), dtype=np.float32)
    return (
        torch.from_numpy(ego),
        torch.from_numpy(neighbors),
        torch.from_numpy(mask),
        torch.from_numpy(map_feat),
        torch.from_numpy(labels),
    )


def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
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
    """Fit Platt scaling coefficients on validation logits."""
    logits = np.asarray(logits).reshape(-1, 1)
    labels = np.asarray(labels)
    ab = _sigmoid_calibration(logits.ravel(), labels)
    return ab


def main():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    device = torch.device('cpu')
    cfg = load_model_config(MODEL_YAML)
    model = CRESTModel(cfg).to(device)
    print(f'Model parameters: {count_parameters(model):,}')

    train_tensors = load_split_tensors('train')
    cal_tensors = load_split_tensors('cal')

    train_ds = TensorDataset(*train_tensors)
    cal_ds = TensorDataset(*cal_tensors)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    cal_loader = DataLoader(cal_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()

    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
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
        print(f'Epoch {epoch}/{EPOCHS} — loss={epoch_loss/n_batches:.4f}, train_auprc={train_metrics["auprc"]:.4f}, '
              f'cal_auprc={cal_metrics["auprc"]:.4f}, cal_brier={cal_metrics["brier"]:.4f} ({train_time:.1f}s)')
        history.append({'epoch': epoch, 'train': train_metrics, 'cal': cal_metrics, 'loss': epoch_loss / n_batches})

    # Save checkpoint
    torch.save(model.state_dict(), CKPT_PATH)
    with open(OUT_DIR / 'history.json', 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    # Fit Platt scaling on calibration set
    model.eval()
    cal_logits, cal_labels = [], []
    with torch.no_grad():
        for ego, neigh, mask, mapf, labels in cal_loader:
            out = model(ego.to(device), neigh.to(device), mask.to(device), mapf.to(device)).cpu().numpy()
            eps = 1e-6
            probs = np.clip(out.ravel(), eps, 1 - eps)
            logits = np.log(probs / (1 - probs))
            cal_logits.extend(logits.tolist())
            cal_labels.extend(labels.numpy().tolist())
    cal_logits = np.array(cal_logits)
    cal_labels = np.array(cal_labels)
    platt_a, platt_b = fit_platt(cal_logits, cal_labels)
    platt = {'a': float(platt_a), 'b': float(platt_b)}
    with open(OUT_DIR / 'platt.json', 'w', encoding='utf-8') as f:
        json.dump(platt, f, indent=2)

    print(f'\nTraining complete. Checkpoint saved to {CKPT_PATH}')
    print(f'Platt scaling: a={platt_a:.6f}, b={platt_b:.6f}')
    print(f'Final cal metrics: {cal_metrics}')
    print('DONE')


if __name__ == '__main__':
    main()
