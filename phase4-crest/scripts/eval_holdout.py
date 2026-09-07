"""Evaluate the saved baseline checkpoint on the holdout split."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, precision_recall_curve
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from crest.models.architecture import CRESTConfig, CRESTModel

CKPT_PATH = ROOT / 'outputs' / 'checkpoints' / 'baseline.pt'
FEATURES_PATH = ROOT / 'outputs' / 'features' / 'holdout.npz'
MODEL_YAML = ROOT / 'configs' / 'model.yaml'
OUT_PATH = ROOT / 'outputs' / 'results' / 'holdout_eval.json'
FIGURE_DATA_PATH = ROOT / 'figure_data.json'
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
BATCH_SIZE = 64
MAP_DIM = 8


def load_model_config(path: Path):
    with open(path, encoding='utf-8') as f:
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
        map_input_dim=arch['map_encoder']['input_dim'],
        map_hidden_dims=arch['map_encoder']['hidden_dims'],
        map_output_dim=arch['map_encoder']['output_dim'],
        fusion_hidden_dims=arch['fusion']['hidden_dims'],
        dropout=arch.get('dropout', 0.1),
    )


def main():
    device = torch.device('cpu')
    cfg = load_model_config(MODEL_YAML)
    model = CRESTModel(cfg).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device, weights_only=True))
    model.eval()

    data = np.load(FEATURES_PATH)
    ego = np.concatenate([data['ego_pos'], data['ego_neg']], axis=0)
    neighbors = np.concatenate([data['neighbors_pos'], data['neighbors_neg']], axis=0)
    mask = np.concatenate([data['mask_pos'], data['mask_neg']], axis=0)
    labels = np.concatenate([data['labels_pos'], data['labels_neg']], axis=0)
    map_feat = np.zeros((len(labels), MAP_DIM), dtype=np.float32)

    ds = TensorDataset(
        torch.from_numpy(ego),
        torch.from_numpy(neighbors),
        torch.from_numpy(mask),
        torch.from_numpy(map_feat),
        torch.from_numpy(labels),
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch_ego, batch_neigh, batch_mask, batch_map, batch_labels in loader:
            out = model(
                batch_ego.to(device),
                batch_neigh.to(device),
                batch_mask.to(device),
                batch_map.to(device),
            )
            all_probs.extend(out.cpu().numpy().tolist())
            all_labels.extend(batch_labels.numpy().tolist())

    probs = np.clip(np.array(all_probs), 1e-6, 1.0 - 1e-6)
    labels_arr = np.array(all_labels)

    precision_arr, recall_arr, thresholds_arr = precision_recall_curve(labels_arr, probs)

    results = {
        'n_samples': int(len(labels_arr)),
        'n_positive': int(labels_arr.sum()),
        'n_negative': int(len(labels_arr) - labels_arr.sum()),
        'auprc': float(average_precision_score(labels_arr, probs)),
        'brier': float(brier_score_loss(labels_arr, probs)),
        'log_loss': float(log_loss(labels_arr, probs)),
        'mean_probability': float(probs.mean()),
    }

    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Persist PR curve arrays into figure_data.json
    with open(FIGURE_DATA_PATH, 'r', encoding='utf-8') as f:
        fig_data = json.load(f)

    fig_data['pr_curve_b1'] = {
        'precision': precision_arr.tolist(),
        'recall': recall_arr.tolist(),
        'thresholds': thresholds_arr.tolist(),
        'auprc': float(average_precision_score(labels_arr, probs)),
        'label': 'B1 ego-only',
    }

    with open(FIGURE_DATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(fig_data, f, indent=2)

    print("PR curve saved to figure_data.json")
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
