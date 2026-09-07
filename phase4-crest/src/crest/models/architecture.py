"""CREST PyTorch architecture.

Target: < 1M parameters total.
Components:
- Ego encoder: MLP over per-vehicle features.
- V2V / neighbor encoder: shared MLP over top-K nearest neighbors, sum-pooled.
- Map encoder: MLP over static road context.
- Fusion: MLP combining ego + V2V + map -> hazard probability.
"""
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn


@dataclass
class CRESTConfig:
    ego_input_dim: int = 6
    ego_hidden_dims: List[int] = None
    ego_output_dim: int = 64

    neighbor_input_dim: int = 6
    neighbor_top_k: int = 10
    neighbor_hidden_dims: List[int] = None
    neighbor_output_dim: int = 32

    map_input_dim: int = 8
    map_hidden_dims: List[int] = None
    map_output_dim: int = 8

    fusion_hidden_dims: List[int] = None
    dropout: float = 0.1

    def __post_init__(self):
        if self.ego_hidden_dims is None:
            self.ego_hidden_dims = [64, 64]
        if self.neighbor_hidden_dims is None:
            self.neighbor_hidden_dims = [32, 32]
        if self.map_hidden_dims is None:
            self.map_hidden_dims = [16, 8]
        if self.fusion_hidden_dims is None:
            self.fusion_hidden_dims = [64, 32]


def _build_mlp(input_dim: int, hidden_dims: List[int], output_dim: int, dropout: float) -> nn.Sequential:
    layers = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class EgoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout: float):
        super().__init__()
        self.net = _build_mlp(input_dim, hidden_dims, output_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NeighborEncoder(nn.Module):
    """Shared MLP per neighbor; output sum-pooled across K neighbors."""

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout: float):
        super().__init__()
        self.net = _build_mlp(input_dim, hidden_dims, output_dim, dropout)

    def forward(self, neighbor_features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # neighbor_features: (batch, K, F)
        encoded = self.net(neighbor_features)  # (batch, K, output_dim)
        if mask is not None:
            encoded = encoded * mask.unsqueeze(-1).float()
            summed = encoded.sum(dim=1)
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1).float()
            return summed / denom
        return encoded.sum(dim=1)


class MapEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout: float):
        super().__init__()
        self.net = _build_mlp(input_dim, hidden_dims, output_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CRESTModel(nn.Module):
    """Full CREST hazard-prediction model."""

    def __init__(self, config: CRESTConfig):
        super().__init__()
        self.config = config
        self.ego_encoder = EgoEncoder(
            config.ego_input_dim, config.ego_hidden_dims, config.ego_output_dim, config.dropout
        )
        self.neighbor_encoder = NeighborEncoder(
            config.neighbor_input_dim, config.neighbor_hidden_dims, config.neighbor_output_dim, config.dropout
        )
        self.map_encoder = MapEncoder(
            config.map_input_dim, config.map_hidden_dims, config.map_output_dim, config.dropout
        )
        fusion_input_dim = config.ego_output_dim + config.neighbor_output_dim + config.map_output_dim
        self.fusion = _build_mlp(fusion_input_dim, config.fusion_hidden_dims, 1, config.dropout)

    def forward(
        self,
        ego_features: torch.Tensor,
        neighbor_features: torch.Tensor,
        neighbor_mask: torch.Tensor,
        map_features: torch.Tensor,
    ) -> torch.Tensor:
        ego = self.ego_encoder(ego_features)
        v2v = self.neighbor_encoder(neighbor_features, neighbor_mask)
        road = self.map_encoder(map_features)
        fused = torch.cat([ego, v2v, road], dim=-1)
        logit = self.fusion(fused)
        return torch.sigmoid(logit).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    cfg = CRESTConfig()
    model = CRESTModel(cfg)
    n_params = count_parameters(model)
    print(f'CREST model parameter count: {n_params:,}')
    assert n_params < 1_000_000, 'Model exceeds 1M parameter budget'

    batch = 4
    ego = torch.randn(batch, cfg.ego_input_dim)
    neighbors = torch.randn(batch, cfg.neighbor_top_k, cfg.neighbor_input_dim)
    mask = torch.ones(batch, cfg.neighbor_top_k, dtype=torch.bool)
    map_feat = torch.randn(batch, cfg.map_input_dim)
    out = model(ego, neighbors, mask, map_feat)
    print('Output shape:', out.shape)
