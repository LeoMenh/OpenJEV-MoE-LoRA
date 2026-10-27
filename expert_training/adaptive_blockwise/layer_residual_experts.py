"""Residual experts that retain information from separate MoE-LoRA depths."""
import torch
from torch import nn


class LayerResidualExperts(nn.Module):
    def __init__(self, hidden=1024, width=256, classes=738, mode='gated', dropout=.5):
        super().__init__()
        if mode not in ('final', 'uniform', 'gated'):
            raise ValueError(mode)
        self.mode = mode
        self.layers = (2,) if mode == 'final' else (0, 1, 2)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(10240, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, classes))
            for _ in self.layers
        ])
        for expert in self.experts:
            nn.init.zeros_(expert[-1].weight)
            nn.init.zeros_(expert[-1].bias)
        self.router = nn.Linear(hidden, len(self.layers)) if mode == 'gated' else None
        if self.router is not None:
            nn.init.zeros_(self.router.weight)
            nn.init.zeros_(self.router.bias)

    def forward(self, normalized_views, inherited_hidden):
        x = normalized_views.reshape(-1, 3, 10240)
        outputs = torch.stack([expert(x[:, layer]) for layer, expert in zip(self.layers, self.experts)], 1)
        if self.router is None:
            weights = outputs.new_full(outputs.shape[:2], 1/len(self.layers))
        else:
            weights = self.router(inherited_hidden).softmax(-1)
        return (outputs*weights[:, :, None]).sum(1), weights
