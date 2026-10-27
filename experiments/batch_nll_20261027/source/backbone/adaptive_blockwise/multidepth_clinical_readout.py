"""Residual clinical readouts of terminal, uniform or case-weighted depth views."""
import torch
from torch import nn


class MultiDepthClinicalReadout(nn.Module):
    def __init__(self, arm, hidden=512, dropout=.5, classes=738):
        super().__init__()
        if arm not in ('terminal', 'uniform_depth', 'adaptive_depth'):
            raise ValueError(arm)
        self.arm = arm
        self.norm = nn.LayerNorm(2048)
        self.gate = nn.Linear(2048, 1) if arm == 'adaptive_depth' else None
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
        self.head = nn.Sequential(nn.Linear(2048, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, classes))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, features, clinical_logits):
        views = self.norm(features.reshape(-1, 4, 2048))
        if self.arm == 'terminal':
            pooled = views[:, -1]
        elif self.arm == 'uniform_depth':
            pooled = views.mean(1)
        else:
            weights = self.gate(views).squeeze(-1).softmax(1)
            pooled = (weights.unsqueeze(-1) * views).sum(1)
        return clinical_logits + self.head(pooled)
