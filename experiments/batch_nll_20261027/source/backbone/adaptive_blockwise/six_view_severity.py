"""Extend a two-view conditional severity expert while preserving its function."""
import torch
from torch import nn


class SixViewSeverity(nn.Module):
    def __init__(self, families, dropout=.5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Unflatten(1, (6,5120)), nn.LayerNorm(5120), nn.Flatten(1),
            nn.Linear(30720,512), nn.GELU(), nn.Dropout(dropout))
        self.shared = nn.Linear(512,3)
        self.factor = nn.Linear(512,64,bias=False)
        self.family = nn.Parameter(torch.zeros(families,64,3))

    def inherit(self, state):
        copied = dict(state)
        original = state['encoder.3.weight']
        assert original.shape == (512,10240)
        expanded = original.new_zeros((512,30720))
        expanded[:,20480:] = original
        copied['encoder.3.weight'] = expanded
        self.load_state_dict(copied, strict=True)

    def forward(self,x):
        h = self.encoder(x)
        return self.shared(h)[:,None,:] + torch.einsum('br,frs->bfs',self.factor(h),self.family)
