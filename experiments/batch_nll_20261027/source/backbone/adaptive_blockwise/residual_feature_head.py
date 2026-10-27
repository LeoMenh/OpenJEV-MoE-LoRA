"""Nonlinear residual classification atop frozen text-only MoE features."""
import torch
from torch import nn


class ResidualFeatureHead(nn.Module):
    def __init__(self, feature_size, num_classes, width=512, dropout=0.2):
        super().__init__()
        self.norm=nn.LayerNorm(feature_size)
        self.hidden=nn.Linear(feature_size,width)
        self.activation=nn.GELU()
        self.dropout=nn.Dropout(dropout)
        self.output=nn.Linear(width,num_classes)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features, base_logits):
        if features.ndim!=2 or base_logits.ndim!=2 or features.shape[0]!=base_logits.shape[0]:
            raise ValueError('Expected matching feature/logit batch')
        residual=self.output(self.dropout(self.activation(self.hidden(self.norm(features.float())))))
        if residual.shape!=base_logits.shape:
            raise ValueError('Base and residual class dimensions differ')
        return base_logits.float()+residual
