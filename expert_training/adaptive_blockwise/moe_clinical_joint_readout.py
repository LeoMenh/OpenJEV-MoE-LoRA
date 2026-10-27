"""Small classifiers on separately normalized frozen MoE/domain representations."""
import torch
from torch import nn


class MoEClinicalJointReadout(nn.Module):
    def __init__(self, arm, hidden=512, dropout=.5, classes=738):
        super().__init__()
        if arm not in ('moe_only', 'clinical_only', 'joint'):
            raise ValueError(arm)
        self.arm = arm
        self.moe_norm = nn.LayerNorm(10240) if arm != 'clinical_only' else None
        self.clinical_norm = nn.LayerNorm(1024) if arm != 'moe_only' else None
        size = {'moe_only': 10240, 'clinical_only': 1024, 'joint': 11264}[arm]
        self.head = nn.Sequential(nn.Linear(size, hidden), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, classes))

    def forward(self, moe, clinical):
        if self.arm == 'joint':
            x = torch.cat((self.moe_norm(moe), self.clinical_norm(clinical)), dim=-1)
        elif self.arm == 'moe_only':
            x = self.moe_norm(moe)
        else:
            x = self.clinical_norm(clinical)
        return self.head(x)
