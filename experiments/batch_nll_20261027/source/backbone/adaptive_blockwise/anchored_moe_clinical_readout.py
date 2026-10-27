"""Parameter-matched plain/anchored joint readouts of frozen clinical/MoE views."""
import torch
from torch import nn


class AnchoredMoEClinicalReadout(nn.Module):
    def __init__(self, arm, hidden=512, dropout=.5, classes=738):
        super().__init__()
        if arm not in ('moe_only', 'clinical_only', 'joint_plain', 'joint_residual'):
            raise ValueError(arm)
        self.arm = arm
        self.moe_norm = nn.LayerNorm(10240) if arm != 'clinical_only' else None
        self.clinical_norm = nn.LayerNorm(1024) if arm != 'moe_only' else None
        width = {'moe_only': 10240, 'clinical_only': 1024, 'joint_plain': 11264, 'joint_residual': 11264}[arm]
        self.head = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, classes))
        if arm == 'joint_residual':
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)

    def forward(self, moe, clinical, clinical_logits):
        if self.arm in ('joint_plain', 'joint_residual'):
            inputs = torch.cat((self.moe_norm(moe), self.clinical_norm(clinical)), dim=-1)
        elif self.arm == 'clinical_only':
            inputs = self.clinical_norm(clinical)
        else:
            inputs = self.moe_norm(moe)
        logits = self.head(inputs)
        return clinical_logits + logits if self.arm == 'joint_residual' else logits
