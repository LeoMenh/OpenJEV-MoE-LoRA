"""Share severity evidence across DRG families, using a trainable family readout."""
import torch
from torch import nn


class HierarchicalFeatureHead(nn.Module):
    def __init__(self, feature_size, family_ids, severity_ids, hidden=1024, dropout=.5):
        super().__init__()
        families = torch.as_tensor(family_ids, dtype=torch.long)
        severity = torch.as_tensor(severity_ids, dtype=torch.long)
        num_families = int(families.max()) + 1
        allowed = torch.zeros(num_families, 3, dtype=torch.bool)
        allowed[families, severity] = True
        if len(set(zip(family_ids, severity_ids))) != len(family_ids):
            raise ValueError('Each class needs a unique family/severity pair')
        self.register_buffer('families', families)
        self.register_buffer('severity', severity)
        self.register_buffer('allowed', allowed)
        self.features = nn.Sequential(nn.LayerNorm(feature_size), nn.Linear(feature_size, hidden),
                                      nn.GELU(), nn.Dropout(dropout))
        self.family_head = nn.Linear(hidden, num_families)
        self.severity_head = nn.Linear(hidden, 3)

    def forward(self, x):
        hidden = self.features(x)
        family = self.family_head(hidden).log_softmax(-1)
        severity = self.severity_head(hidden)[:, None, :].expand(-1, len(self.allowed), -1)
        conditional = severity.masked_fill(~self.allowed, -torch.inf).log_softmax(-1)
        return family[:, self.families] + conditional[:, self.families, self.severity]
