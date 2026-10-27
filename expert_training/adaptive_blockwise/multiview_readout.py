"""Read frozen MoE features using view-specific projections and a sample gate."""
import torch
from torch import nn


class MultiViewReadout(nn.Module):
    def __init__(self, views, view_size, classes, width=512, dropout=0.4, gated=False):
        super().__init__()
        self.views = views
        self.view_size = view_size
        self.gated = gated
        self.norm = nn.LayerNorm(view_size)
        self.projections = nn.ModuleList(
            nn.Linear(view_size, width) for _ in range(views)
        )
        self.gate = nn.Linear(width * views, views) if gated else None
        self.classifier = nn.Sequential(
            nn.GELU(), nn.Dropout(dropout), nn.Linear(width * views, classes)
        )

    def forward(self, x):
        x = self.norm(x.reshape(-1, self.views, self.view_size))
        h = torch.stack([p(x[:, i]) for i, p in enumerate(self.projections)], dim=1)
        if self.gate is not None:
            weights = self.gate(h.flatten(1)).softmax(-1) * self.views
            h = h * weights[:, :, None]
        return self.classifier(h.flatten(1))
