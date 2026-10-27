"""Supervised metric for retrieving training cases from frozen MoE features."""
import torch
from torch import nn
from torch.nn import functional as F

class MetricReadout(nn.Module):
    def __init__(self, views=2, width=512, classes=738, scale=20., dropout=.3):
        super().__init__()
        self.views=views
        self.scale=scale
        self.encoder=nn.Sequential(nn.Unflatten(1,(views,5120)),nn.LayerNorm(5120),
                                   nn.Flatten(1),nn.Linear(views*5120,width),nn.GELU(),
                                   nn.Dropout(dropout),nn.Linear(width,width))
        self.prototypes=nn.Parameter(torch.randn(classes,width)*.02)

    def encode(self,x):
        return F.normalize(self.encoder(x),dim=-1)

    def forward(self,x):
        e=self.encode(x)
        return self.scale*e@F.normalize(self.prototypes,dim=-1).T,e

def supervised_contrastive(embeddings,labels,temperature=.1):
    score=embeddings@embeddings.T/temperature
    diagonal=torch.eye(len(labels),device=labels.device,dtype=torch.bool)
    positives=(labels[:,None]==labels[None,:])&~diagonal
    valid=positives.any(-1)
    if not valid.any():return embeddings.sum()*0
    score=score.masked_fill(diagonal,-1e4)
    logprob=score.log_softmax(-1)
    return -(logprob*positives).sum(-1)[valid].div(positives.sum(-1)[valid]).mean()
