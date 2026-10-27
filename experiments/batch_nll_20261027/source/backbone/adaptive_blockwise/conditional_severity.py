"""Family-conditioned severity sharing. Forward computes every family without labels."""
import torch
from torch import nn

class ConditionalSeverity(nn.Module):
    def __init__(self,families,conditional=True,width=512,rank=64,dropout=.5):
        super().__init__()
        self.families=families
        self.encoder=nn.Sequential(nn.Unflatten(1,(2,5120)),nn.LayerNorm(5120),nn.Flatten(1),
                                   nn.Linear(10240,width),nn.GELU(),nn.Dropout(dropout))
        self.shared=nn.Linear(width,3)
        self.factor=nn.Linear(width,rank,bias=False) if conditional else None
        self.family=nn.Parameter(torch.zeros(families,rank,3)) if conditional else None
        if conditional:nn.init.normal_(self.family,std=.01)

    def forward(self,x):
        h=self.encoder(x);scores=self.shared(h)[:,None,:].expand(-1,self.families,-1)
        if self.factor is not None:scores=scores+torch.einsum('br,frs->bfs',self.factor(h),self.family)
        return scores
