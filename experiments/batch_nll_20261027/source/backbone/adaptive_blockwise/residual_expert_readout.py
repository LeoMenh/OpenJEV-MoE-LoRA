"""Sparse patient routing among small residual DRG experts."""
import torch
from torch import nn
from torch.nn import functional as F


class ResidualExpertReadout(nn.Module):
    def __init__(self,hidden,classes,experts=4,rank=64,top_k=2,dropout=.4):
        super().__init__()
        assert 1<=top_k<=experts
        self.top_k=top_k;self.experts=experts;self.dropout=dropout
        self.norm=nn.LayerNorm(hidden)
        self.router=nn.Linear(hidden,experts)
        self.down=nn.Parameter(torch.empty(experts,hidden,rank))
        self.up=nn.Parameter(torch.zeros(experts,rank,classes))
        nn.init.normal_(self.down,std=hidden**-.5)
        nn.init.zeros_(self.router.weight);nn.init.zeros_(self.router.bias)

    def forward(self,hidden,base,uniform=False):
        x=self.norm(hidden)
        soft=self.router(x).softmax(-1)
        if uniform:weights=torch.ones_like(soft)/self.experts
        else:
            # Keep an equal deterministic starting point; random down projections
            # allow experts to diverge once their zero outputs receive gradients.
            v,i=soft.topk(self.top_k,dim=-1)
            weights=torch.zeros_like(soft).scatter(1,i,v/v.sum(-1,keepdim=True))
        low=F.gelu(torch.einsum('bh,ehr->ber',F.dropout(x,self.dropout,self.training),self.down))
        delta=torch.einsum('ber,erc->bec',low,self.up)
        logits=base+(delta*weights[:,:,None]).sum(1)
        balance=self.experts*soft.mean(0).square().sum()-1
        return logits,balance,soft
