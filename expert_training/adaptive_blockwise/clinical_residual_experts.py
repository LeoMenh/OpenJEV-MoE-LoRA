"""Patient-routed low-rank corrections using frozen semantic and clinical features."""
import torch
from torch import nn
from torch.nn import functional as F


class ClinicalResidualExperts(nn.Module):
    def __init__(self,semantic_width=1024,clinical_width=1024,classes=738,experts=4,rank=64,routing='dense',dropout=.4):
        super().__init__()
        if routing not in ('uniform','dense','sparse'):raise ValueError('Unsupported routing')
        self.semantic_width=semantic_width
        self.clinical_width=clinical_width
        self.routing=routing
        self.experts=experts
        self.dropout=dropout
        self.semantic_norm=nn.LayerNorm(semantic_width)
        self.clinical_norm=nn.LayerNorm(clinical_width)
        self.router=nn.Linear(semantic_width+clinical_width,experts)
        self.down=nn.Parameter(torch.empty(experts,semantic_width+clinical_width,rank))
        self.up=nn.Parameter(torch.zeros(experts,rank,classes))
        nn.init.normal_(self.down,std=(semantic_width+clinical_width)**-.5)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)

    def forward(self,semantic,clinical,base_logits):
        x=torch.cat((self.semantic_norm(semantic),self.clinical_norm(clinical)),dim=-1)
        soft=self.router(x).softmax(-1)
        if self.routing=='uniform':
            weights=torch.ones_like(soft)/self.experts
        elif self.routing=='sparse':
            values,ids=soft.topk(min(2,self.experts),dim=-1)
            weights=torch.zeros_like(soft).scatter(1,ids,values/values.sum(-1,keepdim=True))
        else:weights=soft
        low=F.gelu(torch.einsum('bh,ehr->ber',F.dropout(x,self.dropout,self.training),self.down))
        delta=torch.einsum('ber,erc->bec',low,self.up)
        result=base_logits+(weights[:,:,None]*delta).sum(1)
        balance=self.experts*soft.mean(0).square().sum()-1
        return result,balance,soft
