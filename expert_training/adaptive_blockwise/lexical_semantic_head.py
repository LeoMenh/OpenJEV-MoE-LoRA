"""A text-only lexical input modulates a frozen semantic classifier."""
import torch
from torch import nn
from torch.nn import functional as F


class LexicalSemanticHead(nn.Module):
    def __init__(self,vocabulary,hidden=1024,classes=738,rank=128,dropout=.5):
        super().__init__()
        self.words=nn.EmbeddingBag(vocabulary,rank,mode='sum')
        nn.init.normal_(self.words.weight,std=.02)
        self.lexical=nn.Linear(rank,hidden,bias=False)
        nn.init.zeros_(self.lexical.weight)
        self.classifier=nn.Linear(hidden,classes)
        self.dropout=dropout

    def encode_words(self,indices,offsets,weights):
        return self.words(indices,offsets,per_sample_weights=weights)

    def forward(self,semantic,lexical):
        hidden=semantic+self.lexical(F.dropout(lexical,.2,self.training))
        return self.classifier(F.dropout(F.gelu(hidden),self.dropout,self.training))
