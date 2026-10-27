"""Candidate-specific token reading anchored to an unchanged clinical classifier.

No labels enter forward. The prototypes are frozen rows of the mature classifier,
not DRG descriptions or verified clinical explanations. Both reading modes have
exactly the same trainable parameters and zero initial logit corrections.
"""
import math

import torch
from torch import nn


def clinical_candidates(logits, count=8):
    assert logits.ndim == 2 and 0 < count <= logits.shape[1]
    assert torch.isfinite(logits).all()
    # Canonical class order breaks equal-score ties without consulting labels.
    return torch.argsort(logits.detach(), dim=1, descending=True, stable=True)[:, :count]


class CandidateTokenReadout(nn.Module):
    def __init__(self, prototypes, mode, width=64, dropout=.1, correction_cap=5.):
        super().__init__()
        assert mode in ('uniform', 'label_attention')
        assert prototypes.ndim == 2 and torch.isfinite(prototypes).all()
        assert width > 0 and correction_cap > 0
        self.register_buffer('prototypes', prototypes.detach().float().clone())
        self.normalizer = nn.LayerNorm(prototypes.shape[1])
        # The same projection supplies token keys/values and label queries.
        self.projection = nn.Linear(prototypes.shape[1], width, bias=False)
        self.scorer = nn.Sequential(nn.Linear(3 * width, width), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(width, 1))
        nn.init.zeros_(self.scorer[-1].weight)
        nn.init.zeros_(self.scorer[-1].bias)
        self.mode, self.width, self.correction_cap = mode, width, correction_cap

    def forward(self, token_features, attention_mask, base_logits, candidates):
        assert token_features.ndim == 3 and base_logits.ndim == 2
        batch, length, hidden = token_features.shape
        assert hidden == self.prototypes.shape[1]
        assert attention_mask.shape == (batch, length)
        assert ((attention_mask == 0) | (attention_mask == 1)).all()
        valid = attention_mask.bool()
        assert valid.any(1).all(), 'Every case must contain a valid token.'
        assert torch.isfinite(token_features[valid]).all()
        assert base_logits.shape == (batch, self.prototypes.shape[0])
        assert torch.isfinite(base_logits).all()
        assert candidates.ndim == 2 and candidates.shape[0] == batch and candidates.shape[1] > 0
        assert candidates.dtype == torch.long
        assert (candidates >= 0).all() and (candidates < self.prototypes.shape[0]).all()
        ordered = candidates.sort(1).values
        assert (ordered[:, 1:] != ordered[:, :-1]).all(), 'Candidates must be unique.'
        # Ignore arbitrary padding values before normalization/projection.
        tokens = token_features.float().masked_fill(~valid.unsqueeze(-1), 0.)
        keys = self.projection(self.normalizer(tokens))
        queries = self.projection(self.normalizer(self.prototypes[candidates]))
        if self.mode == 'label_attention':
            scores = torch.einsum('bkr,btr->bkt', queries, keys) / math.sqrt(self.width)
            weights = scores.masked_fill(~valid[:, None, :], -torch.inf).softmax(-1)
            pooled = torch.einsum('bkt,btr->bkr', weights, keys)
        else:
            pooled = (keys * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True)
            pooled = pooled[:, None, :].expand_as(queries)
        features = torch.cat([pooled, queries, pooled * queries], dim=-1)
        raw_delta = self.scorer(features).squeeze(-1)
        delta = self.correction_cap * torch.tanh(raw_delta / self.correction_cap)
        # Other logits remain exactly as supplied. Softmax can change their
        # absolute probability masses; this is not mass-conserving reranking.
        return base_logits.float().scatter_add(1, candidates, delta)
