"""Controlled candidate-relative reading on the unchanged frozen prototypes.

The two inherited modes delegate to the original implementation exactly. The
new mode changes only the attention query; the residual scorer still receives
the original projected class prototype. No new parameters or labels are added.
"""
import math

import torch

from .candidate_token_readout import CandidateTokenReadout


MODES = ('uniform', 'label_attention', 'rival_attention')


def rival_queries(queries):
    """Subtract the equally weighted mean of the other current candidates."""
    assert queries.ndim == 3 and queries.shape[1] >= 2
    assert queries.is_floating_point() and torch.isfinite(queries).all()
    count = queries.shape[1]
    others = (queries.sum(1, keepdim=True) - queries) / (count - 1)
    return queries - others


class RivalCandidateTokenReadout(CandidateTokenReadout):
    def __init__(self, prototypes, mode, width=64, dropout=.1, correction_cap=5.):
        assert mode in MODES
        inherited_mode = 'label_attention' if mode == 'rival_attention' else mode
        super().__init__(prototypes, inherited_mode, width, dropout, correction_cap)
        self.mode = mode

    def forward(self, token_features, attention_mask, base_logits, candidates):
        if self.mode != 'rival_attention':
            return super().forward(token_features, attention_mask, base_logits, candidates)
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
        assert candidates.ndim == 2 and candidates.shape[0] == batch
        assert candidates.shape[1] >= 2, 'Candidate-relative reading needs rivals.'
        assert candidates.dtype == torch.long
        assert (candidates >= 0).all() and (candidates < self.prototypes.shape[0]).all()
        ordered = candidates.sort(1).values
        assert (ordered[:, 1:] != ordered[:, :-1]).all(), 'Candidates must be unique.'
        tokens = token_features.float().masked_fill(~valid.unsqueeze(-1), 0.)
        keys = self.projection(self.normalizer(tokens))
        queries = self.projection(self.normalizer(self.prototypes[candidates]))
        # Center in projected-query space. The classifier prototypes themselves
        # and the scorer inputs retain their original class-specific values.
        reading_queries = rival_queries(queries)
        scores = torch.einsum('bkr,btr->bkt', reading_queries, keys) / math.sqrt(self.width)
        weights = scores.masked_fill(~valid[:, None, :], -torch.inf).softmax(-1)
        pooled = torch.einsum('bkt,btr->bkr', weights, keys)
        features = torch.cat([pooled, queries, pooled * queries], dim=-1)
        raw_delta = self.scorer(features).squeeze(-1)
        delta = self.correction_cap * torch.tanh(raw_delta / self.correction_cap)
        return base_logits.float().scatter_add(1, candidates, delta)
