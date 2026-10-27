"""Candidate readout: preserve last-token logits and learn a pooled residual.

Isolated from running experiments. Routing remains the text-only JointMoE
shared/private LoRA mixture. A zero output projection preserves the transferred
classifier function initially; no labels enter this forward API.
"""
import math

import torch
from torch import nn

from .joint_moe import JointMoE


class ResidualAttentionReadout(nn.Module):
    def __init__(self, hidden_size, num_labels, heads=4):
        super().__init__()
        if hidden_size < 1 or num_labels < 1 or heads < 1:
            raise ValueError("Readout dimensions must be positive")
        self.queries = nn.Parameter(torch.empty(heads, hidden_size))
        nn.init.normal_(self.queries, std=0.02)
        self.score = nn.Linear(hidden_size * heads, num_labels, bias=False)
        nn.init.zeros_(self.score.weight)

    def forward(self, hidden, attention_mask):
        if hidden.ndim != 3 or attention_mask.shape != hidden.shape[:2]:
            raise ValueError("Expected hidden [batch, tokens, hidden] and matching mask")
        mask = attention_mask.bool()
        if not mask.any(dim=1).all():
            raise ValueError("Every example needs at least one valid token")
        # Explicit FP32 reductions keep masking and long-context softmax stable.
        hidden = hidden.float()
        scores = torch.einsum("bth,qh->bqt", hidden, self.queries.float())
        scores = scores / math.sqrt(hidden.shape[-1])
        weights = scores.masked_fill(~mask[:, None, :], -torch.inf).softmax(-1)
        pooled = torch.einsum("bqt,bth->bqh", weights, hidden).flatten(1)
        return self.score(pooled), weights


class PooledJointMoE(JointMoE):
    def __init__(self, core, blocks, num_groups, router_width=128, pool_heads=4):
        super().__init__(core, blocks, num_groups, router_width)
        classifier = core.model
        self.readout = ResidualAttentionReadout(
            classifier.config.hidden_size, classifier.config.num_labels, pool_heads
        ).to(device=classifier.score.weight.device, dtype=torch.float32)

    def forward(self, input_ids, attention_mask, mode="soft"):
        if input_ids.shape != attention_mask.shape or input_ids.ndim != 2:
            raise ValueError("Expected matching 2D token IDs and attention mask")
        if not attention_mask.bool().any(dim=1).all():
            raise ValueError("All-padding examples are unsupported")
        self.context.mask = attention_mask
        self.context.mode = mode
        self.context.probabilities = {}
        classifier = self.core.model
        hidden = classifier.model(input_ids=input_ids, attention_mask=attention_mask,
                                  use_cache=False, return_dict=True).last_hidden_state
        positions = torch.arange(input_ids.shape[1], device=hidden.device)
        last = positions[None, :].expand_as(attention_mask).masked_fill(
            ~attention_mask.to(hidden.device).bool(), -1).max(dim=1).values
        features = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
        logits = classifier.score(features.to(classifier.score.weight.dtype)).float()
        residual, _ = self.readout(hidden, attention_mask.to(hidden.device))
        logits = logits + residual
        probabilities = [p for p in self.context.probabilities.values() if p.shape[-1] > 1]
        balance = logits.sum() * 0.
        for p in probabilities:
            balance = balance + p.shape[-1] * p.mean(0).square().sum() - 1.
        return logits, balance / max(len(probabilities), 1)

    def transfer_original_state(self, state):
        """Strict migration; ordinary resume uses inherited exact-key loading."""
        params = dict(self.named_parameters())
        expected = {n for n, p in params.items() if p.requires_grad and not n.startswith("readout.")}
        if set(state) != expected:
            raise ValueError("Original checkpoint keys do not match the inherited MoE")
        for name, value in state.items():
            if value.shape != params[name].shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid inherited tensor: {name}")
        if torch.count_nonzero(self.readout.score.weight):
            raise ValueError("Function-preserving migration requires a zero residual")
        with torch.no_grad():
            for name, value in state.items():
                params[name].copy_(value.to(params[name].device, dtype=params[name].dtype))
