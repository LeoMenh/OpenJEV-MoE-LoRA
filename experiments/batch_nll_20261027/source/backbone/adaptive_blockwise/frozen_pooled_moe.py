"""Warm up the document readout without drifting the inherited MoE-LoRA."""
import torch

from .pooled_joint_moe import PooledJointMoE


class FrozenPooledMoE(PooledJointMoE):
    def freeze_inherited(self):
        self.requires_grad_(False)
        self.readout.requires_grad_(True)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        # No stochastic inherited adapter dropout during readout optimization.
        self.core.eval()
        self.routers.eval()
        return self

    def forward(self, input_ids, attention_mask, mode="soft"):
        if input_ids.shape != attention_mask.shape or input_ids.ndim != 2:
            raise ValueError("Expected matching 2D token IDs and attention mask")
        if not attention_mask.bool().any(dim=1).all():
            raise ValueError("All-padding examples are unsupported")
        self.context.mask = attention_mask
        self.context.mode = mode
        self.context.probabilities = {}
        classifier = self.core.model
        with torch.no_grad():
            hidden = classifier.model(input_ids=input_ids, attention_mask=attention_mask,
                                      use_cache=False, return_dict=True).last_hidden_state
            mask = attention_mask.to(hidden.device).bool()
            last = torch.arange(input_ids.shape[1], device=hidden.device)[None, :].expand_as(mask)
            last = last.masked_fill(~mask, -1).max(dim=1).values
            features = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
            inherited = classifier.score(features.to(classifier.score.weight.dtype)).float()
        residual, _ = self.readout(hidden, attention_mask.to(hidden.device))
        logits = inherited + residual
        # Routing is frozen: no router-balancing loss is optimized in this stage.
        return logits, logits.sum() * 0.
