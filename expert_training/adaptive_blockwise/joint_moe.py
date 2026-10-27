"""Per-example differentiable routing of the existing shared/private LoRA pool.

No labels, assignment columns or signatures are accepted by the routing API.
The classifier loss is computed by the training loop outside this module.
"""
import types

import torch
from torch import nn
from torch.nn import functional as F

from .modeling import SharedPrivateLoRALinear


class SampleRouter(nn.Module):
    def __init__(self, hidden_size, groups, width=128):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, width),
                                 nn.SiLU(), nn.Linear(width, groups))
        # Begin as an equal mixture of trained experts, not a saturated old teacher surrogate.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        return self.net(features.float())


def mixture_forward(self, hidden):
    output = self.base(hidden)
    context = self.route_context
    if self.block_id not in context.probabilities:
        raise RuntimeError("A block-entry router must run before its LoRA experts")
    weights = context.probabilities[self.block_id].to(hidden.device)
    dropped = self.dropout(hidden).to(self.shared_A.weight.dtype)
    shared = self.shared_B(self.shared_A(dropped)) * self.scaling_shared
    private = torch.zeros_like(shared)
    for group in range(self.num_groups):
        update = self.private_B[group](self.private_A[group](dropped)) * self.scaling_private
        private = private + weights[:, group, None, None] * update
    return output + (shared + private).to(output.dtype)


class MixtureContext:
    def __init__(self, routers, blocks):
        self.routers = routers
        self.blocks = blocks
        self.mask = None
        self.mode = "soft"
        self.probabilities = {}
        self.handles = []

    def attach(self, layers):
        for block in self.blocks:
            b = int(block["block_id"])

            def hook(_module, args, block_id=b):
                hidden = args[0]
                if self.mask is None:
                    raise RuntimeError("Missing attention mask")
                mask = self.mask.to(hidden.device).float().unsqueeze(-1)
                features = (hidden.float() * mask).sum(1) / mask.sum(1).clamp_min(1.)
                if str(block_id) not in self.routers:
                    probs = torch.ones((hidden.shape[0], 1), device=hidden.device)
                else:
                    router = self.routers[str(block_id)]
                    probs = router(features.to(next(router.parameters()).device)).softmax(-1)
                    if self.mode == "uniform":
                        probs = torch.ones_like(probs) / probs.shape[-1]
                    elif self.mode == "hard":
                        probs = F.one_hot(probs.argmax(-1), probs.shape[-1]).float()
                    elif self.mode != "soft":
                        raise ValueError(f"Unknown routing mode {self.mode}")
                self.probabilities[block_id] = probs

            self.handles.append(layers[int(block["start"])].register_forward_pre_hook(hook))


class JointMoE(nn.Module):
    def __init__(self, core, blocks, num_groups, router_width=128):
        super().__init__()
        self.core = core
        core.route_context.close()
        self.routers = nn.ModuleDict()
        for block in blocks:
            b = int(block["block_id"])
            k = int(num_groups[str(b)])
            if k > 1:
                device = next(core.decoder_layers[int(block["start"])].parameters()).device
                self.routers[str(b)] = SampleRouter(core.model.config.hidden_size, k, router_width).to(device)
        self.context = MixtureContext(self.routers, blocks)
        for module in core.modules():
            if isinstance(module, SharedPrivateLoRALinear):
                module.route_context = self.context
                module.forward = types.MethodType(mixture_forward, module)
        self.context.attach(core.decoder_layers)

    def forward(self, input_ids, attention_mask, mode="soft"):
        self.context.mask = attention_mask
        self.context.mode = mode
        self.context.probabilities = {}
        # Intentionally do not accept labels: inference is label-independent by construction.
        output = self.core(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = [p for p in self.context.probabilities.values() if p.shape[-1] > 1]
        balance = output.logits.float().sum() * 0.
        for p in probabilities:
            balance = balance + p.shape[-1] * p.mean(0).square().sum() - 1.
        return output.logits.float(), balance / max(len(probabilities), 1)

    def trainable_state(self):
        return {name: p.detach().cpu().clone() for name, p in self.named_parameters() if p.requires_grad}

    def load_trainable_state(self, state):
        params = dict(self.named_parameters())
        expected = {name for name, p in params.items() if p.requires_grad}
        if set(state) != expected:
            raise ValueError(f"Trainable checkpoint keys differ: {set(state) ^ expected}")
        with torch.no_grad():
            for name, value in state.items():
                params[name].copy_(value.to(params[name].device, dtype=params[name].dtype))
