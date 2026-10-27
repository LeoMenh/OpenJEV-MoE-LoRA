"""Recompute adapted projections, including quantized-base backward intermediates."""
import types
import torch
from torch.utils.checkpoint import checkpoint
from .modeling import SharedPrivateLoRALinear


def low_rank_delta(module, hidden, weights):
    dropped = module.dropout(hidden).to(module.shared_A.weight.dtype)
    shared = module.shared_B(module.shared_A(dropped)) * module.scaling_shared
    private = torch.zeros_like(shared)
    for group in range(module.num_groups):
        update = module.private_B[group](module.private_A[group](dropped)) * module.scaling_private
        private = private + weights[:, group, None, None] * update
    return (shared + private).to(hidden.dtype)


def checkpointed_forward(self, hidden):
    weights = self.route_context.probabilities[self.block_id].to(hidden.device)
    def projection(x, p):
        output = self.base(x)
        return output + low_rank_delta(self, x, p).to(output.dtype)
    if self.training and torch.is_grad_enabled():
        return checkpoint(projection, hidden, weights, use_reentrant=False, preserve_rng_state=True)
    return projection(hidden, weights)


def enable_branch_checkpointing(model):
    count = 0
    for module in model.modules():
        if isinstance(module, SharedPrivateLoRALinear):
            module.forward = types.MethodType(checkpointed_forward, module)
            count += 1
    return count
