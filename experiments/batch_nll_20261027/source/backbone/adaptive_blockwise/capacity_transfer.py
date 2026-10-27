"""Warm-start larger-rank/additional-module MoE LoRA without changing its initial function.

Caller must preserve the frozen base, tokenizer, blocks, expert counts, router
architecture and inference mode. This migrates trainable weights only, not an
optimizer or training cursor. Verify real-model numerical agreement separately.
"""
import torch
from .modeling import SharedPrivateLoRALinear


@torch.no_grad()
def transfer_expanded_joint_state(target, source_state, source_alpha):
    if source_alpha <= 0:
        raise ValueError("Source alpha must be positive")
    parameters = {name: p for name, p in target.named_parameters() if p.requires_grad}
    planned, consumed = {}, set()
    old_pairs, new_pairs = 0, 0
    for prefix, module in target.named_modules():
        if not isinstance(module, SharedPrivateLoRALinear):
            continue
        pairs = [("shared_A.weight", "shared_B.weight", module.scaling_shared)]
        pairs += [(f"private_A.{i}.weight", f"private_B.{i}.weight", module.scaling_private)
                  for i in range(module.num_groups)]
        present = [f"{prefix}.{name}" in source_state for a, b, _ in pairs for name in (a, b)]
        if any(present) and not all(present):
            raise ValueError(f"Incomplete source expert topology for {prefix}")
        for a, b, new_scale in pairs:
            a_key, b_key = f"{prefix}.{a}", f"{prefix}.{b}"
            a_target, b_target = parameters[a_key], parameters[b_key]
            if new_scale <= 0:
                raise ValueError("Target LoRA scale must be positive")
            if not any(present):
                # Extra modules begin as zero updates but retain random A so B
                # receives a useful gradient at the first training step.
                planned[a_key] = a_target.detach().clone()
                planned[b_key] = torch.zeros_like(b_target)
                new_pairs += 1
                continue
            old_a, old_b = source_state[a_key], source_state[b_key]
            if old_a.ndim != 2 or old_b.ndim != 2:
                raise ValueError("LoRA matrices must be rank-2 tensors")
            old_rank = old_a.shape[0]
            if old_rank < 1 or old_b.shape[1] != old_rank:
                raise ValueError("Source A/B rank mismatch")
            if a_target.shape[0] < old_rank or b_target.shape[1] != a_target.shape[0]:
                raise ValueError("Cannot shrink rank in an exact expansion")
            if old_a.shape[1] != a_target.shape[1] or old_b.shape[0] != b_target.shape[0]:
                raise ValueError("Frozen linear input/output dimensions changed")
            expanded_a = a_target.detach().clone()
            expanded_b = torch.zeros_like(b_target)
            expanded_a[:old_rank].copy_(old_a)
            # new_scale * B_new * A_new = (old_alpha/old_rank) * B_old * A_old
            expanded_b[:, :old_rank].copy_(old_b * ((float(source_alpha) / old_rank) / new_scale))
            planned[a_key], planned[b_key] = expanded_a, expanded_b
            consumed.update((a_key, b_key))
            old_pairs += 1
    for key, parameter in parameters.items():
        if key in planned:
            continue
        if key not in source_state or source_state[key].shape != parameter.shape:
            raise ValueError(f"Missing or incompatible classifier/router weight: {key}")
        planned[key] = source_state[key]
        consumed.add(key)
    if consumed != set(source_state):
        raise ValueError(f"Unconsumed source weights: {sorted(set(source_state) - consumed)[:5]}")
    if not all(torch.isfinite(value).all() for value in planned.values()):
        raise ValueError("Non-finite transferred weights")
    # Validation finishes before changing any target parameter.
    for key, value in planned.items():
        parameters[key].copy_(value)
    return {"migrated_lora_pairs": old_pairs, "new_zero_update_pairs": new_pairs,
            "source_parameter_tensors": len(source_state), "target_parameter_tensors": len(parameters),
            "optimizer_migrated": False}
