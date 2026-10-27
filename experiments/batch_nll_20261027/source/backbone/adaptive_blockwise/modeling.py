import math
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .io_utils import TARGET_MODULES, atomic_torch_save
from .signals import infer_layer_id, load_base_classifier


def find_decoder_layers(model, expected_layers=40):
    candidates = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) == expected_layers:
            candidates.append((name, module))
    if not candidates:
        raise ValueError(f"Could not find a {expected_layers}-layer ModuleList in the classifier backbone")
    candidates.sort(key=lambda item: (sum(1 for child in item[1] if hasattr(child, "self_attn")), len(item[0])))
    name, layers = candidates[-1]
    return list(layers), name


class BlockRouteContext:
    def __init__(self, layer_to_block):
        self.layer_to_block = {int(layer): int(block) for layer, block in layer_to_block.items()}
        self.routes = {}
        self.attention_mask = None
        self.routers = None
        self.forced_routes = None
        self.router_logits = {}
        self.router_probabilities = {}
        self.handles = []

    def set_teacher_routes(self, routes):
        self.forced_routes = {int(block): value for block, value in routes.items()}
        self.routers = None
        self.routes = {int(block): value for block, value in routes.items()}
        self.router_logits = {}
        self.router_probabilities = {}

    def set_online(self, routers, forced_routes=None):
        self.routers = routers
        self.forced_routes = None if forced_routes is None else {int(block): value for block, value in forced_routes.items()}
        self.routes = {}
        self.router_logits = {}
        self.router_probabilities = {}

    def clear(self):
        self.routes = {}
        self.forced_routes = None
        self.routers = None
        self.attention_mask = None
        self.router_logits = {}
        self.router_probabilities = {}

    def attach(self, layers):
        for layer_id, layer in enumerate(layers):
            self.handles.append(layer.register_forward_pre_hook(self._make_hook(layer_id)))

    def _make_hook(self, layer_id):
        block_id = self.layer_to_block[layer_id]

        def hook(_module, args):
            if not args:
                return
            # A block route is decided exactly once at its entry and remains
            # fixed for every later layer in the same block.
            if block_id in self.routes:
                return
            hidden = args[0]
            forced = self.forced_routes is not None and block_id in self.forced_routes
            if self.routers is None and not forced:
                return
            if self.routers is not None:
                mask = self.attention_mask.to(hidden.device).float().unsqueeze(-1)
                pooled = (hidden.float() * mask).sum(1) / mask.sum(1).clamp_min(1.0)
                router = self.routers[str(block_id)]
                router_parameter = next(router.parameters())
                logits = router(
                    pooled.to(device=router_parameter.device, dtype=router_parameter.dtype)
                )
                probabilities = F.softmax(logits, dim=-1)
                self.router_logits[block_id] = logits
                self.router_probabilities[block_id] = probabilities
                predicted = probabilities.argmax(dim=-1).detach()
            else:
                predicted = None
            if forced:
                self.routes[block_id] = self.forced_routes[block_id].to(hidden.device, dtype=torch.long)
            else:
                self.routes[block_id] = predicted

        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def get(self, layer_id):
        block = self.layer_to_block[int(layer_id)]
        return self.routes.get(block)


class SharedPrivateLoRALinear(nn.Module):
    def __init__(self, base, layer_id, block_id, num_groups, r_shared, r_private, alpha, dropout):
        super().__init__()
        self.base = base
        self.layer_id = int(layer_id)
        self.block_id = int(block_id)
        self.num_groups = int(num_groups)
        self.scaling_shared = float(alpha) / max(int(r_shared), 1)
        self.scaling_private = float(alpha) / max(int(r_private), 1)
        self.dropout = nn.Dropout(float(dropout))
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.shared_A = nn.Linear(base.in_features, int(r_shared), bias=False)
        self.shared_B = nn.Linear(int(r_shared), base.out_features, bias=False)
        self.private_A = nn.ModuleList([nn.Linear(base.in_features, int(r_private), bias=False) for _ in range(num_groups)])
        self.private_B = nn.ModuleList([nn.Linear(int(r_private), base.out_features, bias=False) for _ in range(num_groups)])
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.shared_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.shared_B.weight)
        for module_a, module_b in zip(self.private_A, self.private_B):
            nn.init.kaiming_uniform_(module_a.weight, a=math.sqrt(5))
            nn.init.zeros_(module_b.weight)

    def forward(self, hidden):
        output = self.base(hidden)
        context = getattr(self, "route_context", None)
        if context is None or context.routes.get(self.block_id) is None:
            return output
        if self.shared_A.weight.device != hidden.device:
            self.shared_A.to(hidden.device)
            self.shared_B.to(hidden.device)
            self.private_A.to(hidden.device)
            self.private_B.to(hidden.device)
        dropped = self.dropout(hidden).to(self.shared_A.weight.dtype)
        shared = self.shared_B(self.shared_A(dropped)) * self.scaling_shared
        group_ids = context.routes[self.block_id].to(hidden.device, dtype=torch.long).clamp(0, self.num_groups - 1)
        private = torch.zeros_like(shared)
        for group in range(self.num_groups):
            mask = group_ids == group
            if mask.any():
                private[mask] = self.private_B[group](self.private_A[group](dropped[mask])) * self.scaling_private
        return output + (shared + private).to(output.dtype)


def _block_value(value, block_id, name):
    """Resolve a scalar or per-block mapping while keeping legacy configs valid."""
    if isinstance(value, dict):
        if str(block_id) in value:
            return int(value[str(block_id)])
        if block_id in value:
            return int(value[block_id])
        raise KeyError(f"{name} is missing block {block_id}")
    return int(value)


def replace_with_blockwise_adapters(model, blocks, target_modules, num_groups, r_shared, r_private, alpha, dropout):
    layer_to_block = {}
    for block in blocks:
        for layer in block["layers"]:
            layer_to_block[int(layer)] = int(block["block_id"])
    layers, layer_module_name = find_decoder_layers(model, expected_layers=len(layer_to_block))
    if sorted(layer_to_block) != list(range(len(layers))):
        raise ValueError("Partition layers do not match the decoder layer list")
    context = BlockRouteContext(layer_to_block)
    module_map = dict(model.named_modules())
    replaced = []
    wanted = set(target_modules or TARGET_MODULES)
    for full_name, module in list(module_map.items()):
        short = full_name.rsplit(".", 1)[-1]
        layer_id = infer_layer_id(full_name)
        if short not in wanted or layer_id not in layer_to_block:
            continue
        if not hasattr(module, "in_features") or not hasattr(module, "out_features"):
            continue
        parent_name, child_name = full_name.rsplit(".", 1)
        wrapper = SharedPrivateLoRALinear(
            module,
            layer_id,
            layer_to_block[layer_id],
            _block_value(num_groups, layer_to_block[layer_id], "num_groups"),
            r_shared,
            _block_value(r_private, layer_to_block[layer_id], "r_private"),
            alpha,
            dropout,
        )
        wrapper.route_context = context
        try:
            adapter_device = module.weight.device
            wrapper.shared_A.to(adapter_device)
            wrapper.shared_B.to(adapter_device)
            wrapper.private_A.to(adapter_device)
            wrapper.private_B.to(adapter_device)
        except Exception:
            pass
        setattr(module_map[parent_name], child_name, wrapper)
        replaced.append(full_name)
    if not replaced:
        raise RuntimeError("No target linear modules were replaced; check model architecture and target_modules")
    context.attach(layers)
    return BlockwiseMoEModel(model, context, layers), replaced


def adapter_state_dict(model):
    keep = ("shared_A", "shared_B", "private_A", "private_B", ".score.", ".classifier.", ".classification_head.")
    return {key: value.detach().cpu() for key, value in model.state_dict().items() if any(fragment in key for fragment in keep)}


def load_adapter_state(model, path):
    state = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected adapter state keys: {unexpected[:10]}")
    return missing


class BlockwiseMoEModel(nn.Module):
    def __init__(self, base_model, route_context, layers):
        super().__init__()
        self.model = base_model
        self.route_context = route_context
        self.decoder_layers = layers

    def set_teacher_routes(self, routes):
        self.route_context.set_teacher_routes(routes)

    def set_online(self, routers, attention_mask, forced_routes=None):
        self.route_context.attention_mask = attention_mask
        self.route_context.set_online(routers, forced_routes=forced_routes)

    def clear_routes(self):
        self.route_context.clear()

    def forward(self, **batch):
        return self.model(**batch)

    def save_adapters(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_torch_save(adapter_state_dict(self), output_dir / "blockwise_adapters.pt")


class LinearSoftmaxTop1Router(nn.Module):
    def __init__(self, hidden_size, num_groups=3):
        super().__init__()
        self.linear = nn.Linear(int(hidden_size), int(num_groups))

    def forward(self, pooled):
        return self.linear(pooled)

    def probabilities(self, pooled):
        return F.softmax(self.forward(pooled), dim=-1)


def load_router_states(routers, state):
    """Load the CPU checkpoint emitted by the router-training stage."""
    for key, router in routers.items():
        if key not in state:
            raise ValueError(f"Router checkpoint is missing block {key}")
        router.load_state_dict(state[key])
