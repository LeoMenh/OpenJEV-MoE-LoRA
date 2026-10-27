"""Exhaustive frozen-expert logits with microbatch-local block-prefix reuse.

Only classification logits are observable here. Cached blocks bypass their
intermediate layers and replay the exact tensor at the block boundary. The
model still builds its own masks/positions and executes its norm, score head
and padding-aware pooling. No KV cache or cross-microbatch state is reused.
"""
from contextlib import contextmanager
from functools import wraps

import torch


@contextmanager
def _prefix_cache(model):
    """Temporarily wrap reusable blocks; retain one boundary per block."""
    layer_blocks = model.route_context.layer_to_block
    blocks = sorted(set(layer_blocks.values()))
    if blocks != list(range(4)):
        raise ValueError('Oracle cache requires four contiguous blocks')
    if [layer_blocks[i] for i in range(len(model.decoder_layers))] != sorted(layer_blocks.values()):
        raise ValueError('Oracle blocks must be contiguous in decoder order')
    ends = {b: max(i for i, block in layer_blocks.items() if block == b) for b in blocks}
    route = []
    boundaries = {}
    restored = []

    def cached_forward(original, block, last):
        @wraps(original)
        def forward(hidden_states, *args, **kwargs):
            prefix = tuple(route[:block+1])
            cached = boundaries.get(block)
            if cached is not None and cached[0] == prefix:
                # The enclosing model observes this block only at its exit.
                return cached[1] if last else hidden_states
            result = original(hidden_states, *args, **kwargs)
            if not isinstance(result, torch.Tensor):
                raise TypeError('Oracle cache requires tensor-returning decoder layers')
            if last:
                boundaries[block] = (prefix, result)
            return result
        return forward

    try:
        for i, layer in enumerate(model.decoder_layers):
            block = layer_blocks[i]
            if block == blocks[-1]:
                continue  # All 27 final-block executions are distinct.
            restored.append((layer, layer.__dict__.get('forward'), 'forward' in layer.__dict__))
            layer.forward = cached_forward(layer.forward, block, i == ends[block])
        yield route
    finally:
        for layer, previous, had_override in reversed(restored):
            if had_override:
                layer.forward = previous
            else:
                del layer.forward
        boundaries.clear()


@torch.no_grad()
def oracle_logits(experts, encoded, routes):
    """Return [candidate, case, label] CPU FP32 logits in original row order.

    Keep microbatch membership and padding identical to FrozenExperts.logits.
    Only loop order changes: one input preparation per microbatch, then all
    routes, instead of input preparation separately for every route.
    """
    if any(module.training for module in experts.model.modules()):
        raise ValueError('Oracle prefix cache requires a frozen eval model')
    if any(parameter.requires_grad for parameter in experts.model.parameters()):
        raise ValueError('Oracle prefix cache requires frozen parameters')
    routes = [tuple(int(x) for x in route) for route in routes]
    if not encoded or not routes:
        raise ValueError('Oracle requires nonempty inputs and routes')
    rows = [[] for _ in routes]
    try:
        for start in range(0, len(encoded), experts.micro):
            inputs = experts._inputs(encoded[start:start+experts.micro])
            with _prefix_cache(experts.model) as active_route:
                for candidate, route in enumerate(routes):
                    experts._set_route(route, len(inputs['input_ids']))
                    active_route[:] = route
                    with torch.autocast(experts.device.type, dtype=torch.bfloat16,
                                        enabled=experts.device.type == 'cuda'):
                        logits = experts.model(**inputs, use_cache=False,
                            output_hidden_states=False, output_attentions=False).logits
                        rows[candidate].append(logits.float().cpu())
    finally:
        experts.model.clear_routes()
    return torch.stack([torch.cat(parts) for parts in rows])
