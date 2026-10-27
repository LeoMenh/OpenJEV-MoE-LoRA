"""Inference-only shrinkage of case routes toward a train-fitted mean.

Hooks leave inherited modules and checkpoint files untouched. Later routers
see the hidden states produced by earlier altered routes, not cached routes
from the original model. Zero shrinkage bypasses all numerical conversion.
"""
import math
from contextlib import contextmanager

import torch


def validate_means(routers, means, coefficient):
    if type(coefficient) not in (int, float) or not math.isfinite(coefficient) or not 0 <= coefficient <= 1:
        raise ValueError('A finite shrinkage coefficient in [0,1] is required.')
    if set(means) != set(routers):
        raise ValueError('Mean routes must cover exactly the learned routers.')
    result = {}
    for key, router in routers.items():
        values = torch.as_tensor(means[key], dtype=torch.float64)
        groups = router.net[-1].out_features
        if values.shape != (groups,) or not torch.isfinite(values).all() or not (values > 0).all():
            raise ValueError('Mean routes must be finite, positive vectors of the correct size.')
        if not torch.isclose(values.sum(), torch.tensor(1., dtype=values.dtype), atol=1e-7, rtol=0):
            raise ValueError('Mean routes must sum to one.')
        result[key] = values.float()
    return result


@contextmanager
def shrunk_routes(model, means, coefficient, observations=None):
    """Install temporary route hooks; remove them after normal/error exit."""
    values = validate_means(model.routers, means, coefficient)
    if model.training:
        raise ValueError('This route intervention is inference-only.')
    if getattr(model.context, '_route_shrinkage_installed', False):
        raise ValueError('Nested route interventions are unsupported.')
    if coefficient == 0:
        yield
        return
    handles = []
    model.context._route_shrinkage_installed = True
    try:
        for key, router in model.routers.items():
            def hook(_module, _arguments, logits, block=key):
                if model.context.mode != 'soft':
                    raise ValueError('Shrinkage requires inherited soft routing.')
                if logits.ndim != 2 or logits.shape[1] != len(values[block]) or not torch.isfinite(logits).all():
                    raise ValueError('Malformed router logits.')
                probability = logits.float().softmax(-1)
                mean = values[block].to(logits.device)
                mixed = (1 - coefficient) * probability + coefficient * mean
                if observations is not None:
                    observations[block] = {'original_at_current_hidden': probability.detach().cpu().clone(),
                                           'desired_route': mixed.detach().cpu().clone()}
                # The unchanged block hook softmaxes these returned logits.
                return mixed.log()
            handles.append(router.register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()
        model.context._route_shrinkage_installed = False
