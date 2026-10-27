"""The 1223 diagnostic must preserve each singleton's twelve Top-1 predictions."""
import pytest
import torch
import numpy as np
from itertools import product
from test_oracle_cache import tiny_experts, examples
from openjev_drg.oracle import oracle_logits


@pytest.mark.parametrize('padding_side', ['left', 'right'])
def test_1223_singleton_prefix_reuse_matches_twelve_full_forwards(padding_side):
    groups = {'0': 1, '1': 2, '2': 2, '3': 3}
    experts = tiny_experts(padding_side, groups=groups)
    experts.micro = 1
    routes = np.asarray(list(product(range(1), range(2), range(2), range(3))))
    for case in examples()[:3]:
        baseline = torch.stack([experts.logits([case], route.tolist()) for route in routes])
        actual = oracle_logits(experts, [case], routes)
        assert torch.equal(actual, baseline)
        assert actual.shape == (12, 1, 3)
        assert torch.equal(actual.argmax(-1), baseline.argmax(-1))
        assert not experts.model.route_context.routes
    assert all('forward' not in layer.__dict__ for layer in experts.model.decoder_layers)
