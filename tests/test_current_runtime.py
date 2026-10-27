"""Current 18-route soft execution must reproduce the frozen expert pool."""
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from transformers import Qwen3Config, Qwen3ForSequenceClassification

from adaptive_blockwise.modeling import SharedPrivateLoRALinear, replace_with_blockwise_adapters
from adaptive_blockwise.joint_moe import SampleRouter
from soft_openjev_runtime import ExpertPool, SampleBridge, soft_linear, routes_for


def test_soft_mixture_matches_hard_routes_and_routes_gradients():
    torch.manual_seed(19)
    layer = SharedPrivateLoRALinear(nn.Linear(7, 5), 10, 1, 2, 2, 2, 4, 0).eval()
    for name, value in layer.named_parameters():
        if 'shared_B' in name or 'private_B' in name:
            nn.init.normal_(value, std=0.1)
    layer.requires_grad_(False)
    context = SimpleNamespace(routes={}, probabilities={})
    object.__setattr__(layer, 'route_context', context)
    x = torch.randn(3, 4, 7)
    hard = []
    for k in range(2):
        context.routes[1] = torch.full((3,), k)
        hard.append(layer(x))
    gates = torch.tensor([[0.2, 0.8], [0.9, 0.1], [0.4, 0.6]], requires_grad=True)
    context.probabilities[1] = gates
    actual = soft_linear(layer, x)
    expected = sum(gates[:, k, None, None] * hard[k] for k in range(2))
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    actual.square().mean().backward()
    assert gates.grad is not None and gates.grad.abs().sum() > 0
    assert all(value.grad is None for value in layer.parameters())
    context.routes[1] = torch.tensor([0, 1, 0])
    context.probabilities[1] = torch.nn.functional.one_hot(context.routes[1], 2).float()
    torch.testing.assert_close(soft_linear(layer, x), layer(x), atol=1e-6, rtol=1e-5)


def test_current_pool_hard_equivalence_and_case_independence():
    torch.manual_seed(21)
    groups = {'0': 1, '1': 2, '2': 3, '3': 3}
    routes = routes_for(groups)
    assert routes.shape == (18, 4) and len(np.unique(routes, axis=0)) == 18
    blocks = [{'block_id': b, 'start': b, 'end': b, 'layers': [b]} for b in range(4)]
    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=2,
                        head_dim=8, num_labels=3, pad_token_id=0, attention_dropout=0.)
    core, _ = replace_with_blockwise_adapters(Qwen3ForSequenceClassification(config),
        blocks, ['q_proj', 'k_proj', 'v_proj', 'o_proj'], groups, 2, 2, 4, 0.)
    for name, weight in core.named_parameters():
        if 'private_B' in name or 'shared_B' in name:
            nn.init.normal_(weight, std=0.1)
    core.requires_grad_(False).eval()
    inputs = {'input_ids': torch.tensor([[1, 2, 3], [4, 5, 0]]),
              'attention_mask': torch.tensor([[1, 1, 1], [1, 1, 0]])}
    forced = torch.tensor([[0, 0, 1, 2], [0, 1, 2, 0]])
    core.set_teacher_routes({b: forced[:, b] for b in range(4)})
    with torch.no_grad():
        reference = core(**inputs).logits
    c = {'source': {'num_groups': groups}, 'expert_meta': {'blocks': blocks}}
    pool = ExpertPool(c, torch.device('cpu'), core=core, tokenizer=None)
    with torch.no_grad():
        torch.testing.assert_close(pool(inputs, forced=forced), reference, atol=1e-6, rtol=1e-5)
    routers = nn.ModuleDict({str(b): SampleRouter(16, groups[str(b)], 8) for b in (1, 2, 3)})
    for router in routers.values():
        nn.init.normal_(router.net[-1].weight, std=0.2)
    with torch.no_grad():
        before = pool(inputs, mlp=routers, capture=True).clone()
        gates_before = {b: x.clone() for b, x in pool.probabilities.items()}
        changed = {k: v.clone() for k, v in inputs.items()}
        changed['input_ids'][1] = torch.tensor([9, 10, 0])
        after = pool(changed, mlp=routers, capture=True)
    torch.testing.assert_close(before[0], after[0], atol=1e-6, rtol=1e-5)
    for b in (1, 2, 3):
        torch.testing.assert_close(gates_before[b][0], pool.probabilities[b][0], atol=1e-6, rtol=1e-5)
    assert all(not parameter.requires_grad for parameter in pool.parameters())


def test_openjev_bridge_keeps_case_inputs_independent():
    torch.manual_seed(22)
    bridge = SampleBridge(16, 8, 3, width=4, prefix=4)
    z = torch.randn(2, 16)
    original = bridge.embeddings(z)
    changed = z.clone(); changed[1] = torch.randn(16)
    assert original.shape == (2, 5, 8)
    torch.testing.assert_close(original[0], bridge.embeddings(changed)[0])
    assert not torch.equal(original[0, :4], original[1, :4])
    torch.testing.assert_close(original[0, -1], original[1, -1])
