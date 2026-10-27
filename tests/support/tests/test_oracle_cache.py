"""The cache must preserve exhaustive route scores, including padding and tail cases."""
import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import DataCollatorWithPadding, PreTrainedTokenizerFast
from transformers import Qwen3Config, Qwen3ForSequenceClassification

from adaptive_blockwise.modeling import replace_with_blockwise_adapters
from openjev_drg.experts import FrozenExperts
from openjev_drg.protocol import ROUTES


def tiny_experts(padding_side='right', groups=None):
    torch.manual_seed(71)
    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=24,
        num_hidden_layers=8, num_attention_heads=2, num_key_value_heads=2,
        head_dim=8, num_labels=3, pad_token_id=0, attention_dropout=0.0)
    blocks = [{'block_id': b, 'layers': [2*b, 2*b+1]} for b in range(4)]
    groups = groups or {'0': 1, '1': 3, '2': 3, '3': 3}
    model, _ = replace_with_blockwise_adapters(Qwen3ForSequenceClassification(config),
        blocks, ['q_proj', 'k_proj', 'v_proj', 'o_proj'], groups, 2, 2, 4, .05)
    for name, weight in model.named_parameters():
        if 'private_B' in name or 'shared_B' in name:
            torch.nn.init.normal_(weight, std=.15)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
        {str(i): i for i in range(32)}, unk_token='1')), pad_token='0')
    tokenizer.padding_side = padding_side
    experts = object.__new__(FrozenExperts)
    experts.model = model.requires_grad_(False).eval()
    experts.config = {'num_groups': groups}
    experts.device = torch.device('cpu')
    experts.micro = 2
    experts.collator = DataCollatorWithPadding(tokenizer)
    return experts


def examples():
    return [{'input_ids': ids, 'attention_mask': [1]*len(ids)}
        for ids in ([1, 2, 3, 4], [5, 6], [7], [8, 9, 10], [11, 12])]


@pytest.mark.parametrize('padding_side', ['left', 'right'])
def test_all_route_logits_losses_and_tail_are_bitwise_equal(padding_side):
    from openjev_drg.oracle import oracle_logits
    experts = tiny_experts(padding_side)
    encoded = examples()
    labels = torch.tensor([0, 2, 1, 1, 0])
    baseline = torch.stack([experts.logits(encoded, r.tolist()) for r in ROUTES])
    actual = oracle_logits(experts, encoded, ROUTES)
    assert torch.equal(actual, baseline)
    assert not torch.equal(baseline[0], baseline[-1]), 'Fixture must exercise private experts'
    expected_costs = [float(torch.nn.functional.cross_entropy(x, labels)) for x in baseline]
    costs = [float(torch.nn.functional.cross_entropy(x, labels)) for x in actual]
    assert costs == expected_costs
    assert np.argsort(costs, kind='stable').tolist() == np.argsort(expected_costs, kind='stable').tolist()
    # Neither later inputs nor a different traversal may reuse a stale boundary.
    other = encoded[::-1][:3]
    actual = oracle_logits(experts, other, ROUTES[::-1])
    assert torch.equal(actual, torch.stack([experts.logits(other, r.tolist()) for r in ROUTES[::-1]]))
    assert not experts.model.route_context.routes
    assert all('forward' not in layer.__dict__ for layer in experts.model.decoder_layers)


def test_shared_prefixes_skip_attention_computation():
    from openjev_drg.oracle import oracle_logits
    experts = tiny_experts()
    calls = [0]*4
    def count(block):
        def hook(*_args):
            calls[block] += 1
        return hook
    handles = [layer.self_attn.register_forward_hook(count(i//2))
        for i, layer in enumerate(experts.model.decoder_layers)]
    try:
        oracle_logits(experts, examples()[:2], ROUTES)
    finally:
        for handle in handles:
            handle.remove()
    assert calls == [2, 6, 18, 54]


def test_forwards_and_routes_restored_after_exception():
    from openjev_drg.oracle import oracle_logits
    experts = tiny_experts()
    encoded = examples()[:1]
    expected = experts.logits(encoded, ROUTES[0].tolist())
    def fail(*_args):
        raise RuntimeError('classifier failed')
    handle = experts.model.model.score.register_forward_hook(fail)
    try:
        with pytest.raises(RuntimeError, match='classifier failed'):
            oracle_logits(experts, encoded, ROUTES)
    finally:
        handle.remove()
    assert all('forward' not in layer.__dict__ for layer in experts.model.decoder_layers)
    assert not experts.model.route_context.routes
    assert torch.equal(expected, experts.logits(encoded, ROUTES[0].tolist()))


def test_cache_rejects_training_mode():
    from openjev_drg.oracle import oracle_logits
    experts = tiny_experts()
    experts.model.train()
    with pytest.raises(ValueError, match='eval'):
        oracle_logits(experts, examples(), ROUTES)
