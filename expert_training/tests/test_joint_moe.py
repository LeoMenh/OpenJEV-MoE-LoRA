import inspect
import importlib.util
from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForSequenceClassification
from adaptive_blockwise.modeling import replace_with_blockwise_adapters
from adaptive_blockwise.joint_moe import JointMoE


def tiny_joint():
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8, num_labels=12,
                         pad_token_id=0, attention_dropout=0., use_cache=False)
    base = Qwen3ForSequenceClassification(config)
    for name, p in base.named_parameters():
        p.requires_grad = name.startswith("score.")
    blocks = [{"block_id": b, "start": b, "layers": [b]} for b in range(4)]
    groups = {"0": 1, "1": 3, "2": 3, "3": 3}
    core, _ = replace_with_blockwise_adapters(base, blocks, ["q_proj", "k_proj", "v_proj", "o_proj"], groups, 2, 2, 4, 0.)
    for name, p in core.named_parameters():
        if "private_B" in name:
            torch.nn.init.normal_(p, std=0.05)
    return JointMoE(core, blocks, groups, 8)


def test_classifier_gradient_reaches_router_and_experts():
    torch.manual_seed(43)
    model = tiny_joint()
    ids = torch.tensor([[2, 3, 0], [5, 6, 7]])
    logits, balance = model(ids, ids.ne(0).long())
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert "labels" not in inspect.signature(model.forward).parameters
    assert not any("group" in p for p in inspect.signature(model.forward).parameters)
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert any(p.grad.abs().sum() > 0 for p in model.routers.parameters())
    assert any(p.grad.abs().sum() > 0 for n, p in model.named_parameters() if "private_B" in n)
    assert model.context.probabilities[0].tolist() == [[1.], [1.]]


def test_per_sample_routing_and_state_round_trip():
    torch.manual_seed(44)
    model = tiny_joint().eval()
    for router in model.routers.values():
        torch.nn.init.normal_(router.net[-1].weight, std=.1)
    ids = torch.tensor([[2, 3, 0], [5, 6, 7]])
    together = model(ids, ids.ne(0).long())[0]
    separate = torch.cat([model(row[None], row[None].ne(0).long())[0] for row in ids])
    torch.testing.assert_close(together, separate, atol=1e-6, rtol=1e-5)
    clone = tiny_joint().eval()
    clone.load_state_dict(model.state_dict())
    clone.load_trainable_state(model.trainable_state())
    torch.testing.assert_close(together, clone(ids, ids.ne(0).long())[0])


def test_nonreentrant_checkpointing_preserves_router_and_expert_gradients():
    torch.manual_seed(45)
    plain = tiny_joint().train()
    checkpointed = tiny_joint().train()
    checkpointed.load_state_dict(plain.state_dict())
    checkpointed.core.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    ids = torch.tensor([[2, 3, 0], [5, 6, 7]])
    for model in (plain, checkpointed):
        logits, balance = model(ids, ids.ne(0).long())
        (torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])) + 0.001 * balance).backward()
    reference = dict(plain.named_parameters())
    for name, parameter in checkpointed.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None
            torch.testing.assert_close(parameter.grad, reference[name].grad, atol=1e-6, rtol=1e-5)


def test_resumed_global_batch_sampler_exact_cover():
    spec = importlib.util.spec_from_file_location("joint_train", Path(__file__).resolve().parents[1] / "scripts/20_train_joint_moe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for size in (1, 15, 16, 17, 33):
        shards = [list(module.GlobalBatchSampler(size, 2, rank, 8, 1, 0, True)) for rank in range(8)]
        real = [i for batches in shards for batch in batches for i in batch if i >= 0]
        assert sorted(real) == list(range(size))
        for rank in range(8):
            remaining = list(module.GlobalBatchSampler(size, 2, rank, 8, 1, 0, True, start=1))
            assert remaining == shards[rank][1:]
