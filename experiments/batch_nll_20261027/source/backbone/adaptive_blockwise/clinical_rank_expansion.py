"""Expand a trained LoRA subspace while preserving its initial update and RNG."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class SplitLoRAA(nn.Linear):
    def forward(self, inputs):
        assert self.out_features == 64 and self.bias is None
        old = F.linear(inputs, self.weight[:16].contiguous())
        extra = F.linear(inputs, self.weight[16:].contiguous())
        return torch.cat((old, extra), dim=-1)


class SplitLoRAB(nn.Linear):
    def forward(self, inputs):
        assert self.in_features == 64 and self.bias is None
        old = F.linear(inputs[..., :16].contiguous(), self.weight[:, :16].contiguous())
        extra = F.linear(inputs[..., 16:].contiguous(), self.weight[:, 16:].contiguous())
        return old + extra


def install_split_rank_kernels(model):
    count = 0
    for layer in model.modules():
        if not hasattr(layer, 'lora_A') or 'default' not in layer.lora_A:
            continue
        a, b = layer.lora_A['default'], layer.lora_B['default']
        assert a.out_features == b.in_features == layer.r['default'] == 64
        assert a.bias is b.bias is None
        # Preserve parameter names, storage, dtype, and checkpoint compatibility.
        a.__class__, b.__class__ = SplitLoRAA, SplitLoRAB
        count += 1
    assert count
    return count


@torch.no_grad()
def expand_clinical_lora(model, rank=64, seed=20261009):
    if rank != 64:
        raise ValueError('This matched experiment expands rank16 to64')
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all()
    expanded = []
    try:
        for name, layer in model.named_modules():
            if not hasattr(layer, 'lora_A') or 'default' not in layer.lora_A:
                continue
            a, b = layer.lora_A['default'], layer.lora_B['default']
            assert a.out_features == b.in_features == layer.r['default'] == 16
            assert layer.scaling['default'] == 2.
            new_a = nn.Linear(a.in_features, rank, bias=False, device=a.weight.device, dtype=a.weight.dtype)
            new_b = nn.Linear(rank, b.out_features, bias=False, device=b.weight.device, dtype=b.weight.dtype)
            # A separate generator keeps the control's global dropout RNG intact.
            generator = torch.Generator(device=a.weight.device).manual_seed(seed + len(expanded))
            nn.init.kaiming_uniform_(new_a.weight, a=math.sqrt(5), generator=generator)
            new_b.weight.zero_()
            new_a.weight[:16].copy_(a.weight)
            new_b.weight[:, :16].copy_(b.weight)
            assert torch.equal(new_a.weight[:16], a.weight)
            assert torch.equal(new_b.weight[:, :16], b.weight)
            assert torch.count_nonzero(new_b.weight[:, 16:]) == 0
            assert torch.count_nonzero(new_a.weight[16:]) > 0
            layer.lora_A['default'], layer.lora_B['default'] = new_a, new_b
            layer.r['default'], layer.lora_alpha['default'] = rank, 2 * rank
            layer.scaling['default'] = 2.
            expanded.append({'module': name, 'old_rank': 16, 'new_rank': rank,
                             'old_coordinates_identical': True, 'new_B_zero': True,
                             'scaling_unchanged': 2.})
        assert expanded
        config = model.peft_config['default']
        assert config.r == 16 and config.lora_alpha == 32 and not config.use_rslora
        config.r, config.lora_alpha = rank, 2 * rank
        assert install_split_rank_kernels(model) == len(expanded)
    finally:
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), cuda_rng))
    return {'rank': rank, 'alpha': 2 * rank, 'scaling': 2., 'expanded_modules': expanded,
            'new_A_initialization': 'Kaiming uniform with separate generator seed20261009+moduleindex',
            'original_coordinates_preserved': True, 'new_B_initially_zero': True,
            'global_cpu_cuda_rng_preserved': True,
            'execution': 'Original16 and additional48 computed separately; avoids changing old BF16 GEMM shapes',
            'mathematical_identity': '2*[B_old,0]@[A_old;A_new] == 2*B_old@A_old'}
