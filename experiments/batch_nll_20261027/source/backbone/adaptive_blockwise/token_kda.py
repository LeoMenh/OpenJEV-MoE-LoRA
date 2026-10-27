"""Three-depth Kimi Delta Attention readout of a frozen Qwen MoE-LoRA.

The production kernel is FLA KDA 0.4.2. The reference recurrence decays the
state BEFORE the delta residual, matching that kernel and Kimi Linear.
"""
import importlib.util
import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]


def reference_kda(q, k, v, log_gate, beta):
    """Float32 recurrent oracle; batch, time, single-head feature dimensions."""
    state = torch.zeros(q.shape[0], q.shape[-1], v.shape[-1], device=q.device, dtype=torch.float32)
    outputs = []
    for t in range(q.shape[1]):
        state = log_gate[:, t].float().exp().unsqueeze(-1) * state
        residual = v[:, t].float() - torch.einsum('bk,bkv->bv', k[:, t].float(), state)
        state = state + beta[:, t].float()[:, None, None] * k[:, t].float()[:, :, None] * residual[:, None, :]
        outputs.append(torch.einsum('bk,bkv->bv', q[:, t].float(), state))
    return torch.stack(outputs, dim=1)


class KDAReadout(nn.Module):
    def __init__(self, hidden_size=5120, width=512, key_size=64, value_size=128, backend='fla'):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(hidden_size, width), nn.LayerNorm(width))
        self.query = nn.Linear(width, key_size, bias=False)
        self.key = nn.Linear(width, key_size, bias=False)
        self.value = nn.Linear(width, value_size, bias=False)
        self.gate = nn.Linear(width, key_size)
        self.beta = nn.Linear(width, 1)
        self.output_norm = nn.LayerNorm(value_size)
        self.pool_query = nn.Parameter(torch.randn(value_size) / math.sqrt(value_size))
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, 4.)
        self.backend = backend

    def forward(self, hidden, mask):
        valid = mask.bool()
        if not valid.any(1).all():
            raise ValueError('KDA requires at least one real token per case')
        # Erase padding BEFORE projection so arbitrary pad states have no effect.
        u = self.project(hidden.float().masked_fill(~valid[..., None], 0.))
        q = self.query(u) * valid[..., None]
        k = F.normalize(self.key(u).float(), dim=-1).to(q.dtype)
        v = self.value(u)
        log_gate = F.logsigmoid(self.gate(u).float()).masked_fill(~valid[..., None], 0.)
        beta = self.beta(u).float().sigmoid().squeeze(-1) * valid
        if self.backend == 'reference':
            outputs = reference_kda(q, k, v, log_gate, beta)
        elif self.backend == 'fla':
            sys.path.insert(0, str(ROOT / 'vendor/kda')) if str(ROOT / 'vendor/kda') not in sys.path else None
            from fla.ops.kda import chunk_kda
            dtype = torch.bfloat16
            outputs, _ = chunk_kda(q.to(dtype).unsqueeze(2), k.to(dtype).unsqueeze(2),
                                  v.to(dtype).unsqueeze(2), log_gate.unsqueeze(2), beta.unsqueeze(2),
                                  scale=1., use_qk_l2norm_in_kernel=False)
            outputs = outputs.squeeze(2).float()
        else:
            raise ValueError(self.backend)
        normalized = self.output_norm(outputs.float())
        scores = torch.einsum('btv,v->bt', normalized, self.pool_query) / math.sqrt(normalized.shape[-1])
        weights = scores.masked_fill(~valid, -torch.inf).softmax(1)
        return torch.einsum('bt,btv->bv', weights, normalized)


class MultiDepthKDA(nn.Module):
    def __init__(self, hidden_size=5120, width=512, key_size=64, value_size=128, num_labels=738, backend='fla'):
        super().__init__()
        self.readers = nn.ModuleList([KDAReadout(hidden_size, width, key_size, value_size, backend) for _ in range(3)])
        self.norm = nn.LayerNorm(value_size * 3)
        self.classifier = nn.Linear(value_size * 3, num_labels)

    def forward(self, hidden_states, attention_mask):
        if len(hidden_states) != 3:
            raise ValueError('Expected token states at depths 20, 30, 40')
        features = torch.cat([reader(hidden, attention_mask) for reader, hidden in zip(self.readers, hidden_states)], dim=-1)
        return self.classifier(self.norm(features)).float(), features.float()


class FrozenQwenTokens:
    """Text-only forward. Never accepts labels or oracle group assignments."""
    def __init__(self, source, joint_config, checkpoint, local_rank):
        spec = importlib.util.spec_from_file_location('_joint_source_trainer', ROOT / 'scripts/20_train_joint_moe.py')
        trainer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trainer)
        build_config = dict(joint_config, gradient_checkpointing=False)
        self.model, self.tokenizer = trainer.build_model(source, build_config, local_rank)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        self.model.load_trainable_state(saved['model'])
        self.model.requires_grad_(False).eval()
        self.best_validation = saved['validation']
        self.values = {}
        layers = self.model.core.decoder_layers
        modules = [layers[19], layers[29], self.model.core.model.model.norm]
        self.handles = [module.register_forward_hook(self._capture(index)) for index, module in enumerate(modules)]

    def _capture(self, index):
        def hook(module, args, output):
            self.values[index] = (output if torch.is_tensor(output) else output[0]).detach()
        return hook

    @torch.no_grad()
    def __call__(self, input_ids, attention_mask):
        self.values.clear()
        self.model.eval()
        logits, _ = self.model(input_ids=input_ids, attention_mask=attention_mask, mode='soft')
        if set(self.values) != {0, 1, 2}:
            raise RuntimeError('Frozen backbone did not produce all three requested depths')
        hidden = [self.values.pop(i) for i in range(3)]
        return logits.detach(), hidden


def supervised_contrastive(features, labels, sample_weight=None, temperature=.1):
    """Global-batch anchors with no loss for anchors lacking positive partners."""
    import torch.distributed as dist
    if sample_weight is None:
        sample_weight = torch.ones(len(labels), device=labels.device)
    if dist.is_initialized():
        from torch.distributed.nn.functional import all_gather
        features = torch.cat(all_gather(features), dim=0)
        gathered_labels = [torch.empty_like(labels) for _ in range(dist.get_world_size())]
        gathered_weights = [torch.empty_like(sample_weight) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_labels, labels)
        dist.all_gather(gathered_weights, sample_weight)
        labels, sample_weight = torch.cat(gathered_labels), torch.cat(gathered_weights)
    features = F.normalize(features.float(), dim=-1)
    real = sample_weight > 0
    allowed = real[:, None] & real[None, :] & ~torch.eye(len(labels), device=labels.device, dtype=torch.bool)
    positives = allowed & (labels[:, None] == labels[None, :])
    scores = features @ features.T / temperature
    scores = scores - scores.max(1, keepdim=True).values.detach()
    log_den = torch.logsumexp(scores.masked_fill(~allowed, -1e4), dim=1)
    row_loss = ((log_den[:, None] - scores) * positives).sum(1) / positives.sum(1).clamp_min(1)
    valid = positives.any(1)
    return (row_loss * valid).sum() / valid.sum().clamp_min(1)
