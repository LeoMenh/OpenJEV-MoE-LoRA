"""Per-case OpenJev gates over a frozen, selected Qwen MoE expert pool.

No old-router state is accepted by this module's training interfaces.
"""
import contextlib
import itertools
import math
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from peft import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, DataCollatorWithPadding

from adaptive_blockwise.io_utils import read_json, seed_everything
from adaptive_blockwise.modeling import replace_with_blockwise_adapters, adapter_state_dict
from adaptive_blockwise.signals import load_base_classifier, load_tokenizer


def routes_for(groups):
    return np.asarray(list(itertools.product(*(range(groups[str(b)]) for b in range(4)))), dtype=np.int64)


def conditional_targets(costs, routes, prefixes, block, temperature):
    """Each row is conditioned on its own discrete prefix, not its batch."""
    result = []
    for row, prefix in zip(costs, prefixes):
        available = np.all(routes[:, :block] == np.asarray(prefix), axis=1)
        values = [row[available & (routes[:, block] == a)].min()
                  for a in range(int(routes[:, block].max()) + 1)]
        result.append(values)
    values = torch.tensor(np.asarray(result), dtype=torch.float32)
    return (-values / temperature).softmax(-1)


def native_pool(c, device):
    seed_everything(c['seed'])
    source = c['source']; meta = c['expert_meta']
    tokenizer = load_tokenizer(source['model_name_or_path'])
    base = load_base_classifier(source['model_name_or_path'], meta['num_labels'], tokenizer,
        no_4bit=source['no_4bit'], device_map={'': device.index or 0}, compute_dtype=torch.bfloat16)
    if not source['no_4bit']:
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    base.config.use_cache = False
    core, _ = replace_with_blockwise_adapters(base, meta['blocks'], source['target_modules'],
        source['num_groups'], meta['r_shared'], meta['r_private'], meta['alpha'], meta['dropout'])
    state = torch.load(Path(c['run_dir']) / 'reused/expert_and_head.pt', map_location='cpu', weights_only=True)
    if set(state) != set(adapter_state_dict(core)):
        raise ValueError('Expert/head state does not exactly cover the frozen pool')
    core.load_state_dict(state, strict=False)
    core.requires_grad_(False).eval()
    return core, tokenizer


def soft_linear(self, hidden):
    result = self.base(hidden)
    p = self.route_context.probabilities[self.block_id]
    dropped = self.dropout(hidden).to(self.shared_A.weight.dtype)
    shared = self.shared_B(self.shared_A(dropped)) * self.scaling_shared
    private = torch.zeros_like(shared)
    for k in range(self.num_groups):
        update = self.private_B[k](self.private_A[k](dropped)) * self.scaling_private
        private = private + p[:, k, None, None] * update
    return result + (shared + private).to(result.dtype)


class ExpertPool(nn.Module):
    def __init__(self, c, device, core=None, tokenizer=None):
        super().__init__()
        from adaptive_blockwise.modeling import SharedPrivateLoRALinear
        if core is None:
            core, tokenizer = native_pool(c, device)
        self.core = core; self.tokenizer = tokenizer; self.config = c
        self.device = device; self.groups = c['source']['num_groups']
        self.collator = DataCollatorWithPadding(tokenizer)
        old = core.route_context
        self.layer_to_block = old.layer_to_block
        old.close()
        # oracle._prefix_cache expects these non-registered execution attributes.
        object.__setattr__(core, 'route_context', self)
        self.probabilities = {}; self.features = {}; self.mask = None
        self.bank = None; self.forced = None; self.capture = False; self.mlp = None
        self.handles = []
        for m in core.modules():
            if isinstance(m, SharedPrivateLoRALinear):
                object.__setattr__(m, 'route_context', self)
                m.forward = types.MethodType(soft_linear, m)
        for block in c['expert_meta']['blocks']:
            b = block['block_id']
            def hook(module, args, block=b):
                h = args[0]; mask = self.mask.float().unsqueeze(-1)
                z = (h.float() * mask).sum(1) / mask.sum(1).clamp_min(1)
                if self.capture:
                    self.features[block] = z.detach().cpu()
                if self.forced is not None:
                    p = F.one_hot(self.forced[:, block], self.groups[str(block)]).float()
                elif block == 0:
                    p = torch.ones((len(h), 1), device=h.device)
                elif self.mlp is not None:
                    p = self.mlp[str(block)](z).softmax(-1)
                else:
                    p = self.bank(block, z).softmax(-1)
                self.probabilities[block] = p
            self.handles.append(core.decoder_layers[block['start']].register_forward_pre_hook(hook))
        # Do not register self on wrapped modules (or a bank inside every layer).
        # route_context is a plain attribute, otherwise module traversal cycles.
        for m in core.modules():
            if isinstance(m, SharedPrivateLoRALinear) and 'route_context' in m._modules:
                del m._modules['route_context']
                object.__setattr__(m, 'route_context', self)
        if 'route_context' in core._modules:
            del core._modules['route_context']
            object.__setattr__(core, 'route_context', self)
        self.eval()

    @property
    def model(self):
        return self.core

    def inputs(self, frame):
        tokens = self.tokenizer(frame.text.astype(str).tolist(), truncation=True,
                                max_length=self.config['source']['max_length'])
        rows = [{k: v[i] for k, v in tokens.items()} for i in range(len(frame))]
        return {k: v.to(self.device) for k, v in self.collator(rows).items()}

    def forward(self, inputs, bank=None, forced=None, capture=False, mlp=None):
        # The old MLP is only supplied by the explicitly separate B0 evaluator.
        object.__setattr__(self, 'bank', bank)
        object.__setattr__(self, 'mlp', mlp)
        self.forced = forced; self.mask = inputs['attention_mask']; self.capture = capture
        self.features = {}; self.probabilities = {}
        return self.core(**inputs, use_cache=False).logits.float()

    def enable_checkpointing(self):
        # Hooks compute each gate once outside the checkpointed layer body.
        # Decoder remains eval: frozen expert dropout must stay disabled.
        for layer in self.core.decoder_layers:
            original = layer.forward
            def forward(*args, _original=original, **kwargs):
                if torch.is_grad_enabled():
                    return checkpoint(_original, *args, use_reentrant=False, **kwargs)
                return _original(*args, **kwargs)
            layer.forward = forward


class SampleBridge(nn.Module):
    def __init__(self, hidden, decoder_hidden, k, width=128, prefix=4):
        super().__init__(); self.prefix = prefix; self.d = decoder_hidden
        self.net = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, width), nn.SiLU(),
                                 nn.Linear(width, prefix * decoder_hidden))
        self.decision = nn.Parameter(torch.empty(1, 1, decoder_hidden))
        nn.init.normal_(self.decision, std=0.02)
        self.head = nn.Linear(decoder_hidden, k)

    def embeddings(self, z):
        p = self.net(z.float()).reshape(len(z), self.prefix, self.d)
        return torch.cat((p, self.decision.expand(len(z), -1, -1)), dim=1)


class BlockLoRA(nn.Module):
    """q/v adapter parameters are stable across all three forwards/backward.

    The decoder has no activation recomputation; autograd retains the correct
    adapter tensors. Switching block never switches requires_grad flags.
    """
    def __init__(self, base, owner, rank, alpha, dropout):
        super().__init__(); self.base = base
        object.__setattr__(self, 'owner', owner)
        device = base.weight.device
        self.A = nn.ModuleDict({str(b): nn.Linear(base.in_features, rank, bias=False,
                                                device=device, dtype=torch.float32) for b in (1, 2, 3)})
        self.B = nn.ModuleDict({str(b): nn.Linear(rank, base.out_features, bias=False,
                                                device=device, dtype=torch.float32) for b in (1, 2, 3)})
        for a in self.A.values(): nn.init.kaiming_uniform_(a.weight, a=math.sqrt(5))
        for b in self.B.values(): nn.init.zeros_(b.weight)
        self.dropout = nn.Dropout(dropout); self.scale = alpha / rank

    def forward(self, x):
        b = str(self.owner.active_block)
        # FP32 adapter arithmetic as in the existing OpenJev implementation.
        with torch.autocast('cuda', enabled=False):
            delta = self.B[b](self.A[b](self.dropout(x.float()))) * self.scale
        return self.base(x) + delta.to(x.dtype)


class SoftRouterBank(nn.Module):
    def __init__(self, c, hidden, device):
        super().__init__(); self.active_block = 1
        lm = AutoModelForCausalLM.from_pretrained(c['openjev_local_model'], dtype=torch.bfloat16,
                                                 attn_implementation='sdpa')
        lm.load_state_dict(torch.load(c['openjev_checkpoint'], map_location='cpu', weights_only=True), strict=True)
        self.decoder = lm.model.to(device); del lm
        self.decoder.requires_grad_(False)
        modules = dict(self.decoder.named_modules())
        for name, module in list(modules.items()):
            if name.rsplit('.', 1)[-1] in ('q_proj', 'v_proj'):
                parent, key = name.rsplit('.', 1)
                setattr(modules[parent], key, BlockLoRA(module, self, c['router_lora_rank'],
                        c['router_lora_alpha'], c['router_lora_dropout']))
        d = self.decoder.config.hidden_size
        self.features = nn.ModuleDict({str(b): SampleBridge(hidden, d, c['source']['num_groups'][str(b)],
            c['router_width'], c['router_prefix_length']) for b in (1, 2, 3)}).to(device)
        self.set_blocks((1, 2, 3))

    def set_blocks(self, blocks):
        self.requires_grad_(False)
        for b in blocks:
            self.features[str(b)].requires_grad_(True)
            for m in self.decoder.modules():
                if isinstance(m, BlockLoRA):
                    m.A[str(b)].requires_grad_(True); m.B[str(b)].requires_grad_(True)

    def forward(self, block, z):
        self.active_block = block
        f = self.features[str(block)]
        e = f.embeddings(z)
        h = self.decoder(inputs_embeds=e.to(torch.bfloat16),
            attention_mask=torch.ones(e.shape[:2], device=e.device, dtype=torch.long),
            use_cache=False).last_hidden_state[:, -1]
        return f.head(h.float())

    def adapted_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()
                if k.startswith('features.') or '.A.' in k or '.B.' in k}

    def load_adapted(self, state):
        if set(state) != set(self.adapted_state()):
            raise ValueError('OpenJev state coverage mismatch')
        self.load_state_dict(state, strict=False)

    def optimizer(self, c, task=False):
        scale = c['task_lr_scale'] if task else 1
        feature = [p for p in self.features.parameters() if p.requires_grad]
        adapter = [p for p in self.decoder.parameters() if p.requires_grad]
        return torch.optim.AdamW([{'params': feature, 'lr': c['router_lr'] * scale},
            {'params': adapter, 'lr': c['router_adapter_lr'] * scale}], weight_decay=0)


@torch.no_grad()
def collect_oracle(pool, frame, routes, temperature, prefix_cache):
    inputs = pool.inputs(frame); device = pool.device
    labels = torch.tensor(frame.label.to_numpy(), device=device)
    costs = []; states = {}; logits_by_route = []
    manager = prefix_cache(pool.core) if prefix_cache else contextlib.nullcontext([])
    with manager as active:
        for route in routes:
            active[:] = list(route)
            forced = torch.tensor(route, device=device).expand(len(frame), -1)
            logits = pool(inputs, forced=forced, capture=True)
            logits_by_route.append(logits.cpu())
            costs.append(F.cross_entropy(logits, labels, reduction='none').cpu())
            for b in (1, 2, 3):
                states[(b, tuple(route[:b]))] = pool.features[b]
    costs = torch.stack(costs, dim=1)
    best = costs.argmin(-1).numpy()
    features = {}; targets = {}
    for b in (1, 2, 3):
        prefixes = routes[best, :b]
        features[str(b)] = torch.stack([states[(b, tuple(prefix))][i] for i, prefix in enumerate(prefixes)])
        targets[str(b)] = conditional_targets(costs.numpy(), routes, prefixes, b, temperature)
    return {'costs': costs, 'features': features, 'targets': targets,
            'labels': labels.cpu(), 'best_routes': torch.tensor(routes[best])}, torch.stack(logits_by_route)
