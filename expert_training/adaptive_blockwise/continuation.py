"""Shared data contracts and model builders for the KDA/RAG continuation."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from .io_utils import dataframe_identity, load_split_csv, read_json, sha256_file, write_json
from .signals import make_dataset
from .token_kda import FrozenQwenTokens, MultiDepthKDA

ROOT = Path(__file__).resolve().parents[1]


def joint_utilities():
    spec = importlib.util.spec_from_file_location('_continuation_joint_utilities', ROOT / 'scripts/20_train_joint_moe.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_frames(config, splits=('train', 'validation')):
    manifest = read_json(Path(config['source_run']) / '00_splits/manifest.json')
    result = {}
    for split in splits:
        frame = load_split_csv(manifest['paths'][split])
        if dataframe_identity(frame) != manifest['dataframe_sha256'][split]:
            raise ValueError(f'Immutable split changed: {split}')
        result[split] = frame
    return result


def development_indices(config, size):
    path = Path(config['run_dir']) / 'development_partition.npz'
    if not path.exists():
        raise FileNotFoundError('Run continuation preparation before training')
    values = np.load(path)
    if not np.array_equal(np.sort(np.r_[values['tune'], values['audit']]), np.arange(size)):
        raise ValueError('Development partition does not cover each case exactly once')
    return values['tune'], values['audit']


def prepare_tokens(config, phase, tokenizer, rank, splits=('train', 'validation')):
    if phase == 'kda':
        return {split: load_from_disk(Path(config['source_run']) / '06_joint/tokenized' / split) for split in splits}
    token_config = config[phase]
    root = Path(config['run_dir']) / 'tokens' / phase
    manifest = read_json(Path(config['source_run']) / '00_splits/manifest.json')
    for split in splits:
        target = root / split
        contract = {'identity': manifest['dataframe_sha256'][split], 'max_length': token_config['max_length'],
                    'model_path': token_config['model_path'], 'phase': phase}
        if rank == 0:
            if (target / 'contract.json').exists():
                if read_json(target / 'contract.json') != contract:
                    raise ValueError('Continuation token cache contract changed')
            else:
                frame = load_split_csv(manifest['paths'][split])
                if dataframe_identity(frame) != contract['identity']:
                    raise ValueError('Split identity mismatch')
                dataset = make_dataset(frame, tokenizer, token_config['max_length'])
                dataset.save_to_disk(target)
                write_json(target / 'contract.json', contract)
        if dist.is_initialized():
            dist.barrier()
    return {split: load_from_disk(root / split) for split in splits}


class TrainableModule(nn.Module):
    def trainable_state(self):
        return {name: p.detach().cpu().clone() for name, p in self.named_parameters() if p.requires_grad}

    def load_trainable_state(self, state):
        expected = {name for name, p in self.named_parameters() if p.requires_grad}
        if set(state) != expected:
            raise ValueError('Trainable checkpoint keys differ')
        parameters = dict(self.named_parameters())
        with torch.no_grad():
            for name, value in state.items():
                parameters[name].copy_(value.to(parameters[name]))


class OnlineKDA(TrainableModule):
    def __init__(self, config, local_rank):
        super().__init__()
        source = read_json(Path(config['source_run']) / 'pipeline_state/config.json')
        joint = read_json(Path(config['source_run']) / 'pipeline_state/joint_config.json')
        # Extractor is intentionally a plain Python object: DDP owns ONLY the
        # readout, and the frozen 4-bit backbone never receives a gradient.
        self.extractor = FrozenQwenTokens(source, joint, config['backbone_checkpoint'], local_rank)
        k = config['kda']
        self.readout = MultiDepthKDA(5120, k['width'], k['key_size'], k['value_size'], config['num_labels']).cuda(local_rank)
        self.tokenizer = self.extractor.tokenizer

    def forward(self, input_ids, attention_mask):
        _, hidden = self.extractor(input_ids, attention_mask)
        return self.readout(hidden, attention_mask)

    def with_backbone(self, input_ids, attention_mask):
        base, hidden = self.extractor(input_ids, attention_mask)
        logits, features = self.readout(hidden, attention_mask)
        return logits, features, base


class DenseEncoder(TrainableModule):
    def __init__(self, config, local_rank):
        super().__init__()
        c = config['embedding']
        self.tokenizer = AutoTokenizer.from_pretrained(c['model_path'], padding_side='right', local_files_only=True)
        base = AutoModel.from_pretrained(c['model_path'], dtype=torch.bfloat16, local_files_only=True)
        self.encoder = get_peft_model(base, LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=16, lora_alpha=32,
                                                     lora_dropout=.05, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']))
        self.classifier = nn.Linear(base.config.hidden_size, config['num_labels'])
        self.cuda(local_rank)
        self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.encoder.config.use_cache = False

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        positions = torch.arange(hidden.shape[1], device=hidden.device)[None].expand_as(attention_mask)
        last = positions.masked_fill(~attention_mask.bool(), -1).max(1).values
        features = hidden[torch.arange(len(hidden), device=hidden.device), last].float()
        return self.classifier(features).float(), features


class ClinicalClassifier(TrainableModule):
    def __init__(self, config, local_rank):
        super().__init__()
        c = config['clinical']
        self.tokenizer = AutoTokenizer.from_pretrained(c['model_path'], local_files_only=True)
        base = AutoModelForSequenceClassification.from_pretrained(c['model_path'], num_labels=config['num_labels'],
                               dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True)
        self.model = get_peft_model(base, LoraConfig(task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32,
                              lora_dropout=.05, target_modules=['Wqkv', 'Wo', 'Wi'], modules_to_save=['head', 'classifier']))
        self.cuda(local_rank)
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    def forward(self, input_ids, attention_mask):
        scores = self.model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
        return scores, scores


class Reranker(TrainableModule):
    def __init__(self, config, local_rank):
        super().__init__()
        c = config['reranker']
        self.tokenizer = AutoTokenizer.from_pretrained(c['model_path'], padding_side='left', local_files_only=True)
        base = AutoModelForCausalLM.from_pretrained(c['model_path'], dtype=torch.bfloat16, local_files_only=True)
        self.model = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
                                     lora_dropout=.05, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']))
        self.no_id = self.tokenizer.convert_tokens_to_ids('no')
        self.yes_id = self.tokenizer.convert_tokens_to_ids('yes')
        if min(self.no_id, self.yes_id) < 0 or self.no_id == self.yes_id:
            raise ValueError('Reranker yes/no tokens are invalid')
        self.cuda(local_rank)
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.model.config.use_cache = False

    def forward(self, input_ids, attention_mask):
        scores = self.model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=1).logits[:, -1]
        scores = scores[:, [self.no_id, self.yes_id]].float()
        return scores, scores


def build_phase(config, phase, local_rank, checkpoint=None):
    constructors = {'kda': OnlineKDA, 'embedding': DenseEncoder, 'clinical': ClinicalClassifier, 'reranker': Reranker}
    model = constructors[phase](config, local_rank)
    # Optimizer states and trainable adapters are FP32, even when the frozen
    # encoders use BF16. Autocast controls their actual matrix products.
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    if checkpoint is not None:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_trainable_state(saved['model'])
    return model


def model_sources():
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in [
        Path(__file__), ROOT / 'adaptive_blockwise/token_kda.py',
        ROOT / 'scripts/623_train_kda_rag.py', ROOT / 'adaptive_blockwise/retrieval_pipeline.py',
    ]}
