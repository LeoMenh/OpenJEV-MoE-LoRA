"""LoRA biomedical document classifier with prefix or equal-mean chunk reading."""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

from .biolink_chunking import pack_biolink_chunks
from .io_utils import read_json, sha256_file


class BioLinkDocumentClassifier(nn.Module):
    def __init__(self, model_path, arm):
        super().__init__()
        assert arm in ('prefix512', 'chunks2048')
        raw = AutoModel.from_pretrained(model_path, local_files_only=True, dtype=torch.bfloat16,
                                       attn_implementation='sdpa', add_pooling_layer=False)
        assert raw.config.hidden_size == 1024 and raw.config.num_hidden_layers == 24
        assert raw.config.max_position_embeddings == 512
        targets = r'.*encoder\.layer\.\d+\.(?:attention\.self\.(?:query|key|value)|attention\.output\.dense|intermediate\.dense|output\.dense)'
        self.encoder = get_peft_model(raw, LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,
                    r=16, lora_alpha=32, lora_dropout=.05, target_modules=targets))
        self.normalizer = nn.LayerNorm(1024)
        self.classifier = nn.Linear(1024, 738)
        self.arm = arm
        self.chunk_limit = 1 if arm == 'prefix512' else 4
        trainable = {name: p for name, p in self.named_parameters() if p.requires_grad}
        assert all('lora_' in name or name.startswith(('normalizer.', 'classifier.')) for name in trainable)
        assert sum('lora_A' in name for name in trainable) == 24 * 6
        for p in trainable.values():
            p.data = p.data.float()

    def forward(self, input_ids, attention_mask, chunk_mask, chunk_limit=None):
        limit = self.chunk_limit if chunk_limit is None else chunk_limit
        assert limit in (1, 4) and input_ids.ndim == 3
        ids, attention, valid = input_ids[:, :limit], attention_mask[:, :limit], chunk_mask[:, :limit].bool()
        assert valid[:, 0].all()
        batch, chunks, length = ids.shape
        active = valid.reshape(-1).nonzero(as_tuple=True)[0]
        outputs = self.encoder(input_ids=ids.reshape(-1, length)[active],
                               attention_mask=attention.reshape(-1, length)[active])
        features = outputs.last_hidden_state[:, 0].float()
        # Unique destinations avoid a floating-point atomic sum across chunks.
        flat = features.new_zeros((batch * chunks, 1024)).index_copy(0, active, features)
        pooled = flat.reshape(batch, chunks, 1024).sum(1) / valid.sum(1, keepdim=True)
        return self.classifier(self.normalizer(pooled)).float()


class BioLinkDocumentExpert:
    def __init__(self, root, checkpoint):
        root, checkpoint = Path(root), Path(checkpoint)
        parent = root / 'runs/biolink_chunk_complement_development_v1'
        contract_path = parent / 'contract.json'
        contract = read_json(contract_path)
        digest = sha256_file(checkpoint)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        provenance = saved['provenance']
        assert not provenance['uses_test_data'] and provenance['fit_split'] == 'train'
        assert provenance['contract_sha256'] == sha256_file(contract_path)
        assert provenance['source_sha256'] == contract['trainer_source_sha256'] == sha256_file(root / 'scripts/465_train_biolink_document_lora.py')
        assert provenance['factory_source_sha256'] == contract['factory_source_sha256'] == sha256_file(Path(__file__))
        assert sha256_file(root / 'adaptive_blockwise/biolink_chunking.py') == contract['chunking_source_sha256']
        for path, expected in contract['model_hashes'].items():
            assert sha256_file(path) == expected, path
        self.model = BioLinkDocumentClassifier(contract['model_path'], provenance['arm']).cuda()
        parameters = {name: p for name, p in self.model.named_parameters() if p.requires_grad}
        assert set(parameters) == set(saved['state']) == set(provenance['trained_parameter_names'])
        with torch.no_grad():
            for name, parameter in parameters.items():
                value = saved['state'][name]
                assert value.shape == parameter.shape and value.dtype == parameter.dtype
                assert torch.isfinite(value).all(), name
                parameter.copy_(value.to(parameter.device))
        self.model.encoder.gradient_checkpointing_disable()
        self.model.requires_grad_(False).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(contract['model_path'], local_files_only=True, use_fast=True)
        self.arm, self.step, self.provenance = provenance['arm'], saved['step'], provenance
        self.artifact_hashes = {str(checkpoint): digest, str(contract_path): sha256_file(contract_path),
                               **contract['model_hashes'], str(Path(__file__)): sha256_file(Path(__file__))}
        assert sha256_file(checkpoint) == digest

    @torch.inference_mode()
    def scores(self, texts, batch_size=16):
        outputs = []
        for start in range(0, len(texts), batch_size):
            ids, masks, valid = pack_biolink_chunks(self.tokenizer, texts[start:start + batch_size])
            length = int(masks[:, :self.model.chunk_limit].sum(-1).max())
            inputs = {'input_ids': torch.tensor(ids[:, :, :length], device='cuda', dtype=torch.long),
                      'attention_mask': torch.tensor(masks[:, :, :length], device='cuda', dtype=torch.long),
                      'chunk_mask': torch.tensor(valid, device='cuda', dtype=torch.bool)}
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = self.model(**inputs)
            assert torch.isfinite(logits).all()
            outputs.append(logits.cpu().numpy())
        return np.concatenate(outputs)

