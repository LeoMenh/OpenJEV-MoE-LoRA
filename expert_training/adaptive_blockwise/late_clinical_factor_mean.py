"""Single clinical expert with audited, same-trajectory factor/head averaging."""
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from .io_utils import read_json, sha256_file


class LateClinicalFactorMean:
    def __init__(self, root, checkpoint, device='cuda:0'):
        root, checkpoint = Path(root), Path(checkpoint)
        checkpoint_digest = sha256_file(checkpoint)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        p = saved['provenance']
        assert p['source_sha256'] == sha256_file(root/'scripts/316_build_late_clinical_factor_mean.py')
        assert p['fit_split'] == 'train' and not p['uses_test_data']
        assert p['average_weights'] == [.5, .5] and p['parent_steps'] == [18000, 19929]
        parents = []
        for path, digest in zip(p['parent_checkpoints'], p['parent_sha256']):
            assert sha256_file(path) == digest
            parents.append(torch.load(path, map_location='cpu', weights_only=False))
        assert parents[0]['provenance'] == parents[1]['provenance']
        assert [x['step'] for x in parents] == p['parent_steps']
        original = parents[0]['provenance']
        assert original['source_sha256'] == p['parent_training_source_sha256'] == sha256_file(root/'scripts/145_train_modern_clinical_lora.py')
        assert original['fit_split'] == 'train' and not original['uses_test_data']
        for key in ('config', 'model_revision', 'model_sha256', 'token_provenance_sha256',
                    'cache_report_hashes', 'base_sha256', 'trained_parameters', 'pretraining_corpus_contains'):
            assert p[key] == original[key]
        assert set(saved['state']) == set(parents[0]['state']) == set(parents[1]['state'])
        for name, value in saved['state'].items():
            assert torch.equal(value, .5*parents[0]['state'][name] + .5*parents[1]['state'][name])
        del parents
        token_root = Path(p['config']['tokens'])
        assert sha256_file(token_root/'provenance.json') == p['token_provenance_sha256']
        tokens = read_json(token_root/'provenance.json')
        model_path = Path(tokens['model_path'])
        assert tokens['model_revision'] == p['model_revision']
        assert sha256_file(model_path/'model.safetensors') == p['model_sha256']
        assert sha256_file(model_path/'tokenizer.json') == tokens['tokenizer_sha256']
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
        self.max_length = p['config']['max_length']
        self.tokenized_length = tokens['max_length']
        assert self.max_length == 512 and self.tokenized_length == 2048
        self.device = torch.device(device)
        raw = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=738,
            local_files_only=True, dtype=torch.bfloat16, attn_implementation=p['config']['attention'])
        self.model = get_peft_model(raw, LoraConfig(task_type=TaskType.SEQ_CLS, r=16,
            lora_alpha=32, lora_dropout=.05, target_modules=['Wqkv','Wo','Wi'], modules_to_save=['head','classifier']))
        self.model.to(device)
        parameters = {n: v for n,v in self.model.named_parameters() if v.requires_grad}
        assert set(parameters) == set(saved['state']) == set(p['trained_parameters'])
        with torch.no_grad():
            for name, value in parameters.items():
                value.data = value.data.float()
                value.copy_(saved['state'][name].to(device))
                assert torch.equal(value.cpu(), saved['state'][name])
        self.model.requires_grad_(False).eval()
        self.provenance, self.selection = p, saved['selected']
        self.checkpoint_steps = p['parent_steps']
        self.artifact_hashes = {str(checkpoint): checkpoint_digest,
            str(model_path/'model.safetensors'): p['model_sha256'],
            str(model_path/'tokenizer.json'): tokens['tokenizer_sha256'],
            str(token_root/'provenance.json'): p['token_provenance_sha256'],
            **dict(zip(p['parent_checkpoints'], p['parent_sha256']))}
        assert sha256_file(checkpoint) == checkpoint_digest

    def tokenize(self, texts):
        batch = self.tokenizer(texts, padding=True, truncation=True,
            max_length=self.tokenized_length, return_tensors='pt')
        ids, mask = batch['input_ids'][:,:self.max_length], batch['attention_mask'][:,:self.max_length]
        length = int(mask.sum(1).max())
        return {'input_ids': ids[:,:length].to(self.device), 'attention_mask': mask[:,:length].to(self.device)}

    @torch.inference_mode()
    def scores(self, texts, batch_size=16):
        scores = []
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            for offset in range(0,len(texts),batch_size):
                value = self.model(**self.tokenize(texts[offset:offset+batch_size])).logits.float()
                assert torch.isfinite(value).all()
                scores.append(value.cpu().numpy())
        return np.concatenate(scores)
