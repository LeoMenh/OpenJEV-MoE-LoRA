"""Prefix BioLinkBERT LoRA with additional encoder base linear adaptation."""
from pathlib import Path
import torch
from transformers import AutoTokenizer
from .biolink_document_expert import BioLinkDocumentClassifier, BioLinkDocumentExpert
from .io_utils import read_json, sha256_file


class BioLinkDenseClassifier(BioLinkDocumentClassifier):
    def __init__(self, model_path, arm='prefix512'):
        assert arm == 'prefix512'
        super().__init__(model_path, arm)
        named = dict(self.named_parameters())
        self.common_parameter_names = [name for name, p in named.items() if p.requires_grad]
        self.dense_parameter_names = [name for name, p in named.items()
                                     if name.startswith('encoder.') and name.endswith('.base_layer.weight')
                                     and p.ndim == 2]
        assert len(self.dense_parameter_names) == 144
        assert {int(name.split('.encoder.layer.', 1)[1].split('.')[0])
                for name in self.dense_parameter_names} == set(range(24))
        for name in self.dense_parameter_names:
            named[name].data = named[name].data.float()
            named[name].requires_grad_(True)
        assert all(p.dtype == torch.float32 for p in self.parameters() if p.requires_grad)


class BioLinkDenseExpert(BioLinkDocumentExpert):
    def __init__(self, root, checkpoint):
        root, checkpoint = Path(root), Path(checkpoint)
        cp = root / 'runs/biolink_dense_complement_development_20261013_v1/contract.json'
        contract, digest = read_json(cp), sha256_file(checkpoint)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        provenance = saved['provenance']
        assert provenance['fit_split'] == 'train' and not provenance['uses_test_data']
        assert provenance['contract_sha256'] == sha256_file(cp)
        assert provenance['source_sha256'] == contract['trainer_source_sha256'] == sha256_file(root / 'scripts/567_train_biolink_dense.py')
        assert provenance['factory_source_sha256'] == contract['factory_source_sha256'] == sha256_file(__file__)
        assert provenance['arm'] == 'prefix512' and provenance['seed'] in contract['training']['seeds']
        assert sha256_file(root / 'adaptive_blockwise/biolink_document_expert.py') == contract['control_factory_source_sha256']
        assert sha256_file(root / 'adaptive_blockwise/biolink_chunking.py') == contract['chunking_source_sha256']
        assert sha256_file(contract['control_contract']) == contract['control_contract_sha256']
        for path, expected in contract['model_hashes'].items():
            assert sha256_file(path) == expected, path
        self.model = BioLinkDenseClassifier(contract['model_path']).cuda()
        parameters = {name: p for name, p in self.model.named_parameters() if p.requires_grad}
        assert set(parameters) == set(saved['state']) == set(provenance['trained_parameter_names'])
        assert sum(p.numel() for p in parameters.values()) == contract['trainable_parameters']
        assert set(self.model.dense_parameter_names) == set(provenance['dense_parameter_names'])
        with torch.no_grad():
            for name, parameter in parameters.items():
                value = saved['state'][name]
                assert value.shape == parameter.shape and value.dtype == parameter.dtype == torch.float32
                assert torch.isfinite(value).all(), name
                parameter.copy_(value.to(parameter.device))
        self.model.encoder.gradient_checkpointing_disable()
        self.model.requires_grad_(False).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(contract['model_path'], local_files_only=True, use_fast=True)
        self.arm, self.step, self.provenance = 'prefix512', saved['step'], provenance
        self.artifact_hashes = {str(checkpoint): digest, str(cp): sha256_file(cp), **contract['model_hashes'],
                               str(Path(__file__).resolve()): sha256_file(__file__)}
        assert sha256_file(checkpoint) == digest
