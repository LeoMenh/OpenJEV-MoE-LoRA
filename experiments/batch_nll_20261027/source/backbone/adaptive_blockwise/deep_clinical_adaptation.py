"""Matched full-pass LoRA refinement with optional all-block dense matrices.

Qwen MoE remains unchanged. The dense clinical arm updates all 112 encoder
attention/MLP base linear weights; embeddings and normalization stay frozen.
"""
from pathlib import Path
import torch
from .modern_clinical_expert import ModernClinicalExpert
from .io_utils import read_json, sha256_file


def build_deep_clinical_model(root, checkpoint, arm):
    if arm not in ('lora_control', 'all28_dense'):
        raise ValueError(arm)
    expert = ModernClinicalExpert(root, checkpoint)
    assert expert.step == 18000 and expert.max_length == expert.tokenized_length == 2048
    model = expert.model
    named = dict(model.named_parameters())
    original = list(expert.provenance['trained_parameters'])
    dense = [name for name, parameter in named.items()
             if '.model.layers.' in name and name.endswith('.base_layer.weight')
             and parameter.ndim == 2]
    assert len(dense) == 112 and not set(dense).intersection(original), dense
    assert {int(name.split('.model.layers.', 1)[1].split('.')[0])
            for name in dense} == set(range(28))
    for name in dense:
        named[name].data = named[name].data.float()
    train_names = set(original) | (set(dense) if arm == 'all28_dense' else set())
    for name, parameter in named.items():
        parameter.requires_grad_(name in train_names)
    trainable = {name: parameter for name, parameter in named.items() if parameter.requires_grad}
    saved_names = set(original) | set(dense)
    saved_parameters = {name: parameter for name, parameter in named.items() if name in saved_names}
    assert set(trainable) == train_names
    assert all(parameter.dtype == torch.float32 for parameter in saved_parameters.values())
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    return expert, trainable, saved_parameters, dense


class DeepAdaptedClinicalExpert(ModernClinicalExpert):
    """Reload one bound dense/control snapshot for actual text-only inference."""
    def __init__(self, root, checkpoint):
        root, checkpoint = Path(root).resolve(), Path(checkpoint).resolve()
        cp = root / 'runs/mature_long_all28_adaptation_development_20261012_v1/contract.json'
        contract, cp_hash = read_json(cp), sha256_file(cp)
        checkpoint_hash = sha256_file(checkpoint)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        provenance = saved['provenance']
        assert provenance['fit_split'] == 'train' and not provenance['uses_test_data']
        assert provenance['contract_sha256'] == cp_hash
        assert provenance['source_sha256'] == contract['trainer_source_sha256']
        assert provenance['factory_source_sha256'] == contract['factory_source_sha256'] == sha256_file(__file__)
        for key in ('trainer_source_sha256', 'factory_source_sha256'):
            source = root / ('scripts/558_train_mature_all28_adaptation.py' if key.startswith('trainer')
                             else 'adaptive_blockwise/deep_clinical_adaptation.py')
            assert sha256_file(source) == contract[key]
        assert saved['parent_step'] == 18000
        assert saved['additional_step'] in [0, 2, *contract['training']['evaluation_steps']]
        assert provenance['config'] == contract['training']
        assert provenance['seed'] in contract['training']['seeds']
        assert sha256_file(contract['initial_checkpoint']) == contract['initial_checkpoint_sha256']
        expert, trainable, saved_parameters, dense = build_deep_clinical_model(
            root, contract['initial_checkpoint'], provenance['arm'])
        assert expert.artifact_hashes == provenance['loaded_artifact_hashes']
        assert set(trainable) == set(provenance['trained_parameter_names'])
        assert set(saved_parameters) == set(saved['state']) == set(provenance['saved_parameter_names'])
        assert dense == provenance['dense_matrix_parameter_names']
        assert sum(p.numel() for p in trainable.values()) == provenance['trainable_parameter_count']
        with torch.no_grad():
            for name, parameter in saved_parameters.items():
                value = saved['state'][name]
                assert value.shape == parameter.shape and value.dtype == torch.float32
                assert torch.isfinite(value).all(), name
                if provenance['arm'] == 'lora_control' and name in dense:
                    assert torch.equal(parameter.detach().cpu(), value), name
                parameter.copy_(value.to(parameter.device))
        self.__dict__.update(expert.__dict__)
        self.model.gradient_checkpointing_disable()
        self.model.requires_grad_(False).eval()
        self.parent_provenance, self.provenance = expert.provenance, provenance
        self.parent_step, self.additional_step = 18000, saved['additional_step']
        self.step, self.selection = 18000 + self.additional_step, None
        self.artifact_hashes[str(checkpoint)] = checkpoint_hash
        self.artifact_hashes[str(cp)] = cp_hash
        self.artifact_hashes[str(Path(__file__).resolve())] = sha256_file(__file__)
        assert sha256_file(checkpoint) == checkpoint_hash and sha256_file(cp) == cp_hash
