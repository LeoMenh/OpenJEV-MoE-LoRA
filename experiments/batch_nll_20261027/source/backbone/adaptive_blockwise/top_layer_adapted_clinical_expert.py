"""Raw-text loading of matched clinical LoRA/top-two-layer refinement snapshots."""
import hashlib
from pathlib import Path

import torch

from .io_utils import read_json, sha256_file
from .modern_clinical_expert import ModernClinicalExpert
from .top_layer_clinical_adaptation import build_top_layer_clinical_model


class TopLayerAdaptedClinicalExpert(ModernClinicalExpert):
    def __init__(self, root, checkpoint):
        root, checkpoint = Path(root).resolve(), Path(checkpoint).resolve()
        contract_path = root / 'runs/mature_long_top2_adaptation_development_v1/contract.json'
        contract = read_json(contract_path)
        contract_digest = sha256_file(contract_path)
        with checkpoint.open('rb') as stream:
            digest = hashlib.sha256()
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
            checkpoint_digest = digest.hexdigest()
            stream.seek(0)
            saved = torch.load(stream, map_location='cpu', weights_only=False)
        provenance = saved['provenance']
        assert provenance['fit_split'] == 'train' and not provenance['uses_test_data']
        assert provenance['original_test_scored_assessments_known'] == 2
        assert provenance['contract_sha256'] == contract_digest
        assert provenance['config'] == contract['training']
        assert provenance['arm'] in contract['arms']
        assert provenance['seed'] in contract['training']['seeds']
        assert saved['parent_step'] == provenance['initial_clinical_step'] == 18000
        assert saved['additional_step'] in [0, *contract['training']['evaluation_steps']]
        trainer = root / 'scripts/457_train_mature_long_top2_adaptation.py'
        factory = root / 'adaptive_blockwise/top_layer_clinical_adaptation.py'
        assert provenance['source_sha256'] == contract['trainer_source_sha256'] == sha256_file(trainer)
        assert provenance['factory_source_sha256'] == contract['factory_source_sha256'] == sha256_file(factory)
        initial = Path(contract['initial_checkpoint'])
        assert provenance['initial_checkpoint_sha256'] == contract['initial_checkpoint_sha256'] == sha256_file(initial)

        expert, trainable, saved_parameters, dense_names = build_top_layer_clinical_model(
            root, initial, provenance['arm'])
        assert provenance['loaded_artifact_hashes'] == expert.artifact_hashes
        assert set(provenance['trained_parameter_names']) == set(trainable)
        assert set(provenance['saved_parameter_names']) == set(saved_parameters) == set(saved['state'])
        assert provenance['dense_matrix_parameter_names'] == dense_names and len(dense_names) == 8
        assert sum(p.numel() for p in trainable.values()) == provenance['trainable_parameter_count']
        assert sum(saved_parameters[n].numel() for n in dense_names) == provenance['dense_matrix_parameter_count']
        with torch.no_grad():
            for name, parameter in saved_parameters.items():
                value = saved['state'][name]
                assert value.shape == parameter.shape and value.dtype == parameter.dtype == torch.float32
                assert torch.isfinite(value).all(), name
                if provenance['arm'] == 'lora_control' and name in dense_names:
                    assert torch.equal(parameter.detach().cpu(), value), name
                parameter.copy_(value.to(parameter.device))
        self.__dict__.update(expert.__dict__)
        self.model.gradient_checkpointing_disable()
        self.model.requires_grad_(False).eval()
        assert all(not p.requires_grad for p in self.model.parameters())
        self.parent_provenance = expert.provenance
        self.provenance = provenance
        self.parent_step = saved['parent_step']
        self.additional_step = saved['additional_step']
        self.step = self.parent_step + self.additional_step
        self.selection = None
        self.tune_correct = saved['tune_correct']
        self.arm, self.seed = provenance['arm'], provenance['seed']
        self.dense_parameter_names = dense_names
        self.artifact_hashes[str(checkpoint)] = checkpoint_digest
        for source in [contract_path, trainer, factory, Path(__file__),
                       root / 'adaptive_blockwise/modern_clinical_expert.py']:
            self.artifact_hashes[str(source)] = sha256_file(source)
        assert sha256_file(checkpoint) == checkpoint_digest
        assert sha256_file(contract_path) == contract_digest

