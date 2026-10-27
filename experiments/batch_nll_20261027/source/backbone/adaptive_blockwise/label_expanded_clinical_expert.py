"""Raw-text reader for explicitly identified matched label-expansion weights.

Expanded historical-tune supervision is retained as training provenance. This
reader does not masquerade a refit as the original train-only step18000 model.
"""
from pathlib import Path

import torch

from .io_utils import read_json, sha256_file
from .modern_clinical_expert import ModernClinicalExpert


class LabelExpandedClinicalExpert(ModernClinicalExpert):
    def __init__(self, root, checkpoint, device='cuda:0'):
        root, checkpoint = Path(root), Path(checkpoint)
        saved_digest = sha256_file(checkpoint)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        provenance = saved['provenance']
        contract_path = root / 'runs/matched_label_expansion_development_v2/contract.json'
        contract = read_json(contract_path)
        assert provenance['contract_sha256'] == sha256_file(contract_path)
        trainer = root / 'scripts/403_train_matched_label_expansion_v2.py'
        assert provenance['source_sha256'] == sha256_file(trainer) == contract['source_sha256']
        assert not provenance['uses_test_data'] and provenance['first_original_test_summary_already_known']
        assert provenance['config'] == contract['training']
        assert provenance['initial_step'] == 18000
        assert provenance['arm'] in ('train_control', 'train_plus_tune')
        expanded = provenance['arm'] == 'train_plus_tune'
        assert provenance['fit_split'] == ('train_plus_historical_tune' if expanded else 'train')
        assert provenance['fit_validation_rows'] == (11808 if expanded else 0)
        assert provenance['heldout_audit_rows'] == 11809
        assert provenance['fit_tune_positions_sha256'] == sha256_file(provenance['fit_tune_positions'])
        assert provenance['fit_tune_positions_sha256'] == contract['membership_sha256']
        initial = Path(contract['initial_checkpoint'])
        assert provenance['initial_checkpoint_sha256'] == sha256_file(initial)
        super().__init__(root, initial, device)
        assert self.step == 18000 and self.max_length == 512 and self.tokenized_length == 2048
        assert self.provenance == provenance['parent_provenance']
        assert provenance['trained_parameters'] == self.provenance['trained_parameters']
        parameters = dict(self.model.named_parameters())
        assert set(saved['state']) == set(provenance['trained_parameters'])
        with torch.no_grad():
            for name, value in saved['state'].items():
                assert value.shape == parameters[name].shape and torch.isfinite(value).all()
                parameters[name].copy_(value.to(parameters[name].device))
                assert torch.equal(parameters[name].cpu(), value)
        self.model.requires_grad_(False).eval()
        self.parent_step = 18000
        self.step = saved['step']
        assert self.step in (2, 1000, 2000)
        self.training_max_length = provenance['config']['training_length']
        self.max_length = provenance['config']['inference_length']
        assert self.training_max_length == 512 and self.max_length == 1536
        self.provenance = provenance
        self.selection = {'temperature': 1., 'weight': .35}
        self.inference_config = {
            'training_length': 512, 'inference_length': 1536,
            'tokenized_length': 2048, 'operation': 'Common2048tokenizer output right-sliced1536',
            'arm': provenance['arm'], 'parent_step': 18000,
            'additional_optimizer_steps': self.step,
            'historical_validation_tune_labels_fit': expanded,
            'full_validation_is_not_heldout_for_expanded_arm': expanded,
        }
        assert sha256_file(checkpoint) == saved_digest
        self.artifact_hashes.update({
            str(checkpoint): saved_digest, str(contract_path): provenance['contract_sha256'],
            str(trainer): provenance['source_sha256'],
            str(root / 'adaptive_blockwise/modern_clinical_expert.py'):
                sha256_file(root / 'adaptive_blockwise/modern_clinical_expert.py'),
            str(root / 'adaptive_blockwise/label_expanded_clinical_expert.py'): sha256_file(__file__),
            str(Path(provenance['fit_tune_positions'])): provenance['fit_tune_positions_sha256'],
        })
