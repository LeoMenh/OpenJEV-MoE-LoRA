"""Frozen mature2048-trained clinical weights with explicit inference budgets.

This changes token visibility only. It does not adapt weights or establish
benefit beyond the training context. Standard tokenizer special tokens are
preserved, including the end token at each truncation budget.
"""
from pathlib import Path

from .modern_clinical_expert import ModernClinicalExpert
from .io_utils import sha256_file


def validate_budget(length):
    assert type(length) is int and length in (2048, 4096, 8192)
    return length


class MatureExtendedContextExpert(ModernClinicalExpert):
    def __init__(self, root, checkpoint, inference_length, device='cuda:0'):
        length = validate_budget(inference_length)
        super().__init__(root, checkpoint, device)
        assert self.step == 18000 and self.max_length == self.tokenized_length == 2048
        self.training_max_length = 2048
        assert self.model.config.max_position_embeddings == 8192
        assert not any(parameter.requires_grad for parameter in self.model.parameters())
        self.max_length = self.tokenized_length = length
        self.inference_config = {
            'training_max_length': 2048, 'inference_max_length': length,
            'input_policy': 'Standard tokenizer prefix truncation with its native special tokens at this budget.',
            'weights_changed': False, 'candidate_from_test': False,
            'scope': 'Inference-context intervention; long-context accuracy and deployment remain unproved.',
        }
        source = Path(__file__).resolve()
        self.artifact_hashes[str(source)] = sha256_file(source)

