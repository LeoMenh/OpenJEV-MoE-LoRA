"""Four distinct rival-readout seeds share the unchanged clinical encoder.

Explicit preflight mode permits only two-update smoke checkpoints. Production
requires completed full-budget training and each selected best checkpoint.
"""
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax
from .candidate_token_expert import CandidateTokenBackbone
from .candidate_token_readout import clinical_candidates
from .rival_candidate_token_readout import RivalCandidateTokenReadout as CandidateTokenReadout
from .io_utils import read_json, sha256_file


def validate_checkpoint_items(items):
    assert isinstance(items, list) and len(items) == 4
    assert [item['seed'] for item in items] == [20261031, 20261032, 20261033, 20261034]
    paths = []
    for item in items:
        assert set(item) == {'path', 'sha256', 'seed', 'step'}
        assert type(item['seed']) is int and type(item['step']) is int and item['step'] > 0
        assert isinstance(item['path'], str) and item['path']
        digest = item['sha256']
        assert isinstance(digest, str) and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest)
        paths.append(str(Path(item['path']).resolve()))
    assert len(set(paths)) == 4, 'Four distinct checkpoint paths are required.'


class SharedRivalCandidateTokenPool(CandidateTokenBackbone):
    def __init__(self, root, clinical_expert, items, *, allow_preflight=False):
        validate_checkpoint_items(items)
        assert type(allow_preflight) is bool
        self.expert = clinical_expert
        self.native = clinical_expert.model.base_model.model
        assert self.expert.max_length == 2048 and items
        assert not any(parameter.requires_grad for parameter in self.expert.model.parameters())
        prototypes = self.prototypes()
        self.heads, self.provenances, self.steps = [], [], []
        self.artifact_hashes = dict(clinical_expert.artifact_hashes)
        checked_contracts = {}
        for item in items:
            path = Path(item['path'])
            digest = sha256_file(path)
            assert digest == item['sha256']
            saved = torch.load(path, map_location='cpu', weights_only=False)
            provenance = saved['provenance']
            cp = Path(provenance['contract_path'])
            if cp not in checked_contracts:
                assert sha256_file(cp) == provenance['contract_sha256']
                contract = read_json(cp)
                for source, expected in {**contract['artifact_hashes'], **contract['source_hashes']}.items():
                    assert sha256_file(source) == expected, source
                checked_contracts[cp] = contract
                self.artifact_hashes.update(contract['artifact_hashes'])
                self.artifact_hashes.update(contract['source_hashes'])
                self.artifact_hashes[str(cp)] = provenance['contract_sha256']
            contract = checked_contracts[cp]
            assert provenance['contract_sha256'] == sha256_file(cp)
            assert provenance['fit_split'] == 'train' and not provenance['uses_test_data']
            assert provenance['mode'] in contract['arms'] and provenance['seed'] in contract['training']['seeds']
            assert saved['step'] == item['step'] and provenance['seed'] == item['seed']
            assert provenance['mode'] == 'rival_attention'
            assert contract['experiment_kind'] == 'rival_candidate_token_readout_matched'
            if allow_preflight:
                assert not contract['formal_training_allowed'] and not provenance['formal_training_started']
                assert saved['step'] == 2 and path.name == 'smoke.pt'
            else:
                assert contract['formal_training_allowed'] and provenance['formal_training_started']
                assert saved['step'] in contract['training']['evaluation_steps'] and path.name == 'best.pt'
                assert read_json(path.parent / 'process.json')['status'] == 'complete'
                final = read_json(path.parent / 'report.json')
                assert final['passed'] and final['actual_steps'] == contract['training']['steps']
                assert final['checkpoint_sha256'] == digest
            trainer = Path(contract['training_sources']['rival_attention'])
            assert provenance['source_sha256'] == contract['source_hashes'][str(trainer)]
            assert torch.equal(prototypes.cpu(), saved['state']['prototypes'])
            cfg = contract['architecture']
            head = CandidateTokenReadout(prototypes, provenance['mode'], cfg['width'], cfg['dropout'], cfg['correction_cap']).cuda()
            assert set(head.state_dict()) == set(saved['state'])
            for name, value in saved['state'].items():
                assert value.shape == head.state_dict()[name].shape and torch.isfinite(value).all(), name
            head.load_state_dict(saved['state'])
            head.requires_grad_(False).eval()
            self.heads.append(head)
            self.provenances.append(provenance)
            self.steps.append(saved['step'])
            self.artifact_hashes[str(path)] = digest
            assert sha256_file(path) == digest
        assert len({value['contract_sha256'] for value in self.provenances}) == 1
        assert len({value['mode'] for value in self.provenances}) == 1
        self.candidate_count = contract['architecture']['candidate_count']
        self.artifact_hashes[str(Path(__file__))] = sha256_file(__file__)

    @torch.no_grad()
    def probabilities(self, texts):
        assert 0 < len(texts) <= 16
        previous = torch.get_float32_matmul_precision()
        try:
            # Head training is FP32 outside autocast; preserve its precision.
            torch.set_float32_matmul_precision('highest')
            features, mask, base = self.encode(self.expert.tokenize(texts))
            candidates = clinical_candidates(base, self.candidate_count)
            pool = np.zeros((len(texts), 738), dtype=np.float64)
            for head in self.heads:
                scores = head(features, mask, base, candidates).cpu().numpy()
                pool += softmax(scores, axis=1).astype(np.float64)
            pool /= len(self.heads)
            return softmax(base.cpu().numpy(), axis=1), pool
        finally:
            torch.set_float32_matmul_precision(previous)
