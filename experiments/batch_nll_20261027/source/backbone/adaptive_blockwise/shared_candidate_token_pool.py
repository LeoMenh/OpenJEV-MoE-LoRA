"""Several frozen readout heads share one unchanged clinical token forward pass."""
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax
from .candidate_token_expert import CandidateTokenBackbone
from .candidate_token_readout import CandidateTokenReadout, clinical_candidates
from .io_utils import read_json, sha256_file


class SharedCandidateTokenPool(CandidateTokenBackbone):
    def __init__(self, root, clinical_expert, items):
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
