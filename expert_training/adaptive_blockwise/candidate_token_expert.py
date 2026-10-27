"""Frozen clinical backbone and raw-text inference for candidate-token readouts."""
from pathlib import Path

import numpy as np
import torch
from .candidate_token_readout import CandidateTokenReadout, clinical_candidates
from .modern_clinical_expert import ModernClinicalExpert
from .io_utils import read_json, sha256_file


class CandidateTokenBackbone:
    def __init__(self, root, checkpoint):
        self.expert = ModernClinicalExpert(root, checkpoint)
        assert self.expert.max_length == 2048
        self.native = self.expert.model.base_model.model
        assert not any(p.requires_grad for p in self.expert.model.parameters())

    def prototypes(self):
        classifier = self.native.classifier.modules_to_save[self.expert.model.active_adapter]
        assert classifier.weight.shape == (738, 1024)
        return classifier.weight.detach().float().clone()

    def encode(self, inputs):
        observed = {}

        def capture(_module, _inputs, outputs):
            observed['tokens'] = outputs.last_hidden_state

        hook = self.native.model.register_forward_hook(capture)
        try:
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                base = self.expert.model(**inputs).logits.float()
                tokens = self.native.head(observed['tokens']).float()
            return tokens.detach(), inputs['attention_mask'], base.detach()
        finally:
            hook.remove()


class CandidateTokenExpert:
    def __init__(self, root, checkpoint):
        root, checkpoint = Path(root), Path(checkpoint)
        checkpoint_hash = sha256_file(checkpoint)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        provenance = saved['provenance']
        contract_path = Path(provenance['contract_path'])
        contract = read_json(contract_path)
        assert provenance['contract_sha256'] == sha256_file(contract_path)
        assert provenance['fit_split'] == 'train' and not provenance['uses_test_data']
        assert provenance['mode'] in contract['arms'] and provenance['seed'] in contract['training']['seeds']
        for path, digest in contract['source_hashes'].items():
            assert sha256_file(path) == digest, path
        for path, digest in contract['artifact_hashes'].items():
            assert sha256_file(path) == digest, path
        self.backbone = CandidateTokenBackbone(root, contract['clinical_checkpoint'])
        prototypes = self.backbone.prototypes()
        assert torch.equal(prototypes.cpu(), saved['state']['prototypes'])
        cfg = contract['architecture']
        self.head = CandidateTokenReadout(prototypes, provenance['mode'], cfg['width'],
                                         cfg['dropout'], cfg['correction_cap']).cuda()
        assert set(self.head.state_dict()) == set(saved['state'])
        for name, value in saved['state'].items():
            assert torch.isfinite(value).all(), name
            assert value.shape == self.head.state_dict()[name].shape
        self.head.load_state_dict(saved['state'])
        self.head.requires_grad_(False).eval()
        self.candidate_count = cfg['candidate_count']
        self.step, self.provenance = saved['step'], provenance
        self.artifact_hashes = {**contract['artifact_hashes'], **contract['source_hashes'],
                               str(checkpoint): checkpoint_hash, str(contract_path): sha256_file(contract_path)}
        assert sha256_file(checkpoint) == checkpoint_hash

    @torch.no_grad()
    def scores(self, texts, batch_size=16):
        result = []
        for start in range(0, len(texts), batch_size):
            inputs = self.backbone.expert.tokenize(texts[start:start + batch_size])
            features, mask, base = self.backbone.encode(inputs)
            candidates = clinical_candidates(base, self.candidate_count)
            logits = self.head(features, mask, base, candidates)
            assert torch.isfinite(logits).all()
            result.append(logits.cpu().numpy())
        return np.concatenate(result)
