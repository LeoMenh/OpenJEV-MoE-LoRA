"""Frozen MoE readout with train-only memory and shared severity correction."""
from pathlib import Path
import joblib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.special import softmax
from .io_utils import read_json, sha256_file


class MemoryReadoutFusion:
    def __init__(self, root, checkpoint_sha256, device='cuda:0', projection_path=None, extra_head_paths=(), hierarchical_path=None, hierarchical_weight=.2):
        root = Path(root)
        self.device = device
        cache = root / 'runs/frozen_features_v1'
        report = read_json(cache / 'train/report.json')
        assert report['split'] == 'train' and not report['uses_test_data']
        assert report['checkpoint_sha256'] == checkpoint_sha256
        direct_path = root / 'runs/direct_feature_head_v1/best.pt'
        saved = torch.load(direct_path, map_location='cpu', weights_only=False)
        provenance = saved['provenance']
        assert provenance['cache_report_hashes']['train'] == sha256_file(cache / 'train/report.json')
        assert provenance['source_sha256'] == sha256_file(root / 'scripts/58_train_direct_feature_head.py')
        cfg = provenance['config']
        self.head = nn.Sequential(nn.LayerNorm(provenance['feature_size']),
            nn.Linear(provenance['feature_size'], cfg['hidden']), nn.GELU(),
            nn.Dropout(cfg['dropout']), nn.Linear(cfg['hidden'], provenance['num_classes'])).to(device)
        self.head.load_state_dict(saved['state'], strict=True)
        self.head.requires_grad_(False).eval()
        self.hierarchical_head = None
        self.hierarchical_weight = hierarchical_weight
        if not 0 <= hierarchical_weight <= 1:
            raise ValueError('Hierarchical weight must be in [0,1]')
        if hierarchical_path is not None:
            from .hierarchical_feature_head import HierarchicalFeatureHead
            hierarchical_path = Path(hierarchical_path).resolve()
            hierarchical = torch.load(hierarchical_path, map_location='cpu', weights_only=False)
            hp = hierarchical['provenance']; hc = hp['config']
            assert hp['cache_report_hashes'] == provenance['cache_report_hashes']
            assert hp['hierarchy_source_sha256'] == sha256_file(root/'adaptive_blockwise/hierarchical_feature_head.py')
            assert hp['source_sha256'] == sha256_file(root/'scripts/68_train_hierarchical_head.py')
            self.hierarchical_head = HierarchicalFeatureHead(hp['feature_size'], hc['family_ids'], hc['severity_ids'], hc['hidden'], hc['dropout']).to(device)
            self.hierarchical_head.load_state_dict(hierarchical['state'], strict=True)
            self.hierarchical_head.requires_grad_(False).eval()
        self.extra_heads = []
        extra_hashes = {}
        for path in extra_head_paths:
            path = Path(path).resolve()
            replica = torch.load(path, map_location='cpu', weights_only=False)
            prov = replica['provenance']
            assert prov['cache_report_hashes'] == provenance['cache_report_hashes']
            assert prov['source_sha256'] == sha256_file(root/'scripts/65_train_direct_head_replica.py')
            rcfg = prov['config']
            extra = nn.Sequential(nn.LayerNorm(prov['feature_size']), nn.Linear(prov['feature_size'], rcfg['hidden']),
                nn.GELU(), nn.Dropout(rcfg['dropout']), nn.Linear(rcfg['hidden'], prov['num_classes'])).to(device)
            extra.load_state_dict(replica['state'], strict=True)
            extra.requires_grad_(False).eval()
            self.extra_heads.append(extra)
            extra_hashes[str(path.relative_to(root))] = sha256_file(path)
        center_path = root / 'runs/semantic_memory_v1/train_center.npy'
        self.center = torch.tensor(np.load(center_path), device=device)
        features = np.load(cache / 'train/features.npy', mmap_mode='r')
        scores = np.load(cache / 'train/scores.npz')
        assert len(features) == report['num_samples']
        assert np.array_equal(scores['positions'], np.arange(len(features)))
        self.index = torch.tensor(np.asarray(features), device=device)
        self.index = F.normalize(self.index - self.center, dim=1)
        self.projection = None
        if projection_path is not None:
            projection_path = Path(projection_path).resolve()
            pca_report = read_json(projection_path.parent / 'report.json')
            assert pca_report['fit_split'] == 'train' and not pca_report['uses_test_data']
            assert pca_report['cache_report_sha256']['train'] == sha256_file(cache/'train/report.json')
            self.projection = torch.tensor(np.load(projection_path), device=device)
            assert self.projection.shape == (features.shape[1], pca_report['dimensions'])
            self.index = F.normalize(self.index @ self.projection, dim=1)
        self.labels = torch.tensor(scores['labels'], dtype=torch.long, device=device)
        self.classes = provenance['num_classes']
        lexical = root / 'runs/fulltext_lexical_v1'
        self.vectorizer = joblib.load(lexical / 'vectorizer.joblib')
        self.svm = joblib.load(lexical / 'linear_svm.joblib')
        severity = root / 'runs/severity_lexical_v1'
        self.severity = joblib.load(severity / 'classifier.joblib')
        assert np.array_equal(self.svm.classes_, np.arange(self.classes))
        assert np.array_equal(self.severity.classes_, np.arange(3))
        family = read_json(severity / 'family_map.json')
        self.family_ids = list(family['families'].values())
        self.severity_ids = np.asarray(family['severity'])
        self.artifact_hashes = {str(p.relative_to(root)): sha256_file(p) for p in (
            direct_path, center_path, cache/'train/features.npy', cache/'train/scores.npz',
            lexical/'vectorizer.joblib', lexical/'linear_svm.joblib',
            severity/'classifier.joblib', severity/'family_map.json')}
        self.artifact_hashes.update(extra_hashes)
        expected_cache = read_json(root/'runs/residual_feature_head_v1/provenance.json')['cache_sha256']['train']
        for name in ('features.npy', 'scores.npz'):
            assert self.artifact_hashes[str((cache/'train'/name).relative_to(root))] == expected_cache[name]
        severity_report = read_json(severity/'report.json')
        assert self.artifact_hashes[str((lexical/'vectorizer.joblib').relative_to(root))] == severity_report['vectorizer_sha256']
        if projection_path is not None:
            self.artifact_hashes[str(projection_path.relative_to(root))] = sha256_file(projection_path)
        if hierarchical_path is not None:
            self.artifact_hashes[str(hierarchical_path.relative_to(root))] = sha256_file(hierarchical_path)

    @torch.inference_mode()
    def probabilities(self, features, base_logits, texts):
        if len(features) != len(texts):
            raise ValueError('Text and feature batch lengths differ')
        features = features.to(self.device, dtype=torch.float32)
        q = F.normalize(features - self.center, dim=1)
        if self.projection is not None:
            q = F.normalize(q @ self.projection, dim=1)
        similarities, ids = (q @ self.index.T).topk(16, dim=1)
        weights = F.softmax(similarities / .07, dim=1)
        memory = torch.zeros((len(q), self.classes), device=self.device)
        memory.scatter_add_(1, self.labels[ids], weights)
        direct = self.head(features).softmax(-1).cpu().numpy()
        for head in self.extra_heads:
            direct += head(features).softmax(-1).cpu().numpy()
        direct /= 1 + len(self.extra_heads)
        lexical = self.vectorizer.transform(texts)
        svm = self.svm.decision_function(lexical).astype(np.float32)
        base = softmax(base_logits.float().cpu().numpy() + 2 * svm, axis=1)
        probabilities = .25 * base + .25 * memory.cpu().numpy() + .5 * direct
        severity = self.severity.predict_proba(lexical)
        for ids in self.family_ids:
            old = probabilities[:, ids].copy()
            new = old * np.maximum(severity[:, self.severity_ids[ids]], 1e-8) ** .5
            probabilities[:, ids] = new / (new.sum(1, keepdims=True) + 1e-30) * old.sum(1, keepdims=True)
        if self.hierarchical_head is not None:
            h = self.hierarchical_head(features).softmax(-1).cpu().numpy()
            probabilities = (1-self.hierarchical_weight)*probabilities+self.hierarchical_weight*h
        return probabilities
