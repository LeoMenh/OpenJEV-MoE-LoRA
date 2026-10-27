"""Partition all DRGs into documented severity families and singleton labels."""
import torch
from torch.nn import functional as F
from .io_utils import read_json


class ClinicalFamilyObjective:
    def __init__(self, mapping_path, device):
        documented = list(read_json(mapping_path)['families'].values())
        flat = [label for group in documented for label in group]
        assert len(flat) == len(set(flat)) and all(0 <= label < 738 for label in flat)
        covered = set(flat)
        groups = documented + [[label] for label in range(738) if label not in covered]
        assert sorted(label for group in groups for label in group) == list(range(738))
        lookup = torch.full((738, max(map(len, groups))), -1, dtype=torch.long)
        self.family_ids = torch.empty(738, dtype=torch.long, device=device)
        for family_id, labels in enumerate(groups):
            for label in labels:
                lookup[label, :len(labels)] = torch.tensor(labels)
                self.family_ids[label] = family_id
        self.valid = (lookup >= 0).to(device)
        self.lookup = lookup.clamp_min(0).to(device)
        self.metadata = {'documented_families': len(documented), 'documented_labels': len(flat),
                         'singleton_labels': 738 - len(flat), 'total_groups': len(groups),
                         'objective': 'CE with smoothing .05 + .25 * negative log summed true-family probability',
                         'singleton_policy': 'Unmapped DRGs each form their own family; singleton family NLL equals unsmoothed class CE.'}

    def components(self, logits, targets, label_smoothing=.05):
        logits = logits.float()
        ce = F.cross_entropy(logits, targets, label_smoothing=label_smoothing)
        family_logits = logits.gather(1, self.lookup[targets]).masked_fill(~self.valid[targets], -torch.inf)
        family_nll = (logits.logsumexp(1) - family_logits.logsumexp(1)).mean()
        return ce, family_nll

    def __call__(self, logits, targets, label_smoothing=.05):
        ce, family_nll = self.components(logits, targets, label_smoothing)
        return ce + .25 * family_nll
