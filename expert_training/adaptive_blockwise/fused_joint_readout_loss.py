"""Train a proposed clinical readout against a frozen 50/50 final mixture."""
import math

import torch
from torch.nn import functional as F


def joint_readout_loss(logits, frozen_prior, first_labels, second_labels, alpha, objective):
    if objective not in ('fused_nll', 'hybrid_nll'):
        raise ValueError(objective)
    clinical_log = F.log_softmax(logits, dim=-1)
    prior_log = frozen_prior.clamp_min(torch.finfo(frozen_prior.dtype).tiny).log()
    mixture_log = torch.logaddexp(clinical_log - math.log(2), prior_log - math.log(2))
    fused_loss = alpha * F.nll_loss(mixture_log, first_labels) + (1 - alpha) * F.nll_loss(mixture_log, second_labels)
    if objective == 'fused_nll':
        return fused_loss
    clinical_loss = alpha * F.nll_loss(clinical_log, first_labels) + (1 - alpha) * F.nll_loss(clinical_log, second_labels)
    return .5 * clinical_loss + .5 * fused_loss
