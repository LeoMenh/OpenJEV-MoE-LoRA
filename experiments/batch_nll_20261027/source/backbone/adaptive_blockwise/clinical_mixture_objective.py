"""Train a complementary expert against a frozen MoE probability distribution."""
import math

import torch
from torch.nn import functional as F


class ClinicalMixtureObjective:
    def __init__(self, weight=.35, anchor=.25):
        if not 0 < weight < 1 or anchor < 0:
            raise ValueError('Mixture weight must be in (0,1); anchor must be nonnegative')
        self.weight = weight
        self.anchor = anchor
        self.metadata = {
            'formula': '-log((1-w)*softmax(frozen_moe)_y+w*softmax(clinical)_y)'
                       '+anchor*CE_label_smoothed(clinical,y)',
            'weight': weight, 'anchor': anchor,
            'prior_temperature': 1., 'clinical_temperature': 1.,
            'prior_receives_gradient': False,
            'training_prior': 'frozen original MoE train logits, canonical batch8',
            'limitation': 'In-sample training prior; deployment fixed ensemble differs from original backbone prior.'
        }

    def components(self, logits, targets, prior_logits, label_smoothing=.05):
        clinical_logprob = logits.float().log_softmax(-1)
        prior_logprob = prior_logits.detach().float().log_softmax(-1)
        log_mixture = torch.logaddexp(
            math.log1p(-self.weight) + prior_logprob,
            math.log(self.weight) + clinical_logprob)
        mixture_nll = F.nll_loss(log_mixture, targets)
        anchor_ce = F.cross_entropy(logits.float(), targets, label_smoothing=label_smoothing)
        return mixture_nll, anchor_ce

    def __call__(self, logits, targets, prior_logits, label_smoothing=.05):
        mixture_nll, anchor_ce = self.components(logits, targets, prior_logits, label_smoothing)
        return mixture_nll + self.anchor * anchor_ce
