"""Interpolate fixed arithmetic pooling with a weighted geometric pool."""
import numpy as np
from scipy.special import softmax


def agreement_pool(main, clinical, geometric_clinical_weight, geometric_mix, epsilon=1e-30):
    if main.shape != clinical.shape or main.ndim != 2 or main.shape[1] != 738:
        raise ValueError('Both branches must have matching [N,738] probabilities')
    if not 0 <= geometric_clinical_weight <= 1 or not 0 <= geometric_mix <= 1:
        raise ValueError('Weights outside [0,1]')
    arithmetic = .5 * main + .5 * clinical
    if geometric_mix == 0:
        return arithmetic
    main_log = np.log(np.maximum(main, epsilon))
    clinical_log = np.log(np.maximum(clinical, epsilon))
    geometric = softmax((1-geometric_clinical_weight)*main_log + geometric_clinical_weight*clinical_log, axis=1)
    return (1-geometric_mix)*arithmetic + geometric_mix*geometric
