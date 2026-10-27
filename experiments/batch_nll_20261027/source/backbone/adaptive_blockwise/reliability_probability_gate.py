"""A small output gate, distinct from Qwen's internal blockwise expert routing."""
import numpy as np
from scipy.special import expit


FEATURE_NAMES = (
    'moe_entropy', 'clinical_entropy', 'moe_top_probability', 'clinical_top_probability',
    'moe_top2_margin', 'clinical_top2_margin', 'top_label_disagreement',
    'clinical_probability_at_moe_top', 'moe_probability_at_clinical_top',
    'jensen_shannon_divergence',
)


def reliability_features(moe, clinical):
    """Probability-only inputs; labels and sample IDs cannot enter this API."""
    assert moe.shape == clinical.shape and moe.ndim == 2
    assert np.isfinite(moe).all() and np.isfinite(clinical).all()
    assert (moe >= 0).all() and (clinical >= 0).all()
    assert np.allclose(moe.sum(1), 1., atol=2e-6)
    assert np.allclose(clinical.sum(1), 1., atol=2e-6)
    p, q = moe.astype(np.float64), clinical.astype(np.float64)
    pi, qi = p.argmax(1), q.argmax(1)
    rows = np.arange(len(p))
    pt = np.sort(np.partition(p, -2, axis=1)[:, -2:], axis=1)
    qt = np.sort(np.partition(q, -2, axis=1)[:, -2:], axis=1)
    pm, qm = np.maximum(p, 1e-30), np.maximum(q, 1e-30)
    entropy_p = -(p * np.log(pm)).sum(1)
    entropy_q = -(q * np.log(qm)).sum(1)
    middle = np.maximum((p + q) / 2, 1e-30)
    js = .5 * ((p * (np.log(pm) - np.log(middle))).sum(1) +
               (q * (np.log(qm) - np.log(middle))).sum(1))
    return np.column_stack((entropy_p, entropy_q, pt[:, 1], qt[:, 1],
                            pt[:, 1] - pt[:, 0], qt[:, 1] - qt[:, 0],
                            (pi != qi).astype(float), q[rows, pi], p[rows, qi], js))


def weights_from_features(features, model):
    if model['kind'] == 'identity':
        return np.full(len(features), .35, dtype=np.float64)
    if model['kind'] == 'scalar':
        design = np.ones((len(features), 1), dtype=np.float64)
    elif model['kind'] == 'affine':
        mean, scale = np.array(model['mean']), np.array(model['scale'])
        design = np.column_stack((np.ones(len(features)), (features - mean) / scale))
    else:
        raise ValueError(model['kind'])
    # This interval contains the exact .35 identity at zero coefficients.
    return .05 + .60 * expit(design @ np.array(model['coefficients']))


def fuse_with_weights(moe, clinical, weights):
    assert len(moe) == len(clinical) == len(weights)
    assert np.isfinite(weights).all() and (weights >= .05).all() and (weights <= .65).all()
    w = np.asarray(weights, dtype=np.float32)[:, None]
    return (1 - w) * moe + w * clinical


def fused_probabilities(moe, clinical, model):
    features = reliability_features(moe, clinical)
    return fuse_with_weights(moe, clinical, weights_from_features(features, model))
