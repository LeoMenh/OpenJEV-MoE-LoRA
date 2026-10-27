"""Label-free output gate for long clinical branch, including visibility/hierarchy.

An output mixture weight, distinct from the Qwen blockwise adapter routers.
"""
import numpy as np
from scipy.special import expit
from adaptive_blockwise.reliability_probability_gate import FEATURE_NAMES, reliability_features

LENGTH_FEATURE_NAMES = ('clinical_length_up_to512', 'clinical_excess512',
                        'clinical_excess1536', 'clinical_at2048_cap')
FAMILY_FEATURE_NAMES = ('moe_family_entropy', 'clinical_family_entropy',
                        'moe_family_top_probability', 'clinical_family_top_probability',
                        'aggregate_family_disagreement', 'clinical_support_at_moe_family',
                        'moe_support_at_clinical_family', 'family_js_divergence')
ALL_FEATURE_NAMES = tuple(FEATURE_NAMES) + LENGTH_FEATURE_NAMES + FAMILY_FEATURE_NAMES
GROUP_INDICES = {
    'probability': list(range(10)),
    'probability_length': list(range(14)),
    'probability_family': list(range(10)) + list(range(14, 22)),
    'all': list(range(22)),
}


def evidence_features(moe, clinical, clinical_lengths, families):
    """Only probabilities, nonpadding token counts and fixed class ontology."""
    base = reliability_features(moe, clinical)
    length = np.asarray(clinical_lengths, dtype=np.float64)
    families = np.asarray(families, dtype=np.int64)
    assert length.shape == (len(moe),) and ((length > 0) & (length <= 2048)).all()
    assert families.shape == (moe.shape[1],) and (families >= 0).all()
    length_features = np.column_stack((np.minimum(length, 512) / 512,
                                      np.maximum(length - 512, 0) / 1536,
                                      np.maximum(length - 1536, 0) / 512,
                                      (length == 2048).astype(float)))
    count = int(families.max()) + 1
    masses = []
    for probability in [moe, clinical]:
        mass = np.zeros((len(moe), count), dtype=np.float64)
        for cls, family in enumerate(families):
            mass[:, family] += probability[:, cls]
        masses.append(mass)
    p, q = masses
    pi, qi = p.argmax(1), q.argmax(1)
    rows = np.arange(len(p))
    lp, lq = np.log(np.maximum(p, 1e-30)), np.log(np.maximum(q, 1e-30))
    middle = np.log(np.maximum((p + q) / 2, 1e-30))
    family_features = np.column_stack((-(p * lp).sum(1), -(q * lq).sum(1),
                                      p[rows, pi], q[rows, qi], (pi != qi).astype(float),
                                      q[rows, pi], p[rows, qi],
                                      .5 * ((p * (lp - middle)).sum(1) + (q * (lq - middle)).sum(1))))
    result = np.column_stack((base, length_features, family_features))
    assert result.shape == (len(moe), len(ALL_FEATURE_NAMES)) and np.isfinite(result).all()
    return result


def weights_from_features(features, model):
    if model['kind'] == 'identity':
        return np.full(len(features), .5, dtype=np.float64)
    if model['kind'] == 'scalar':
        design = np.ones((len(features), 1), dtype=np.float64)
    elif model['kind'] == 'affine':
        subset = features[:, model['feature_indices']]
        normalized = (subset - np.asarray(model['mean'])) / np.asarray(model['scale'])
        design = np.column_stack((np.ones(len(features)), normalized))
    else:
        raise ValueError(model['kind'])
    return .05 + .9 * expit(design @ np.asarray(model['coefficients']))


def fuse_with_weights(moe, clinical, weights):
    assert len(moe) == len(clinical) == len(weights)
    assert np.isfinite(weights).all() and ((weights >= .05) & (weights <= .95)).all()
    weight = np.asarray(weights, dtype=np.float32)[:, None]
    return (1 - weight) * moe + weight * clinical
