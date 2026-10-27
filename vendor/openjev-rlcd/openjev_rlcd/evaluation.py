"""Scoring and paired statistics for the epistemic tasks (one correct option: MMLU-Pro, GSM8K-Verify).

Every arm is recomputed from its saved per-item distributions with the same functions, raw and after
temperature scaling fitted on the dev items (options beyond an item's count stay at -inf).
  score metrics : accuracy, Brier (multi-class, vs one-hot), NLL, ECE (10 bins, confidence vs correct)
  decisions     : AURC, label-free automation cov@e / realized risk@e, oracle coverage (selective_metrics)
Paired tests on test items (per-item values averaged over seeds first): mean difference with a 95%
bootstrap CI and a two-sided sign-flip permutation p-value -- tests of the MEAN, which is what an
expected proper score is.
"""
import math

import numpy as np
import torch

from .metrics import fit_temperature, selective_metrics

BUDGETS = (0.05, 0.1, 0.2, 0.3)
SPLITS, KMAX = None, 10  # SPLITS: the task's splits, set by the caller (TASKS[task]["load"]())


def padded_logits(dists):
    z = torch.full((len(dists), KMAX), float("-inf"), dtype=torch.float64)
    for i, p in enumerate(dists):
        z[i, :len(p)] = torch.tensor(p, dtype=torch.float64).clamp_min(1e-12).log()
    return z


def lists(z, items):
    return [row[:len(it["q"])] for row, it in zip(z.softmax(-1).tolist(), items)]


def score(items, probs):
    brier, nll, acc = [], [], []
    for it, p in zip(items, probs):
        y = it["q"].index(1.0)
        brier.append(sum((a - (i == y)) ** 2 for i, a in enumerate(p)))
        nll.append(-math.log(max(p[y], 1e-12)))
        acc.append(int(p.index(max(p)) == y))
    n = len(items)
    return dict(acc=sum(acc) / n, brier=sum(brier) / n, nll=sum(nll) / n, maxp=sum(max(p) for p in probs) / n,
                **selective_metrics(items, probs, BUDGETS), per_item_brier=brier, per_item_correct=acc,
                per_item_conf=[max(p) for p in probs])


def scored(test, dev, seed):
    ti, di = SPLITS["test"][:len(test)], SPLITS["dev"][:len(dev)]
    zt, zd = padded_logits(test), padded_logits(dev)
    t = fit_temperature(list(zd.float()), di, seed)
    return score(ti, lists(zt, ti)), score(ti, lists(zt / t, ti)), t


def aurc_boot(seeds, idx):
    """AURC (0/1 risk, ranked by confidence) on resampled items idx [B, n], averaged over seeds -> [B]."""
    out = 0.0
    for conf, risk in seeds:
        c, r = conf[idx], risk[idx]
        order = np.argsort(-c, axis=1, kind="stable")
        rs = np.take_along_axis(r, order, 1)
        out = out + (np.cumsum(rs, 1) / np.arange(1, rs.shape[1] + 1)).mean(1)
    return out / len(seeds)


def paired_aurc(x, y, n=4000, seed=0):
    m = len(x["brier"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m, (n, m))
    full = np.arange(m)[None]
    d = aurc_boot(x["seeds"], full)[0] - aurc_boot(y["seeds"], full)[0]
    boot = aurc_boot(x["seeds"], idx) - aurc_boot(y["seeds"], idx)
    return d, np.percentile(boot, 2.5), np.percentile(boot, 97.5), 2 * min((boot >= 0).mean(), (boot <= 0).mean())


def paired(a, b, n=20000, seed=0):
    """Mean of a - b, 95% bootstrap CI over items, two-sided sign-flip permutation p-value."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    flips = (rng.integers(0, 2, (n, len(d))) * 2 - 1) @ d / len(d)
    return d.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5), (np.abs(flips) >= abs(d.mean())).mean()
