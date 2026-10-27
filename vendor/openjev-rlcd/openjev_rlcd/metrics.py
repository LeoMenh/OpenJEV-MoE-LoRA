"""Readout metrics and post-hoc temperature scaling, shared by every training script and the analysis.

Items carry q (the outcome distribution: one-hot for epistemic tasks, the human label distribution for
ChaosNLI) and counts (weights for drawing one outcome). Predictions are lists of probabilities.
"""
import math
import random

import torch


def fit_temperature(dev_logits, dev_items, seed):
    """Post-hoc temperature scaling as practitioners would do it: a small dev set with ONE
    label per item (fixed draw), T chosen by grid search on dev NLL."""
    ys = [random.Random(f"ts-{seed}-{it['uid']}").choices(range(len(it["counts"])), weights=it["counts"])[0]
          for it in dev_items]
    z = torch.stack(dev_logits)
    y = torch.tensor(ys)
    grid = torch.exp(torch.linspace(math.log(0.1), math.log(50.0), 311))  # wide: overconfident models need T > 8
    nll = torch.stack([torch.nn.functional.cross_entropy(z / t, y) for t in grid])
    return float(grid[nll.argmin()])


def selective_metrics(items, probs, budgets=(0.1, 0.2)):
    """Decision quality when the model decides alone on its most confident items. Risk of deciding
    item x alone = P(the outcome disagrees with the model's choice) = 1 - q[argmax p]; items are
    ranked by max p.
      ece          : 10-bin ECE of max p against that agreement probability
      aurc         : area under the risk-coverage curve (lower = better)
      cov@e        : label-free automation -- the largest top-ranked set whose PREDICTED error
                     mean(1 - max p) is <= e; risk@e is its realized risk (<= e when calibrated)
      oracle_cov@e : the largest top-ranked set whose REALIZED risk is <= e (ranking quality only)
    """
    rows = sorted(((max(p), 1 - it["q"][p.index(max(p))]) for it, p in zip(items, probs)), key=lambda r: -r[0])
    n = len(rows)
    bins = [[] for _ in range(10)]
    for c, r in rows:
        bins[min(9, int(c * 10))].append((c, 1 - r))
    out = dict(ece=sum(len(b) / n * abs(sum(c for c, _ in b) - sum(a for _, a in b)) / len(b) for b in bins if b))
    pred_cum = real_cum = aurc = 0.0
    sc = {e: (0, 0.0) for e in budgets}
    oracle = {e: 0 for e in budgets}
    for k, (c, r) in enumerate(rows, 1):
        pred_cum += 1 - c
        real_cum += r
        aurc += real_cum / k
        for e in budgets:
            if pred_cum / k <= e:
                sc[e] = (k, real_cum / k)
            if real_cum / k <= e:
                oracle[e] = k
    out["aurc"] = aurc / n
    for e in budgets:
        out[f"cov@{e}"], out[f"risk@{e}"], out[f"oracle_cov@{e}"] = sc[e][0] / n, sc[e][1], oracle[e] / n
    return out


def _kl(a, b):
    return sum(x * math.log(x / y) for x, y in zip(a, b) if x > 0)


def jsd(p, q):
    """Jensen-Shannon divergence in bits."""
    mid = [(a + b) / 2 for a, b in zip(p, q)]
    return (_kl(p, mid) + _kl(q, mid)) / (2 * math.log(2))


def readout_metrics(items, dists):
    div, l2, acc, agree, rows = [], [], [], [], []
    for it, p in zip(items, dists):
        q = it["q"]
        div.append(jsd(p, q))
        l2.append(sum((a - b) ** 2 for a, b in zip(p, q)))
        acc.append(int(p.index(max(p)) == q.index(max(q))))
        agree.append(sum(a * b for a, b in zip(p, q)))
        rows.append((max(p), q[p.index(max(p))]))
    bins = [[] for _ in range(10)]
    for c, y in rows:
        bins[min(9, int(c * 10))].append((c, y))
    n = len(items)
    return dict(jsd=sum(div) / n, l2=sum(l2) / n, majority_acc=sum(acc) / n, agreement_with_rater=sum(agree) / n,
                mode_prob_ece=sum(len(b) / n * abs(sum(c for c, _ in b) / len(b) - sum(y for _, y in b) / len(b)) for b in bins if b),
                mean_mode_prob=sum(c for c, _ in rows) / n, **selective_metrics(items, dists), per_item_jsd=div)
