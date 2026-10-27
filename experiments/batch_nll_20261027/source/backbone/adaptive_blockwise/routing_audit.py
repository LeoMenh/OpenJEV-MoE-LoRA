"""Additive route statistics; zero-weight distributed padding is excluded."""
import math
import torch


def route_totals(probabilities, sample_weights):
    p = probabilities.detach().double()
    w = sample_weights.to(p.device).double()
    if p.ndim != 2 or w.shape != p.shape[:1]:
        raise ValueError("Route statistics require [batch, experts] and [batch] weights")
    mass = (p * w[:, None]).sum(0)
    hard = torch.nn.functional.one_hot(p.argmax(-1), p.shape[-1]).double()
    counts = (hard * w[:, None]).sum(0)
    entropy = -(p * p.clamp_min(1e-30).log()).sum(-1)
    return torch.cat((mass, counts, (entropy * w).sum().view(1),
                      (p.max(-1).values * w).sum().view(1), w.sum().view(1)))


def summarize_routes(totals):
    values = totals.detach().cpu().tolist()
    k = (len(values) - 3) // 2
    n = values[-1]
    if n <= 0 or k < 1:
        raise ValueError("No real samples in routing audit")
    return {"num_samples": int(n), "mean_mixture_weight": [v / n for v in values[:k]],
            "argmax_counts": values[k:2*k], "mean_entropy_nats": values[-3] / n,
            "normalized_mean_entropy": values[-3] / n / math.log(k) if k > 1 else 0.,
            "mean_max_weight": values[-2] / n}
