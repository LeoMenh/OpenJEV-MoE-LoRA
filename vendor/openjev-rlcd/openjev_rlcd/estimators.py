"""Rao-Blackwellized calibrated objective for reasoning models.

Sample only the rationale r_i ~ pi(r | x); after each rationale, READ the
model's explicit answer distribution u_i = pi(. | x, r_i) over the K labels
instead of sampling an answer from it. The model's decision distribution is
the mixture p = E_r[u(r)], and the objective is the (negative) Brier score of p
against the outcome Y ~ q:
    J = 2 p.q - ||p||^2   (maximized at p = q).
Unbiased estimate from M independent rationales:
    J_hat = (2/M) sum_i u_i[Y] - sum_{i != j} u_i . u_j / (M (M - 1)).
Gradient = pathwise part (d J_hat / d u, exact, no sampling noise)
         + score-function part for the rationales only:
    sum_i (local_i - b_i) grad log pi(r_i),
    local_i = (2/M) u_i[Y] - (2 / (M (M - 1))) sum_{j != i} u_i . u_j,
    b_i     = (2/M) mean_{j != i} u_j[Y] - (2 / (M (M - 1))) sum_{j != i} pbar_{-ij} . u_j,
    pbar_{-ij} = mean of u_k over k != i, j        (b_i uses only other rationales).
Special cases: no rationale (u_i identical, deterministic) -> exactly direct
Brier on an explicit distribution (the System-One scorer); one-hot u_i (answer
sampled) -> the sample-only estimator of the mixture objective.
"""
import torch


def rb_surrogate(u, logp_r, y, extra_local=None):
    """u: [M, K] answer distributions (differentiable); logp_r: [M] rationale log-probs
    (differentiable); y: outcome index. Returns a scalar LOSS whose expected gradient
    equals grad of E[Brier(p, Y)] = grad (||p||^2 - 2 p.q) up to a constant.

    extra_local: optional [M] per-rationale reward terms that depend only on r_i
    (e.g. -beta/M * log(pi/pi_ref)(r_i) for a KL penalty, -lam/M * 1[length limit]).
    They enter the score-function part with a leave-one-out baseline, so the
    gradient stays unbiased for the regularized objective E[J + sum_i extra_i]."""
    m = u.shape[0]
    if m < 3:
        raise ValueError("needs M >= 3 rationales")
    pair = 1.0 / (m * (m - 1))
    gram = u @ u.T
    off = gram.sum() - gram.diagonal().sum()
    j_hat = (2.0 / m) * u[:, y].sum() - pair * off                      # pathwise part
    ud = u.detach()
    g = ud @ ud.T
    local = (2.0 / m) * ud[:, y] - 2 * pair * (g.sum(1) - g.diagonal())
    adv = torch.empty(m, dtype=u.dtype, device=u.device)
    total = ud.sum(0)
    for i in range(m):
        others = [j for j in range(m) if j != i]
        hit_base = (2.0 / m) * ud[others, y].mean()
        coll = 0.0
        for j in others:
            pbar = (total - ud[i] - ud[j]) / (m - 2)
            coll = coll + pbar @ ud[j]
        adv[i] = local[i] - (hit_base - 2 * pair * coll)
    if extra_local is not None:
        e = extra_local.detach().to(adv.dtype)
        adv = adv + e - (e.sum() - e) / (m - 1)
    return -(j_hat + (adv.detach() * logp_r).sum())


def single_surrogate(u, logp_r, y, extra_local=None):
    """Per-rationale objective: score every rationale's OWN answer distribution,
        J_single = E_r[2 u(r)_Y - ||u(r)||^2] = J(p) - Var_r(u),   Var_r(u) = E_r ||u(r) - p||^2.
    The mixture objective J(p) (rb_surrogate) therefore pays a bonus Var_r(u) for rationales
    that DISAGREE, whether or not the disagreement is grounded; J_single removes it. Because the
    outcome Y is independent of the model's own rationale given x, the optimum of J_single is
    u(r) = q(x) for every rationale: one query is calibrated, and so is p.
    Unbiased gradient: pathwise d/du of mean_i R_i plus REINFORCE on the rationales with a
    leave-one-out baseline, R_i = 2 u_i[Y] - ||u_i||^2. Same arguments/return as rb_surrogate."""
    m = u.shape[0]
    if m < 2:
        raise ValueError("needs M >= 2 rationales")
    r = 2 * u[:, y] - (u * u).sum(1)
    local = r.detach() / m
    if extra_local is not None:
        local = local + extra_local.detach().to(local.dtype)
    adv = local - (local.sum() - local) / (m - 1)
    return -(r.mean() + (adv * logp_r).sum())


def lambda_surrogate(u, logp_r, y, lam, extra_local=None):
    """J_lam = J_single + lam * Var_r(u) = (1 - lam) J_single + lam J(p): lam=1 is the mixture
    objective, lam=0 the per-rationale one; for any lam < 1 the unique optimum is u(r) = q."""
    if lam == 1.0:
        return rb_surrogate(u, logp_r, y, extra_local)
    if lam == 0.0:
        return single_surrogate(u, logp_r, y, extra_local)
    return lam * rb_surrogate(u, logp_r, y, extra_local) + (1 - lam) * single_surrogate(u, logp_r, y, extra_local)
