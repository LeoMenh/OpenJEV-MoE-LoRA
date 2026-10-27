"""Exact-enumeration checks of the estimators in openjev_rlcd/estimators.py.

Tabular stand-in for a reasoning model: rationale r in {0..R-1} with
pi(r) = softmax(a); answer distribution after r: u(r) = softmax(B[r]).
Decision distribution p = sum_r pi(r) u(r). Enumerate all M-tuples of
rationales and all outcomes Y ~ q, and compare the expected surrogate
gradient (w.r.t. a and B) with the gradient of ||p||^2 - 2 p.q."""
import itertools

import torch

from openjev_rlcd.estimators import lambda_surrogate, rb_surrogate


def check(r_n, k, m, seed):
    torch.manual_seed(seed)
    a0 = torch.randn(r_n, dtype=torch.float64)
    b0 = torch.randn(r_n, k, dtype=torch.float64)
    q = torch.distributions.Dirichlet(torch.ones(k, dtype=torch.float64)).sample()

    a, b = a0.clone().requires_grad_(True), b0.clone().requires_grad_(True)
    p = (a.softmax(0)[:, None] * b.softmax(1)).sum(0)
    (p.pow(2).sum() - 2 * (p * q).sum()).backward()
    target = torch.cat([a.grad, b.grad.flatten()])

    pi = a0.softmax(0)
    exp = torch.zeros_like(target)
    for tup in itertools.product(range(r_n), repeat=m):
        w = torch.prod(pi[list(tup)])
        for y in range(k):
            a, b = a0.clone().requires_grad_(True), b0.clone().requires_grad_(True)
            u = b.softmax(1)[list(tup)]
            logp_r = a.log_softmax(0)[list(tup)]
            rb_surrogate(u, logp_r, y).backward()
            exp += w * q[y] * torch.cat([a.grad, b.grad.flatten()])
    return (exp - target).abs().max().item()


def check_regularized(r_n=3, k=3, m=3, seed=7, beta=0.3):
    """extra_local = -(beta/M) * log(pi/pi_ref)(r_i) - (1/M) * cost[r_i]; target objective
    ||p||^2 - 2 p.q + beta * KL(pi || pi_ref) + sum_r pi(r) cost(r)  (as a loss)."""
    torch.manual_seed(seed)
    a0 = torch.randn(r_n, dtype=torch.float64)
    b0 = torch.randn(r_n, k, dtype=torch.float64)
    ref = torch.randn(r_n, dtype=torch.float64).log_softmax(0)
    cost = torch.rand(r_n, dtype=torch.float64)
    q = torch.distributions.Dirichlet(torch.ones(k, dtype=torch.float64)).sample()
    a, b = a0.clone().requires_grad_(True), b0.clone().requires_grad_(True)
    pi = a.softmax(0)
    p = (pi[:, None] * b.softmax(1)).sum(0)
    (p.pow(2).sum() - 2 * (p * q).sum() + beta * (pi * (pi.log() - ref)).sum() + (pi * cost).sum()).backward()
    target = torch.cat([a.grad, b.grad.flatten()])
    w_pi = a0.softmax(0)
    exp = torch.zeros_like(target)
    for tup in itertools.product(range(r_n), repeat=m):
        w = torch.prod(w_pi[list(tup)])
        for y in range(k):
            a, b = a0.clone().requires_grad_(True), b0.clone().requires_grad_(True)
            logp_r = a.log_softmax(0)[list(tup)]
            extra = -(beta / m) * (logp_r.detach() - ref[list(tup)]) - cost[list(tup)] / m
            rb_surrogate(b.softmax(1)[list(tup)], logp_r, y, extra_local=extra).backward()
            exp += w * q[y] * torch.cat([a.grad, b.grad.flatten()])
    return (exp - target).abs().max().item()


def check_lambda(lam, r_n=3, k=3, m=3, seed=11, beta=0.2):
    """lambda_surrogate (with KL + cost shaping) vs the exact gradient of
    -[(1 - lam) E_r J(u_r) + lam J(p)] + beta KL(pi || pi_ref) + E_r cost, and the identity
    J(p) = E_r J(u_r) + Var_r(u)."""
    torch.manual_seed(seed)
    a0 = torch.randn(r_n, dtype=torch.float64)
    b0 = torch.randn(r_n, k, dtype=torch.float64)
    ref = torch.randn(r_n, dtype=torch.float64).log_softmax(0)
    cost = torch.rand(r_n, dtype=torch.float64)
    q = torch.distributions.Dirichlet(torch.ones(k, dtype=torch.float64)).sample()
    a, b = a0.clone().requires_grad_(True), b0.clone().requires_grad_(True)
    pi, u = a.softmax(0), b.softmax(1)
    p = pi @ u
    j_p = 2 * p @ q - p @ p
    j_single = pi @ (2 * u @ q - (u * u).sum(1))
    var = pi @ ((u - p) ** 2).sum(1)
    assert abs(j_p - j_single - var).item() < 1e-12
    (-((1 - lam) * j_single + lam * j_p) + beta * (pi * (pi.log() - ref)).sum() + (pi * cost).sum()).backward()
    target = torch.cat([a.grad, b.grad.flatten()])
    w_pi = a0.softmax(0)
    exp = torch.zeros_like(target)
    for tup in itertools.product(range(r_n), repeat=m):
        w = torch.prod(w_pi[list(tup)])
        for y in range(k):
            a, b = a0.clone().requires_grad_(True), b0.clone().requires_grad_(True)
            logp_r = a.log_softmax(0)[list(tup)]
            extra = -(beta / m) * (logp_r.detach() - ref[list(tup)]) - cost[list(tup)] / m
            lambda_surrogate(b.softmax(1)[list(tup)], logp_r, y, lam, extra_local=extra).backward()
            exp += w * q[y] * torch.cat([a.grad, b.grad.flatten()])
    return (exp - target).abs().max().item()


def main():
    for lam in [0.0, 0.5, 1.0]:
        err = check_lambda(lam)
        assert err < 1e-12, (lam, err)
        print(f"lambda={lam}: J_lam = (1-lam) E_r J(u_r) + lam J(p), KL + cost shaping: max|diff| = {err:.1e}")
    print("PASS: J(p) = E_r J(u_r) + Var_r(u); per-rationale and interpolated estimators exactly unbiased")
    err = check_regularized()
    assert err < 1e-12, err
    print(f"PASS: KL + per-rationale cost regularization unbiased for the regularized objective (max|diff|={err:.1e})")
    worst = 0.0
    for r_n, k, m, seed in [(3, 3, 3, 0), (2, 3, 4, 1), (3, 2, 3, 2), (4, 3, 3, 3), (3, 4, 3, 4)]:
        err = check(r_n, k, m, seed)
        worst = max(worst, err)
        print(f"R={r_n} K={k} M={m}: RB estimator vs grad E[Brier(p)]  max|diff| = {err:.2e}")
    assert worst < 1e-12
    print("PASS: Rao-Blackwellized estimator is exactly unbiased (rationale policy AND answer head)")


if __name__ == "__main__":
    main()


def test_interpolated_objective_unbiased():
    for lam in [0.0, 0.5, 1.0]:
        assert check_lambda(lam) < 1e-12


def test_shaping_unbiased():
    assert check_regularized() < 1e-12


def test_mixture_estimator_unbiased():
    for r_n, k, m, seed in [(3, 3, 3, 0), (2, 3, 4, 1), (3, 2, 3, 2), (4, 3, 3, 3), (3, 4, 3, 4)]:
        assert check(r_n, k, m, seed) < 1e-12
