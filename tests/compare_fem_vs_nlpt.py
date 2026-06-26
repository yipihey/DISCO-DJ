"""FEM cascade vs DISCO-DJ nLPT (10th order) comparison at matched σ².

Runs both solvers, matches target variance ``σ²(tr G)``, and prints the
four diagnostic statistics from the task spec: ``<J>``, ``σ²_J``, ``S₃``
(sample skewness of δ = 1/J − 1), and ``ξ̄ = ⟨1/J⟩ − 1``.

Reference bands from §9 of the RG-flow paper:

- σ²=0.1:   S₃ ≈ 1.4–1.5, σ²_J ≈ 0.097–0.100, ξ̄ ≈ 0.11–0.12.
- σ²=0.3:   S₃ ≈ 1.4–1.5, σ²_J ≈ 0.29–0.30,   ξ̄ ≈ 0.7–0.9.

Usage::

    python tests/compare_fem_vs_nlpt.py --N 64 --n_steps 15 --g2c 0.0 0.05 0.1 0.2
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from discodj import DiscoDJ
from discodj.cascade.fem_cascade import fem_cascade


def stats_from_J(J):
    J = np.asarray(J).ravel()
    delta = 1.0 / J - 1.0
    return {
        "mean_J": float(J.mean()),
        "var_J": float(J.var()),
        "xi_bar": float(delta.mean()),
        "S3": float((delta**3).mean() / (delta.var() ** 1.5 + 1e-30)),
    }


def stats_from_G(G):
    """Add the actually-realised Var[tr G] to the summary."""
    F = jnp.eye(3) + G
    J = jnp.linalg.det(F)
    trG = jnp.einsum("...ii->...", G)
    s = stats_from_J(J)
    s["sigma2_trG"] = float(jnp.var(trG))
    return s


# ------------------------------------------------------------------- 10LPT


def run_nlpt(N: int, boxsize: float, a: float, n_order: int, seed: int):
    """Run nLPT at scale factor ``a`` and return stats + timing."""
    t0 = time.time()
    dj = DiscoDJ(dim=3, res=N, boxsize=boxsize)
    dj = dj.with_timetables()
    dj = dj.with_linear_ps()
    dj = dj.with_ics(seed=seed, white_noise_space="fourier")
    dj = dj.with_lpt(n_order=n_order)
    psi = dj.evaluate_lpt_psi_at_a(a=a, n_order=n_order)
    J, _, dpsi = dj.evaluate_jacobian_from_psi(psi, only_det=False)
    G = dpsi - jnp.eye(3)[None, None, None, :, :]
    jax.block_until_ready(J)
    dt = time.time() - t0
    s = stats_from_G(G)
    s["time"] = dt
    return s


def scan_a_for_sigma2(N: int, boxsize: float, target_sigma2: float,
                     n_order: int, seed: int, tol: float = 0.02) -> float:
    """Binary-search the scale factor ``a`` that gives ``Var[tr G] ≈ target``.

    The 1LPT trace variance scales like ``D(a)²``; higher-order corrections
    shift this a bit. We bracket on a ∈ [0.01, 0.5] and bisect until the
    relative error is below ``tol`` or we hit 12 iterations.
    """
    # One-time setup so we don't rebuild the pipeline for every ``a``.
    dj = DiscoDJ(dim=3, res=N, boxsize=boxsize)
    dj = dj.with_timetables()
    dj = dj.with_linear_ps()
    dj = dj.with_ics(seed=seed, white_noise_space="fourier")
    dj = dj.with_lpt(n_order=n_order)

    def sigma2_at(a):
        psi = dj.evaluate_lpt_psi_at_a(a=a, n_order=n_order)
        J, _, dpsi = dj.evaluate_jacobian_from_psi(psi, only_det=False)
        G = dpsi - jnp.eye(3)[None, None, None, :, :]
        return float(jnp.var(jnp.einsum("...ii->...", G))), J, G

    lo, hi = 0.01, 0.5
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        s2, J, G = sigma2_at(mid)
        if abs(s2 - target_sigma2) / target_sigma2 < tol:
            return mid, s2, J, G
        if s2 < target_sigma2:
            lo = mid
        else:
            hi = mid
    return mid, s2, J, G


# ------------------------------------------------------------------- FEM


def run_fem(N: int, boxsize: float, sigma2: float, n_steps: int, g2c: float,
            seed: int):
    key = jax.random.PRNGKey(seed)
    t0 = time.time()
    res = fem_cascade(sigma2=sigma2, n_steps=n_steps, N=N, boxsize=boxsize,
                      key=key, g2c=g2c)
    jax.block_until_ready(res.J)
    dt = time.time() - t0
    s = stats_from_G(res.G)
    s["time"] = dt
    s["sigma2_trG"] = res.sigma2
    return s


# ------------------------------------------------------------------- driver


def print_row(label, s):
    print(f"  {label:20s}  σ²={s['sigma2_trG']:.4f}  <J>={s['mean_J']:.4f}  "
          f"var(J)={s['var_J']:.4f}  S3={s['S3']:7.3f}  ξ̄={s['xi_bar']:.4f}  "
          f"dt={s['time']:5.1f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--boxsize", type=float, default=250.0,
                   help="Mpc/h — Planck18EEBAOSN preset")
    p.add_argument("--n_steps", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sigma2", type=float, nargs="+", default=[0.1, 0.3])
    p.add_argument("--g2c", type=float, nargs="+", default=[0.0, 0.05, 0.1])
    p.add_argument("--n_order", type=int, default=10)
    args = p.parse_args()

    for target in args.sigma2:
        print("\n" + "=" * 72)
        print(f"Target Var[tr G] = {target}")
        print("=" * 72)

        # 10LPT reference at matched σ².
        print(f"\n--- {args.n_order}LPT (matched a via bisection) ---")
        t0 = time.time()
        a_match, s2_lpt, J_lpt, G_lpt = scan_a_for_sigma2(
            args.N, args.boxsize, target, args.n_order, args.seed
        )
        s_lpt = stats_from_G(G_lpt)
        s_lpt["time"] = time.time() - t0
        s_lpt["sigma2_trG"] = s2_lpt
        print_row(f"{args.n_order}LPT a={a_match:.3f}", s_lpt)

        # FEM cascade, scanning g2c.
        print(f"\n--- FEM cascade ({args.n_steps} steps) ---")
        for g2c in args.g2c:
            s_fem = run_fem(args.N, args.boxsize, target, args.n_steps, g2c,
                            args.seed)
            print_row(f"FEM g2c={g2c:.3f}", s_fem)


if __name__ == "__main__":
    main()
