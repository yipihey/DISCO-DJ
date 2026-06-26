"""Fit cascade model to PM simulation at matched σ²_trG or matched a.

Given a PM snapshot, run the cascade with various g_T (and optionally g_S),
match either σ²_J (primary) or a broader target and report the best
coefficients. Outputs a table and optionally saves the best-fit G for
downstream PDF comparison.

Usage::

    python tests/fit_cascade_to_pm.py --sim N256_cutoff --a 0.3 --g_T_scan 0.0 0.02 0.05 0.1 0.2
"""

import argparse
import time

import numpy as np
import jax
import jax.numpy as jnp

from discodj.analysis.sim_store import (
    load_sim, deformation_gradient_jacobian,
)
from discodj.cascade import fem_cascade_v2


def cascade_stats(J):
    """Report (σ²_trG, <J>, σ²_J, S₃(J))."""
    arr = np.asarray(J).ravel()
    m = arr.mean(); v = arr.var()
    S3 = np.mean((arr - m) ** 3) / max(v, 1e-30) ** 1.5
    return m, v, S3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sim", default="N256_cutoff")
    p.add_argument("--a", type=float, default=0.3)
    p.add_argument("--N_cascade", type=int, default=32)
    p.add_argument("--n_steps", type=int, default=20)
    p.add_argument("--method", default="midpoint",
                   choices=["euler", "midpoint"])
    p.add_argument("--use_abs_J", type=int, default=1)
    p.add_argument("--g_T_scan", type=float, nargs="+",
                   default=[0.0, 0.02, 0.05, 0.1, 0.2])
    p.add_argument("--g_S_ratio_scan", type=float, nargs="+", default=[1.0],
                   help="g_S / g_T ratios to scan (1.0 = single-coupling)")
    p.add_argument("--sigma2_mode", default="match_achieved",
                   choices=["match_a", "match_sigma2"],
                   help="match_a: cascade runs to same σ² as PM achieves. "
                        "match_sigma2: caller passes --sigma2 explicitly.")
    p.add_argument("--sigma2", type=float, default=None)
    p.add_argument("--seeds", type=int, default=3,
                   help="Number of cascade seeds to average over")
    p.add_argument("--L", type=float, default=250.0)
    p.add_argument("--root", default="sims")
    args = p.parse_args()

    # -------------- PM reference --------------
    tag = args.sim
    parts = tag.split("_", 1)
    N_sim = int(parts[0].lstrip("N"))
    cutoff_mode = parts[1]
    snap = load_sim(N_sim, cutoff_mode, a=args.a, root=args.root)
    print(f"PM reference: {tag}  a={snap.a:.3f}  N_sim={N_sim}  L={snap.L}")

    # Jacobian stats from the PM sim, for target comparison.
    G_pm, J_pm = deformation_gradient_jacobian(snap)
    trG_pm = np.einsum("...ii->...", G_pm)
    sigma2_pm = float(np.var(trG_pm))
    mJ_pm, vJ_pm, S3_pm = cascade_stats(J_pm)
    print(f"  σ²(trG)_PM={sigma2_pm:.4f}   <J>={mJ_pm:.4f}   σ²_J={vJ_pm:.4f}   S₃(J)={S3_pm:+.3f}")

    # -------------- Cascade scan --------------
    if args.sigma2 is not None:
        sigma2_target = args.sigma2
    else:
        sigma2_target = sigma2_pm

    print(f"\nCascade scan: N={args.N_cascade}, n_steps={args.n_steps}, "
          f"method={args.method}, use_abs_J={bool(args.use_abs_J)}, "
          f"σ²_target={sigma2_target:.4f}, {args.seeds} seeds\n")
    hdr = (f"{'g_T':>6} {'g_S':>6} | "
           f"{'σ²_trG':>8} {'σ²_J':>8} {'S₃(J)':>8} | "
           f"{'Δσ²_J':>8} {'ΔS₃':>8} {'dt(s)':>6}")
    print(hdr); print("-" * len(hdr))
    for g_T in args.g_T_scan:
        for ratio in args.g_S_ratio_scan:
            g_S = g_T * ratio
            # Ensemble average over seeds
            s2s, vs, S3s = [], [], []
            t0 = time.time()
            for seed in range(args.seeds):
                key = jax.random.PRNGKey(42 + seed)
                res = fem_cascade_v2(
                    sigma2=sigma2_target, n_steps=args.n_steps,
                    N=args.N_cascade, boxsize=args.L, key=key,
                    g_T=g_T, g_S=g_S,
                    method=args.method, use_abs_J=bool(args.use_abs_J),
                    cg_tol=1e-8, cg_maxiter=15,
                )
                jax.block_until_ready(res.J)
                _, v, S3 = cascade_stats(res.J)
                s2s.append(res.sigma2_trG); vs.append(v); S3s.append(S3)
            dt = time.time() - t0
            s2_m = np.mean(s2s); v_m = np.mean(vs); S3_m = np.mean(S3s)
            dv = v_m - vJ_pm
            dS3 = S3_m - S3_pm
            print(f"{g_T:6.3f} {g_S:6.3f} | "
                  f"{s2_m:8.4f} {v_m:8.4f} {S3_m:+8.3f} | "
                  f"{dv:+8.4f} {dS3:+8.3f} {dt:6.1f}")


if __name__ == "__main__":
    main()
