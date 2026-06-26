"""Scan cascade v3 across (alpha_nl, beta_nl, g_T) against two PM targets.

Targets:
  T1: N=256_cutoff, a=0.30 → σ²_J=0.151, S₃=0.44 (mild nonlin, resolved)
  T2: N=256_cutoff, a=0.70 → σ²_J=0.653, S₃=0.97 (stronger nonlin, shell x)
"""

import numpy as np
import jax

from discodj.analysis.sim_store import load_sim, deformation_gradient_jacobian
from discodj.cascade import fem_cascade_v3


def jmoments(J):
    a = np.asarray(J).ravel()
    m = a.mean(); v = a.var()
    S3 = np.mean((a - m) ** 3) / max(v, 1e-30) ** 1.5
    return m, v, S3


def one_target(N_sim, cutoff_mode, a, label):
    snap = load_sim(N_sim, cutoff_mode, a=a)
    G_pm, J_pm = deformation_gradient_jacobian(snap)
    s2 = float(np.var(np.einsum("...ii->...", G_pm)))
    mJ, vJ, S3 = jmoments(J_pm)
    print(f"\n=== {label}: {N_sim}_{cutoff_mode} a={snap.a:.2f}  "
          f"σ²_trG={s2:.3f}  σ²_J={vJ:.3f}  S₃={S3:+.3f}")
    return s2, vJ, S3


def run_cascade(sigma2, g_T, alpha_nl, beta_nl, N=32, n_steps=10, seeds=3):
    vJs, S3s = [], []
    for seed in range(seeds):
        key = jax.random.PRNGKey(42 + seed)
        r = fem_cascade_v3(sigma2=sigma2, n_steps=n_steps, N=N, boxsize=250.0,
                           key=key, g_T=g_T, alpha_nl=alpha_nl, beta_nl=beta_nl,
                           method="midpoint", use_abs_J=True,
                           cg_tol=1e-8, cg_maxiter=15)
        jax.block_until_ready(r.J)
        _, v, S3 = jmoments(r.J)
        vJs.append(v); S3s.append(S3)
    return np.mean(vJs), np.mean(S3s)


def main():
    targets = [
        one_target(256, "cutoff", 0.3, "T1"),
        one_target(256, "cutoff", 0.7, "T2"),
    ]
    for (s2_pm, vJ_pm, S3_pm), label in zip(targets, ("T1", "T2")):
        print(f"\n--- cascade scan for {label} σ²_trG={s2_pm:.3f}  target σ²_J={vJ_pm:.3f} S₃={S3_pm:+.3f} ---")
        print(f"{'g_T':>5} {'α_nl':>6} {'β_nl':>6} | {'σ²_J':>8} {'S₃':>7} | {'Δσ²_J':>8} {'ΔS₃':>8}")
        for g_T in (0.0, 0.05):
            for alpha in (0.0, 0.5, 17/21):
                for beta in (0.0, 3/7):
                    vJ, S3 = run_cascade(s2_pm, g_T, alpha, beta)
                    dv = vJ - vJ_pm; dS3 = S3 - S3_pm
                    print(f"{g_T:5.2f} {alpha:6.3f} {beta:6.3f} | "
                          f"{vJ:8.4f} {S3:+7.3f} | {dv:+8.4f} {dS3:+8.3f}")


if __name__ == "__main__":
    main()
