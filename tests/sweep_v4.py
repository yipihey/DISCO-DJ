"""Scan cascade v4 (CLPT-style increment) against PM targets.

Reports ⟨J⟩, σ²_J, S₃(J), and κ₄(J) (excess kurtosis) so we can see
where the 3LPT cubic coupling ``γ_nl`` earns its keep — the quadratic
β_nl predominantly shifts S₃, while γ should pull the J-PDF tails
(kurtosis) toward the PM target.
"""

import numpy as np
import jax

from discodj.analysis.sim_store import load_sim, deformation_gradient_jacobian
from discodj.cascade import fem_cascade_v4


def J_moments(J):
    a = np.asarray(J).ravel()
    m = a.mean()
    d = a - m
    v = np.mean(d * d)
    S3 = np.mean(d ** 3) / max(v, 1e-30) ** 1.5
    k4 = np.mean(d ** 4) / max(v, 1e-30) ** 2 - 3.0
    return m, v, S3, k4


def cascade_run(sigma2, g_T, alpha, beta, gamma, N=32, n_steps=10, seeds=3):
    vs, s3s, k4s = [], [], []
    for seed in range(seeds):
        key = jax.random.PRNGKey(42 + seed)
        try:
            r = fem_cascade_v4(sigma2=sigma2, n_steps=n_steps, N=N,
                               boxsize=250.0, key=key,
                               g_T=g_T, alpha_nl=alpha,
                               beta_nl=beta, gamma_nl=gamma,
                               method="midpoint", use_abs_J=True,
                               cg_tol=1e-8, cg_maxiter=15)
            jax.block_until_ready(r.J)
            _, v, s3, k4 = J_moments(r.J)
        except Exception:
            v, s3, k4 = np.nan, np.nan, np.nan
        vs.append(v); s3s.append(s3); k4s.append(k4)
    return np.mean(vs), np.mean(s3s), np.mean(k4s)


def main():
    targets = [
        (256, "nocutoff", 0.1, "T2 (σ²≈0.10, no cutoff)"),
        (256, "cutoff", 0.3, "T1 (σ²≈0.16, cutoff)"),
    ]
    for Nsim, cutoff, a, label in targets:
        snap = load_sim(Nsim, cutoff, a=a)
        G_pm, J_pm = deformation_gradient_jacobian(snap)
        sigma2 = float(np.var(np.einsum("...ii->...", G_pm)))
        _, vJ_pm, S3_pm, k4_pm = J_moments(J_pm)
        print(f"\n=== {label}: σ²={sigma2:.4f}  σ²_J={vJ_pm:.4f}  "
              f"S₃={S3_pm:+.3f}  κ₄={k4_pm:+.3f}")
        print(f"{'g_T':>4} {'α':>5} {'β':>6} {'γ':>6} | "
              f"{'σ²_J':>7} {'S₃':>7} {'κ₄':>7} | "
              f"{'Δσ²_J':>7} {'ΔS₃':>7} {'Δκ₄':>7}")
        # Start from v3 best-fit: β≈-1.5 (T2) or -3.0 (T1), scan γ.
        base = (0.05, 0.0, -1.5) if "nocutoff" in cutoff else (0.15, 0.5, -3.0)
        gamma_grid = (-0.5, -0.2, 0.0, 0.238, 0.5, 1.0, 2.0)
        for gamma in gamma_grid:
            vJ, S3, k4 = cascade_run(sigma2, *base, gamma)
            print(f"{base[0]:4.2f} {base[1]:5.2f} {base[2]:6.2f} {gamma:+6.2f} | "
                  f"{vJ:7.4f} {S3:+7.3f} {k4:+7.3f} | "
                  f"{vJ - vJ_pm:+7.4f} {S3 - S3_pm:+7.3f} {k4 - k4_pm:+7.3f}")


if __name__ == "__main__":
    main()
