"""Convergence test for cascade v5 in n_steps.

v5 evaluates 2LPT/3LPT at the *accumulated* Gaussian field rather than
per-step. At g_T = g_S = 0 the cascade is equivalent to:

    G_linear = Σ_i dG₁_i   (n_steps independent Gaussian increments)
    G_total  = G_linear + β·F₂(G_linear) + γ·F₃(G_linear)

so the moments should be **n_steps-independent** by construction
(the sum of n Gaussian increments with total variance σ² has the same
distribution regardless of n). Any drift must come from seed noise.

Compares to v3 (which fails this test — see check_cascade_nsteps_convergence).
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

from discodj.analysis import load_sim, deformation_gradient_jacobian
from discodj.analysis.spectral_slope import reconstruct_pk
from discodj.cascade import fem_cascade_v5


N_CAS, BOXSIZE = 64, 250.0
SEEDS = 3
N_STEPS_LIST = [2, 5, 10, 20, 40]


def Jmom(J):
    a = np.asarray(J).ravel()
    m = a.mean(); d = a - m
    v = float(np.mean(d * d))
    s3 = float(np.mean(d**3) / max(v, 1e-30)**1.5)
    k4 = float(np.mean(d**4) / max(v, 1e-30)**2 - 3.0)
    return float(m), v, s3, k4


def avg(sigma2, pk_k, pk_Pk, beta, n_steps):
    vs, S3s, k4s = [], [], []
    for s in range(SEEDS):
        r = fem_cascade_v5(sigma2=sigma2, n_steps=n_steps, N=N_CAS,
                           boxsize=BOXSIZE, key=jax.random.PRNGKey(8300 + s),
                           g_T=0.0, alpha_nl=0.0, beta_nl=float(beta), gamma_nl=0.0,
                           pk_k=pk_k, pk_Pk=pk_Pk,
                           method="midpoint", use_abs_J=True)
        jax.block_until_ready(r.J)
        _, v, s3, k4 = Jmom(r.J)
        vs.append(v); S3s.append(s3); k4s.append(k4)
    return float(np.mean(vs)), float(np.mean(S3s)), float(np.mean(k4s))


def main():
    for a_pm in (0.05, 0.07, 0.10):
        snap = load_sim(256, "nocutoff", a=a_pm)
        G_pm, _ = deformation_gradient_jacobian(snap)
        s2 = float(jnp.var(jnp.einsum("...ii->...", G_pm)))
        k_np, Pk_np = reconstruct_pk(snap)
        pk_k, pk_Pk = jnp.asarray(k_np), jnp.asarray(Pk_np)
        beta_35 = 0.6 * (0.5 * np.log(s2) - 1.0)

        print(f"\n=== a={a_pm:.2f}  σ²={s2:.4f}  σ={np.sqrt(s2):.3f} ===")
        for beta, tag in [(0.0, "β=0 (Gaussian)"),
                          (beta_35, f"β={beta_35:+.3f} (3/5 law)")]:
            print(f"  {tag}")
            print(f"    {'n_steps':>8} | {'σ²_J':>8} {'S₃':>8} {'κ₄':>8} | "
                  f"{'Δσ²/σ²':>8} {'ΔS₃':>8} | time")
            ref = None
            for n in N_STEPS_LIST:
                t0 = time.time()
                v, s3, k4 = avg(s2, pk_k, pk_Pk, beta, n)
                dt = time.time() - t0
                if ref is None:
                    ref = (v, s3, k4); dv = ds3 = 0.0
                else:
                    dv = (v - ref[0]) / ref[0] * 100
                    ds3 = s3 - ref[1]
                print(f"    {n:>8} | {v:8.4f} {s3:+8.3f} {k4:+8.3f} | "
                      f"{dv:+7.2f}% {ds3:+8.3f} | {dt:.1f}s")


if __name__ == "__main__":
    main()
