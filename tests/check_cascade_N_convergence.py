"""N-convergence check for the cascade moments.

The σ²_J mismatch (cascade ~1-3% below PM at low σ) biases the joint fit.
Check whether this is a finite-N_CAS artifact by evaluating (σ²_J, S₃) at
fixed β with N_CAS ∈ {32, 64, 128}. If the moments stabilise, the bias is
physical; if they drift, we need larger N.

Test: a=0.03 (σ²=0.015) and a=0.10 (σ²=0.102), each at its β_best(S₃-only)
from the refined fit. 3 seeds per (N, snap).
"""

import time
import numpy as np
import jax, jax.numpy as jnp

from discodj.analysis import load_sim, deformation_gradient_jacobian
from discodj.analysis.spectral_slope import reconstruct_pk
from discodj.cascade import fem_cascade_v3


N_STEPS, BOXSIZE, SEEDS = 10, 250.0, 3


def Jmom(J):
    a = np.asarray(J).ravel()
    m = a.mean(); d = a - m
    v = float(np.mean(d * d))
    s3 = float(np.mean(d**3) / max(v, 1e-30)**1.5)
    return float(m), v, s3


# snap, β target
CASES = [
    (0.03, -1.774),
    (0.10, -1.232),
]
NS = [32, 64, 128]


def main():
    for a_pm, beta in CASES:
        snap = load_sim(256, "nocutoff", a=a_pm)
        G_pm, J_pm = deformation_gradient_jacobian(snap)
        s2 = float(jnp.var(jnp.einsum("...ii->...", G_pm)))
        _, v_pm, S3_pm = Jmom(J_pm)
        k_np, Pk_np = reconstruct_pk(snap)
        pk_k, pk_Pk = jnp.asarray(k_np), jnp.asarray(Pk_np)

        print(f"\n=== a={a_pm:.2f}  σ²={s2:.4f}  β={beta:+.3f}   "
              f"PM: σ²_J={v_pm:.4f} S₃={S3_pm:+.3f} ===")
        print(f"{'N':>5} | {'σ²_J':>8} ({'Δ/PM':>6}) | {'S₃':>8} ({'ΔS₃':>7}) | {'time':>6}")
        for N in NS:
            t0 = time.time()
            vs, S3s = [], []
            for s in range(SEEDS):
                r = fem_cascade_v3(sigma2=s2, n_steps=N_STEPS, N=N,
                                   boxsize=BOXSIZE,
                                   key=jax.random.PRNGKey(4500 + s),
                                   g_T=0.0, alpha_nl=0.0, beta_nl=float(beta),
                                   pk_k=pk_k, pk_Pk=pk_Pk,
                                   method="midpoint", use_abs_J=True)
                jax.block_until_ready(r.J)
                _, v, s3 = Jmom(r.J)
                vs.append(v); S3s.append(s3)
            v, s3 = np.mean(vs), np.mean(S3s)
            dt = time.time() - t0
            print(f"{N:>5} | {v:8.4f} ({(v/v_pm-1)*100:+5.1f}%) | "
                  f"{s3:+8.3f} ({s3-S3_pm:+7.3f}) | {dt:5.0f}s")


if __name__ == "__main__":
    main()
