"""Widen the β scan for the nocut a=0.30 snapshot.

The clean σ²≤0.15 fit gives β(ln σ) = -0.60 + 0.60·ln σ. Extrapolating
to the nocut a=0.30 point (σ²≈0.69, ln σ ≈ -0.19) predicts β ≈ -0.72.
Our earlier scan over β∈[-4, +1] returned β_best = -4.0 (boundary hit)
with scan-reported S₃_cas(β=-4) still below PM's S₃=+1.25. Two
possibilities:
  (a) the cascade genuinely can't reach PM's S₃ at this σ² — a=0.30 is
      outside the linear-β regime (high-σ saturation);
  (b) the fit was forced to the boundary by weighting on σ²_J, which
      pulled β toward -4 even though S₃ gets worse.

Probe: evaluate the cascade over β ∈ [-8, +1] and report σ²_J, S₃, κ₄.
Plot the (β → S₃) curve and mark PM's S₃. If S₃_cas(β) asymptotes
below +1.25, the point is genuinely unreachable and should be excluded.
"""

import json, time

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax, jax.numpy as jnp

from discodj.analysis import load_sim, deformation_gradient_jacobian
from discodj.analysis.spectral_slope import reconstruct_pk
from discodj.cascade import fem_cascade_v3


N_CAS, N_STEPS, BOXSIZE = 32, 10, 250.0
SEEDS = 3
BETA_GRID = np.linspace(-8.0, 1.0, 19)


def Jmom(J):
    a = np.asarray(J).ravel()
    m = a.mean(); d = a - m
    v = float(np.mean(d * d))
    s3 = float(np.mean(d**3) / max(v, 1e-30)**1.5)
    k4 = float(np.mean(d**4) / max(v, 1e-30)**2 - 3.0)
    return float(m), v, s3, k4


def main():
    snap = load_sim(256, "nocutoff", a=0.30)
    G_pm, J_pm = deformation_gradient_jacobian(snap)
    s2 = float(jnp.var(jnp.einsum("...ii->...", G_pm)))
    m_pm, v_pm, S3_pm, k4_pm = Jmom(J_pm)
    print(f"nocut a=0.30:  σ²={s2:.3f}  PM S₃={S3_pm:+.3f}  κ₄={k4_pm:+.3f}  "
          f"σ²_J={v_pm:.4f}")

    k_np, Pk_np = reconstruct_pk(snap)
    pk_k, pk_Pk = jnp.asarray(k_np), jnp.asarray(Pk_np)

    rows = []
    print(f"\n{'β':>6} | {'σ²_J':>7} {'S₃':>7} {'κ₄':>7}")
    for beta in BETA_GRID:
        vs, S3s, k4s = [], [], []
        t0 = time.time()
        for seed in range(SEEDS):
            key = jax.random.PRNGKey(2000 + seed)
            r = fem_cascade_v3(sigma2=s2, n_steps=N_STEPS, N=N_CAS,
                               boxsize=BOXSIZE, key=key,
                               g_T=0.0, alpha_nl=0.0, beta_nl=float(beta),
                               pk_k=pk_k, pk_Pk=pk_Pk,
                               method="midpoint", use_abs_J=True)
            jax.block_until_ready(r.J)
            _, v, s3, k4 = Jmom(r.J)
            vs.append(v); S3s.append(s3); k4s.append(k4)
        v, s3, k4 = np.mean(vs), np.mean(S3s), np.mean(k4s)
        rows.append(dict(beta=float(beta), v=float(v), S3=float(s3), k4=float(k4)))
        print(f"{beta:+6.2f} | {v:7.4f} {s3:+7.3f} {k4:+7.3f}   ({time.time()-t0:.0f}s)")

    # Plot S₃(β) and σ²_J(β) with PM markers
    betas = np.array([r["beta"] for r in rows])
    S3s   = np.array([r["S3"]   for r in rows])
    vs    = np.array([r["v"]    for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(betas, S3s, "o-", label="cascade")
    axes[0].axhline(S3_pm, ls="--", color="red", label=f"PM S₃={S3_pm:+.2f}")
    axes[0].set_xlabel(r"$\beta_{nl}$"); axes[0].set_ylabel(r"$S_3$")
    axes[0].set_title("S₃ vs β — nocut a=0.30"); axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(betas, vs, "o-", label="cascade")
    axes[1].axhline(v_pm, ls="--", color="red", label=f"PM σ²_J={v_pm:.3f}")
    axes[1].set_xlabel(r"$\beta_{nl}$"); axes[1].set_ylabel(r"$\sigma^2_J$")
    axes[1].set_title("σ²_J vs β — nocut a=0.30"); axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("docs/probe_a030_nocut.png", dpi=120)
    print(f"\nSaved docs/probe_a030_nocut.png")

    # Dump JSON
    with open("docs/probe_a030_nocut.json", "w") as f:
        json.dump(dict(sigma2=s2, S3_PM=S3_pm, k4_PM=k4_pm, v_PM=v_pm,
                       scan=rows), f, indent=2)
    print("Saved docs/probe_a030_nocut.json")

    # Diagnostic: is PM S₃ inside the scan?
    S3_min, S3_max = S3s.min(), S3s.max()
    print(f"\nCascade S₃ range over β∈[{betas.min()}, {betas.max()}]: "
          f"[{S3_min:+.3f}, {S3_max:+.3f}]")
    if S3_pm > S3_max:
        print(f"  → PM S₃={S3_pm:+.3f} is ABOVE cascade max → truly unreachable.")
    elif S3_pm < S3_min:
        print(f"  → PM S₃={S3_pm:+.3f} is BELOW cascade min → truly unreachable.")
    else:
        # find crossing β
        # S₃ tends to decrease with β, so find where S₃_cas = S3_pm
        idx = np.argmin(np.abs(S3s - S3_pm))
        print(f"  → PM S₃ reachable near β ≈ {betas[idx]:+.2f} "
              f"(cascade S₃={S3s[idx]:+.3f})")


if __name__ == "__main__":
    main()
