"""Plot J-PDFs for extreme cosmology: cascade vs PM, β_nl fixed at −1.5.

Tests whether the cascade's running coupling β_nl(σ²) ≈ -1.5 (fit at
Planck cosmology) is universal — i.e., whether the same β_nl fits PM at
a wildly different cosmology, or needs to be refit.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

from discodj import DiscoDJ
from discodj.analysis.sim_store import SimSnapshot, deformation_gradient_jacobian
from discodj.cascade import fem_cascade_v3, pk_table_from_discodj

EXTREME_COSMO = dict(
    Omega_c=0.45, Omega_b=0.05, Omega_k=0.0, h=0.67,
    n_s=1.25, sigma8=1.0, w0=-1.0, wa=0.0,
)


def load_extreme(cutoff_mode, a, root="sims"):
    path = f"{root}/N256_{cutoff_mode}_extreme.npz"
    data = np.load(path, allow_pickle=True)
    meta = json.loads(str(data["meta"]))
    xs, vs, as_ = data["x"], data["v"], data["a"]
    idx = int(np.argmin(np.abs(as_ - a)))
    return SimSnapshot(x=xs[idx], v=vs[idx], a=float(as_[idx]),
                       N=meta["N"], L=meta["L"], meta=meta)


def J_hist(J, J_range=(-1.5, 4.0), bins=200):
    J = np.asarray(J).ravel()
    J = J[np.isfinite(J)]
    h, edges = np.histogram(J, bins=bins, range=J_range, density=True)
    return 0.5 * (edges[:-1] + edges[1:]), h


def main():
    L = 250.0
    N_cas = 32
    n_steps = 10
    beta_nl = -1.5     # our Planck-fit value; test if it still works
    seeds = 3

    # Pre-build extreme-cosmo P(k) for cascade shaping
    dj = (DiscoDJ(dim=3, res=N_cas, boxsize=L, cosmo=EXTREME_COSMO)
          .with_timetables().with_linear_ps())
    pk_k, pk_Pk = pk_table_from_discodj(dj)
    k_cut = 0.5 * np.pi * 64 / L
    pk_Pk_cut = pk_Pk * jnp.exp(-(pk_k / k_cut) ** 2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    cmap = plt.get_cmap("viridis")
    for col, cm in enumerate(["cutoff", "nocutoff"]):
        ax = axes[col]
        ax.set_yscale("log"); ax.set_xlim(-1.5, 4.0); ax.set_ylim(1e-3, 1e1)
        ax.axvline(0, color="grey", lw=1, ls="--", alpha=0.5)
        ax.set_xlabel(r"$J = \det(\mathbf{I} + \mathbf{G})$")
        if col == 0:
            ax.set_ylabel("Probability density")
        ax.set_title(f"PM {cm} — extreme cosmo (n_s=1.25, σ_8=1.0, Ω_c=0.45)",
                     fontsize=10)
        pk_shape = pk_Pk_cut if cm == "cutoff" else pk_Pk
        for i, a_pm in enumerate([0.1, 0.3]):
            snap = load_extreme(cm, a_pm)
            G_pm, J_pm = deformation_gradient_jacobian(snap)
            sigma2 = float(jnp.var(jnp.einsum("...ii->...", G_pm)))
            # PM histogram
            c, h = J_hist(J_pm)
            color = cmap(0.2 + 0.6 * i)
            ax.plot(c, h, color=color, lw=1.4, drawstyle="steps-mid",
                    label=fr"PM $\sigma={np.sqrt(sigma2):.2f}$")
            # Cascade at matched σ² using extreme-cosmo P(k)
            all_h = []
            for seed in range(seeds):
                key = jax.random.PRNGKey(300 + seed)
                r = fem_cascade_v3(sigma2=sigma2, n_steps=n_steps, N=N_cas,
                                   boxsize=L, key=key,
                                   g_T=0.0, alpha_nl=0.0, beta_nl=beta_nl,
                                   pk_k=pk_k, pk_Pk=pk_shape,
                                   method="midpoint", use_abs_J=True)
                jax.block_until_ready(r.J)
                c2, h2 = J_hist(r.J)
                all_h.append(h2)
            hmean = np.mean(np.stack(all_h, 0), axis=0)
            ax.plot(c2, hmean, color=color, lw=1.0, ls="--",
                    label=fr"cascade $\sigma={np.sqrt(sigma2):.2f}$")
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(r"Extreme cosmology: PM vs cascade (β$_{nl}$=-1.5, same as Planck fit)")
    fig.tight_layout()
    out = "docs/J_pdf_extreme_cosmo.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
