"""Cross-validate the Planck-fit β_eff(σ², γ) on the extreme cosmology.

Loads the joint-fit coefficients from docs/cascade_two_param_fit.json,
builds the extreme-cosmo P(k), evaluates γ for each extreme snapshot, and
runs the cascade at β_eff. The PDF overlay tests whether the fit is
cosmology-universal.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

from discodj import DiscoDJ
from discodj.analysis import deformation_gradient_jacobian
from discodj.analysis.sim_store import SimSnapshot
from discodj.analysis.spectral_slope import (
    gamma_for_snapshot, smoothed_sigma2, spectral_gamma, R_from_sigma2,
)
from discodj.cascade import fem_cascade_v3


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


def J_hist(J, rng=(-1.5, 4.0), bins=200):
    J = np.asarray(J).ravel()
    J = J[np.isfinite(J)]
    h, edges = np.histogram(J, bins=bins, range=rng, density=True)
    return 0.5 * (edges[:-1] + edges[1:]), h


def main():
    fit = json.load(open("docs/cascade_two_param_fit.json"))
    b0, b_s, b_g = fit["beta_0"], fit["beta_sigma"], fit["beta_gamma"]
    print(f"Planck-fit β_eff = {b0:+.3f} + ({b_s:+.3f})·σ² + ({b_g:+.3f})·γ\n")

    # Build extreme-cosmo Pk
    dj_x = (DiscoDJ(dim=3, res=32, boxsize=250.0, cosmo=EXTREME_COSMO)
            .with_timetables().with_linear_ps())
    k_ext = np.asarray(dj_x._pk_table["k"])
    Pk_ext = np.asarray(dj_x._pk_table["Pk"])
    k_cut_suite = 0.5 * np.pi * 64 / 250.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    cmap = plt.get_cmap("viridis")
    for col, cm in enumerate(["cutoff", "nocutoff"]):
        ax = axes[col]
        ax.set_yscale("log"); ax.set_xlim(-1.5, 4.0); ax.set_ylim(1e-3, 1e1)
        ax.axvline(0, color="grey", lw=1, ls="--", alpha=0.5)
        ax.set_xlabel(r"$J = \det(\mathbf{I} + \mathbf{G})$")
        if col == 0: ax.set_ylabel("Probability density")
        ax.set_title(f"extreme cosmo {cm}  —  Planck-fit β_eff applied")

        # Pk used for THIS cutoff mode
        Pk_use = Pk_ext.copy()
        if cm == "cutoff":
            Pk_use = Pk_use * np.exp(-(k_ext / k_cut_suite) ** 2)
        pk_k_j = jnp.asarray(k_ext); pk_Pk_j = jnp.asarray(Pk_use)

        for i, a_pm in enumerate([0.1, 0.3]):
            try:
                snap = load_extreme(cm, a_pm)
            except FileNotFoundError:
                continue
            G_pm, J_pm = deformation_gradient_jacobian(snap)
            sigma2 = float(jnp.var(jnp.einsum("...ii->...", G_pm)))
            # γ from the extreme P(k)
            R_eff = R_from_sigma2(k_ext, Pk_use, sigma2)
            gamma = spectral_gamma(k_ext, Pk_use, R_eff)
            b_eff = b0 + b_s * sigma2 + b_g * gamma
            print(f"  {cm} a={snap.a:.2f}  σ²={sigma2:.3f}  γ={gamma:+.2f}  "
                  f"→ β_eff={b_eff:+.3f}")
            # PM hist
            cP, hP = J_hist(J_pm)
            color = cmap(0.2 + 0.6 * i)
            ax.plot(cP, hP, color=color, lw=1.4, drawstyle="steps-mid",
                    label=fr"PM $\sigma={np.sqrt(sigma2):.2f}$")
            # Cascade
            hs = []
            for seed in range(3):
                r = fem_cascade_v3(sigma2=sigma2, n_steps=10, N=32, boxsize=250.0,
                                   key=jax.random.PRNGKey(900 + seed),
                                   g_T=0.0, alpha_nl=0.0, beta_nl=float(b_eff),
                                   pk_k=pk_k_j, pk_Pk=pk_Pk_j,
                                   method="midpoint", use_abs_J=True)
                jax.block_until_ready(r.J)
                c, h = J_hist(r.J); hs.append(h)
            hm = np.mean(np.stack(hs), 0)
            ax.plot(c, hm, color=color, lw=1.0, ls="--",
                    label=fr"cascade $\beta={b_eff:+.2f}$")
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("Held-out extreme cosmology  (n_s=1.25, σ_8=1.0, Ω_c=0.45)")
    fig.tight_layout()
    out = "docs/J_pdf_two_param_extreme.png"
    fig.savefig(out, dpi=120)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
