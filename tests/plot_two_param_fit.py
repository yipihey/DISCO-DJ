"""Plot J-PDF overlay with β_eff(σ², γ) from the two-parameter joint fit.

Loads fit results from docs/cascade_two_param_fit.json, then for each
Planck snapshot runs the cascade at β_eff(σ², γ) and overlays on PM.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

from discodj.analysis import (
    load_sim, deformation_gradient_jacobian,
    gamma_for_snapshot, reconstruct_pk,
)
from discodj.cascade import fem_cascade_v3


def J_hist(J, rng=(-1.5, 4.0), bins=200):
    J = np.asarray(J).ravel()
    J = J[np.isfinite(J)]
    h, edges = np.histogram(J, bins=bins, range=rng, density=True)
    return 0.5 * (edges[:-1] + edges[1:]), h


def Jmom(J):
    a = np.asarray(J).ravel()
    m = a.mean(); d = a - m
    v = float(np.mean(d*d))
    s3 = float(np.mean(d**3) / max(v, 1e-30)**1.5)
    k4 = float(np.mean(d**4) / max(v, 1e-30)**2 - 3.0)
    return m, v, s3, k4


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fit", default="docs/cascade_two_param_fit.json")
    p.add_argument("--out", default="docs/J_pdf_two_param_fit.png")
    p.add_argument("--seeds", type=int, default=3)
    args = p.parse_args()

    with open(args.fit) as f:
        fit = json.load(f)
    b0, b_s, b_g = fit["beta_0"], fit["beta_sigma"], fit["beta_gamma"]
    print(f"β_eff = {b0:+.3f} + ({b_s:+.3f})·σ² + ({b_g:+.3f})·γ")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    cmap = plt.get_cmap("viridis")
    for col, cm in enumerate(["cutoff", "nocutoff"]):
        ax = axes[col]
        ax.set_yscale("log"); ax.set_xlim(-1.5, 4.0); ax.set_ylim(1e-3, 1e1)
        ax.axvline(0, color="grey", lw=1, ls="--", alpha=0.5)
        ax.set_xlabel(r"$J = \det(\mathbf{I} + \mathbf{G})$")
        if col == 0:
            ax.set_ylabel("Probability density")
        ax.set_title(f"PM {cm}  (β_eff from joint fit)")
        a_list = [0.03, 0.05, 0.07, 0.10, 0.30]
        for i, a_pm in enumerate(a_list):
            try:
                snap = load_sim(256, cm, a=a_pm)
            except FileNotFoundError:
                continue
            G_pm, J_pm = deformation_gradient_jacobian(snap)
            sigma2 = float(jnp.var(jnp.einsum("...ii->...", G_pm)))
            if sigma2 > 0.3 and a_pm == 0.30 and cm == "nocutoff":
                continue
            _, R, gamma = gamma_for_snapshot(snap, sigma2_trG=sigma2)
            b_eff = b0 + b_s * sigma2 + b_g * gamma
            # PM hist
            cP, hP = J_hist(J_pm)
            color = cmap(0.1 + 0.85 * i / (len(a_list) - 1))
            ax.plot(cP, hP, color=color, lw=1.4, drawstyle="steps-mid",
                    label=fr"PM $\sigma={np.sqrt(sigma2):.2f}$")
            # Cascade with β_eff
            k_np, Pk_np = reconstruct_pk(snap)
            pk_k = jnp.asarray(k_np); pk_Pk = jnp.asarray(Pk_np)
            hs = []
            for seed in range(args.seeds):
                r = fem_cascade_v3(sigma2=sigma2, n_steps=10, N=32, boxsize=250.0,
                                   key=jax.random.PRNGKey(700 + seed),
                                   g_T=0.0, alpha_nl=0.0, beta_nl=float(b_eff),
                                   pk_k=pk_k, pk_Pk=pk_Pk,
                                   method="midpoint", use_abs_J=True)
                jax.block_until_ready(r.J)
                c, h = J_hist(r.J); hs.append(h)
            hm = np.mean(np.stack(hs), 0)
            ax.plot(c, hm, color=color, lw=1.0, ls="--",
                    label=fr"cascade $\beta={b_eff:+.2f}$")
        ax.legend(loc="upper right", fontsize=7)

    fig.suptitle(
        rf"Two-parameter fit  β_eff = {b0:+.2f} + ({b_s:+.2f})·σ² + "
        rf"({b_g:+.2f})·γ")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
