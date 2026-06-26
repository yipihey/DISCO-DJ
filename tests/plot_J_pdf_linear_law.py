"""Apply the linear law β_nl(ln σ) = β₀ + β_ln · ln σ to all nocut snapshots
and overlay cascade J-PDFs against PM.

Reads coefficients from docs/beta_nocut_refit.json (or falls back to
docs/beta_logsigma_fit.json if the former is missing), rebuilds P(k) from
each snapshot's cosmo, and overlays PM + cascade J-histograms at a panel
per snapshot.

Usage:
  python tests/plot_J_pdf_linear_law.py
"""
import json, os

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax, jax.numpy as jnp

from discodj.analysis import load_sim, deformation_gradient_jacobian
from discodj.analysis.spectral_slope import reconstruct_pk
from discodj.cascade import fem_cascade_v3


N_CAS, N_STEPS, BOXSIZE, SEEDS = 32, 10, 250.0, 3
A_LIST = [0.03, 0.05, 0.07, 0.10, 0.30, 0.50, 0.70, 1.0]


def Jmom(J):
    a = np.asarray(J).ravel()
    m = a.mean(); d = a - m
    v = float(np.mean(d * d))
    s3 = float(np.mean(d**3) / max(v, 1e-30)**1.5)
    return float(m), v, s3


def J_hist(J, rng=(-2.0, 6.0), bins=200):
    J = np.asarray(J).ravel()
    J = J[np.isfinite(J)]
    h, edges = np.histogram(J, bins=bins, range=rng, density=True)
    return 0.5 * (edges[:-1] + edges[1:]), h


def load_law():
    for path in ("docs/beta_nocut_refit.json", "docs/beta_logsigma_fit.json"):
        if os.path.exists(path):
            d = json.load(open(path))
            return float(d["intercept"]), float(d["slope"]), path
    raise FileNotFoundError("No β(ln σ) fit JSON found")


def main():
    b0, b_ln, src = load_law()
    print(f"Using β = {b0:+.3f} + ({b_ln:+.3f}) · ln σ   (from {src})\n")

    ncol = 4; nrow = (len(A_LIST) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 3.2 * nrow), sharey=True)
    axes = np.atleast_2d(axes).ravel()

    for ax, a_pm in zip(axes, A_LIST):
        try:
            snap = load_sim(256, "nocutoff", a=a_pm)
        except FileNotFoundError:
            ax.set_visible(False); continue
        G_pm, J_pm = deformation_gradient_jacobian(snap)
        s2 = float(jnp.var(jnp.einsum("...ii->...", G_pm)))
        _, v_pm, S3_pm = Jmom(J_pm)
        ln_sig = 0.5 * np.log(s2)
        b_eff = b0 + b_ln * ln_sig
        print(f"  a={a_pm:.2f}  σ={np.sqrt(s2):.2f}  ln σ={ln_sig:+.2f}  "
              f"β_eff={b_eff:+.2f}  PM S₃={S3_pm:+.2f}")

        k_np, Pk_np = reconstruct_pk(snap)
        pk_k, pk_Pk = jnp.asarray(k_np), jnp.asarray(Pk_np)
        hs, v_cas_list, S3_cas_list = [], [], []
        for s in range(SEEDS):
            r = fem_cascade_v3(sigma2=s2, n_steps=N_STEPS, N=N_CAS,
                               boxsize=BOXSIZE, key=jax.random.PRNGKey(1300 + s),
                               g_T=0.0, alpha_nl=0.0, beta_nl=float(b_eff),
                               pk_k=pk_k, pk_Pk=pk_Pk,
                               method="midpoint", use_abs_J=True)
            jax.block_until_ready(r.J)
            c, h = J_hist(r.J); hs.append(h)
            _, v_c, S3_c = Jmom(r.J)
            v_cas_list.append(v_c); S3_cas_list.append(S3_c)
        hm = np.mean(np.stack(hs), 0)
        v_cas = float(np.mean(v_cas_list)); S3_cas = float(np.mean(S3_cas_list))

        cP, hP = J_hist(J_pm)
        ax.plot(cP, hP, color="tab:blue", lw=1.4, drawstyle="steps-mid",
                label=f"PM  σ={np.sqrt(s2):.2f}  S₃={S3_pm:+.2f}")
        ax.plot(cP, hm, color="tab:red", lw=1.0, ls="--",
                label=f"cas β={b_eff:+.2f}  S₃={S3_cas:+.2f}")
        ax.set_yscale("log"); ax.set_xlim(-1.5, 4.0); ax.set_ylim(1e-3, 1e1)
        ax.axvline(0, color="grey", lw=0.5, ls=":")
        ax.set_title(f"nocut a={a_pm:.2f}", fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlabel(r"$J$")

    for ax in axes[len(A_LIST):]:
        ax.set_visible(False)
    fig.suptitle(fr"Nocut suite — linear law $\beta = {b0:+.2f} + ({b_ln:+.2f})\ln\sigma$",
                 fontsize=12)
    fig.tight_layout()
    out = "docs/J_pdf_linear_law_nocut.png"
    fig.savefig(out, dpi=130)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
