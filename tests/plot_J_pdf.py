"""PDF of the Lagrangian volume J = det(I + G).

Plots the PM reference (N=256, with and without small-scale cutoff) and
— when cascade results are supplied — overlays them on the same axes so
we can eyeball convergence of the cascade model to the PM target.

Usage::

    python tests/plot_J_pdf.py                        # PM reference only
    python tests/plot_J_pdf.py --cascade v5 --beta_nl -1.5 --gamma_nl 0.0
"""

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

from discodj.analysis.sim_store import load_sim, deformation_gradient_jacobian
from discodj.cascade import (
    fem_cascade, fem_cascade_v2, fem_cascade_v3, fem_cascade_v4, fem_cascade_v5,
)


def J_hist(J, bins=200, J_range=(-1.5, 4.0)):
    J = np.asarray(J).ravel()
    J = J[np.isfinite(J)]
    # Clip to keep axis reasonable but preserve shape near tails.
    h, edges = np.histogram(J, bins=bins, range=J_range, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, h


def compute_pm_J(N, cutoff_mode, a):
    snap = load_sim(N, cutoff_mode, a=a)
    G_pm, J_pm = deformation_gradient_jacobian(snap)
    sigma = float(np.sqrt(np.var(np.einsum("...ii->...", G_pm))))
    return J_pm, sigma, float(snap.a)


def run_cascade(version, sigma2, boxsize, N_cas, key, **kwargs):
    fn = {
        "v1": fem_cascade, "v2": fem_cascade_v2, "v3": fem_cascade_v3,
        "v4": fem_cascade_v4, "v5": fem_cascade_v5,
    }[version]
    # Strip kwargs that the chosen version doesn't accept
    import inspect
    sig = inspect.signature(fn).parameters
    kw = {k: v for k, v in kwargs.items() if k in sig}
    res = fn(sigma2=sigma2, N=N_cas, boxsize=boxsize, key=key, **kw)
    jax.block_until_ready(res.J)
    return np.asarray(res.J)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cutoff", choices=["cutoff", "nocutoff", "both"],
                   default="cutoff")
    p.add_argument("--N_pm", type=int, default=256)
    p.add_argument("--a", type=float, nargs="+",
                   default=[0.1, 0.3, 0.5, 0.7, 1.0])
    # Cascade overlay options
    p.add_argument("--cascade", choices=["v1", "v2", "v3", "v4", "v5", "none"],
                   default="none")
    p.add_argument("--N_cas", type=int, default=32)
    p.add_argument("--n_steps", type=int, default=10)
    p.add_argument("--method", default="midpoint")
    p.add_argument("--g_T", type=float, default=0.0)
    p.add_argument("--alpha_nl", type=float, default=0.0)
    p.add_argument("--beta_nl", type=float, default=0.0)
    p.add_argument("--gamma_nl", type=float, default=0.0)
    p.add_argument("--k_cut", type=float, default=None,
                   help="Gaussian cutoff in cascade increment (h/Mpc)")
    p.add_argument("--pk_shape", action="store_true",
                   help="Shape the cascade increment with DiscoDJ's LCDM P(k)")
    p.add_argument("--pk_cutoff", action="store_true",
                   help="When pk_shape is on AND this flag is set, also multiply "
                        "P(k) by a Gaussian at k_cut_from_suite "
                        "(matches PM 'cutoff' sims exactly)")
    p.add_argument("--use_abs_J", type=int, default=1)
    p.add_argument("--seeds", type=int, default=2,
                   help="number of cascade seeds to average histogram over")
    p.add_argument("--out", default="docs/J_pdf.png")
    p.add_argument("--title_suffix", default="")
    args = p.parse_args()

    cutoff_modes = ["cutoff", "nocutoff"] if args.cutoff == "both" else [args.cutoff]
    # Compute PM reference histograms.
    pm_data = {}
    print("Loading PM snapshots...")
    for cm in cutoff_modes:
        for a in args.a:
            J, sigma, a_eff = compute_pm_J(args.N_pm, cm, a)
            pm_data[(cm, a)] = (J_hist(J), sigma, a_eff)
            print(f"  PM {cm} a={a_eff:.2f}  σ={sigma:.3f}")

    # Optional LCDM P(k) shape for cascade increments.
    pk_k_arr = pk_Pk_arr = None
    if args.pk_shape:
        from discodj import DiscoDJ
        from discodj.cascade import pk_table_from_discodj
        dj_ref = (DiscoDJ(dim=3, res=args.N_cas, boxsize=250.0)
                  .with_timetables().with_linear_ps())
        pk_k_arr, pk_Pk_arr = pk_table_from_discodj(dj_ref)
        if args.pk_cutoff:
            # Mimic the PM cutoff sims: exp(-½(k/k_cut)²) applied to Pk.
            k_cut_suite = 0.5 * np.pi * 64 / 250.0
            import jax.numpy as jnp
            pk_Pk_arr = pk_Pk_arr * jnp.exp(-(pk_k_arr / k_cut_suite) ** 2)
        print(f"LCDM P(k) loaded; k range [{float(pk_k_arr.min()):.3e}, "
              f"{float(pk_k_arr.max()):.2f}] h/Mpc")

    # Run cascade (if requested) at matching σ²_trG from each PM snapshot.
    cascade_data = {}
    if args.cascade != "none":
        print(f"\nRunning cascade {args.cascade} at matching σ²...")
        for cm in cutoff_modes:
            for a in args.a:
                (_, _), sigma, a_eff = pm_data[(cm, a)]
                sigma2 = sigma ** 2
                # Average histogram over seeds for smooth curves.
                all_hists = []
                t0 = time.time()
                for seed in range(args.seeds):
                    key = jax.random.PRNGKey(100 + seed)
                    J = run_cascade(
                        args.cascade, sigma2, boxsize=250.0, N_cas=args.N_cas,
                        key=key, n_steps=args.n_steps,
                        g_T=args.g_T, alpha_nl=args.alpha_nl,
                        beta_nl=args.beta_nl, gamma_nl=args.gamma_nl,
                        k_cut=args.k_cut,
                        pk_k=pk_k_arr, pk_Pk=pk_Pk_arr,
                        method=args.method, use_abs_J=bool(args.use_abs_J),
                    )
                    centers, h = J_hist(J)
                    all_hists.append(h)
                hmean = np.mean(np.stack(all_hists, 0), axis=0)
                cascade_data[(cm, a)] = ((centers, hmean), sigma, a_eff)
                print(f"  cascade {cm} a={a_eff:.2f} σ²={sigma2:.3f}  "
                      f"{time.time()-t0:.0f}s")

    # Plot.
    fig, axes = plt.subplots(
        1, len(cutoff_modes), figsize=(6 * len(cutoff_modes), 6),
        squeeze=False, sharey=True,
    )
    cmap = plt.get_cmap("viridis")
    for col, cm in enumerate(cutoff_modes):
        ax = axes[0, col]
        ax.set_yscale("log")
        ax.axvline(0, color="grey", lw=1, ls="--", alpha=0.6)
        ax.text(0.02, 0.5, "shell crossing\n($J = 0$)", transform=ax.transData,
                rotation=90, va="center", color="grey", alpha=0.6, fontsize=8)
        ax.set_xlim(-1.5, 4.0)
        ax.set_ylim(1e-3, 1e1)
        ax.set_xlabel(r"$J = \det(\mathbf{I} + \mathbf{G})$")
        if col == 0:
            ax.set_ylabel("Probability density")
        ax.set_title(f"PM {cm}  (N={args.N_pm})")
        a_sorted = sorted(args.a)
        for i, a in enumerate(a_sorted):
            (centers, h), sigma, a_eff = pm_data[(cm, a)]
            color = cmap(i / max(len(a_sorted) - 1, 1))
            label = fr"PM $\sigma={sigma:.2f}$"
            ax.plot(centers, h, drawstyle="steps-mid", color=color, lw=1.4,
                    label=label)
            if (cm, a) in cascade_data:
                (cc, hc), _, _ = cascade_data[(cm, a)]
                ax.plot(cc, hc, color=color, lw=1.0, ls="--",
                        label=fr"cascade $\sigma={sigma:.2f}$")
        ax.legend(loc="upper right", fontsize=8)

    suptitle = "PDF of Lagrangian volume $J$"
    if args.cascade != "none":
        suptitle += f"  |  cascade {args.cascade}"
        for k in ("g_T", "alpha_nl", "beta_nl", "gamma_nl"):
            v = getattr(args, k)
            if v != 0.0:
                suptitle += fr" ${k.replace('_nl','')}={v:+.2f}$"
    if args.title_suffix:
        suptitle += f"  |  {args.title_suffix}"
    fig.suptitle(suptitle, y=0.995)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
