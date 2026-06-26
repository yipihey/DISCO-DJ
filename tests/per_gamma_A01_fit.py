"""Per-γ A(γ), B(γ) extraction from multi-seed 2LPT stride-1 data.

For each sim (fixed γ) separately, fit
    S_3^V(σ) = 8/7 + A(γ) σ² + B(γ) σ⁴
by OLS on the pooled seed+a rows, bootstrap over seeds for errors on
A(γ) and B(γ). Then fit A(γ) vs γ linearly to extract
    A(γ) = A_0 + A_1_tom · γ_tom   (our convention; γ_tom < 0)
    A(γ) = A_0 − A_1_paper · γ_tom  → A_1_paper = −A_1_tom (sign flip)

Two per-γ diagnostics delivered:
  1. A_0 extracted from each γ slice independently — confirms γ-independence.
  2. Quadratic fit per γ isolates linear-in-σ² term from curvature.

Output: console table + JSON with per-γ (A, B) and global (A_0, A_1).
"""

from __future__ import annotations

import json

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def fit_AB(sigma2, S3):
    """Fit Y = A·σ² + B·σ⁴ to Y = S_3^V − 8/7 via OLS.
    Returns (A, B)."""
    Y = S3 - 8.0/7.0
    X = np.column_stack([sigma2, sigma2**2])
    c, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return float(c[0]), float(c[1])


def main():
    rows = json.load(open("docs/pss_lpt_multiseed.json"))
    # Filter: stride=1, lpt_order=2, σ_lin ∈ [0.05, 0.60]
    rows = [r for r in rows if r["stride"] == 1 and r["lpt_order"] == 2
            and 0.05 <= r["sigma_lin"] <= 0.60 and r["kappa2"] > 0]
    gammas_tom = sorted(set(r["gamma"] for r in rows))
    print(f"Using {len(rows)} rows, strides: stride=1 only, LPT order=2, "
          f"σ_lin ∈ [0.05, 0.60].")
    print(f"γ_tom values: {gammas_tom}  (γ_paper = −γ_tom)\n")

    # Diagnostic 1: per-γ A, B with bootstrap-over-seeds.
    rng = np.random.default_rng(0)
    N_boot = 2000
    per_gamma = {}
    print(f"=== Per-γ quadratic fits  S_3^V = 8/7 + A·σ² + B·σ⁴ ===")
    print(f"{'γ_tom':>7} {'γ_paper':>9} {'N':>4} "
          f"{'A (68%)':>28} {'B (68%)':>28}")
    for g_tom in gammas_tom:
        sub = [r for r in rows if abs(r["gamma"] - g_tom) < 0.01]
        seeds = sorted(set(r["seed"] for r in sub))
        boot_A = []; boot_B = []
        for _ in range(N_boot):
            draw = rng.choice(seeds, size=len(seeds), replace=True)
            samp = [r for r in sub if r["seed"] in draw]
            if len(samp) < 3: continue
            sig2 = np.array([r["sigma_lin"]**2 for r in samp])
            S3  = np.array([r["S3_V"] for r in samp])
            A, B = fit_AB(sig2, S3)
            boot_A.append(A); boot_B.append(B)
        sig2 = np.array([r["sigma_lin"]**2 for r in sub])
        S3  = np.array([r["S3_V"] for r in sub])
        A_full, B_full = fit_AB(sig2, S3)
        A_lo, A_hi = np.percentile(boot_A, [16, 84])
        B_lo, B_hi = np.percentile(boot_B, [16, 84])
        per_gamma[g_tom] = dict(
            gamma_tom=g_tom, gamma_paper=-g_tom, N=len(sub),
            A_full=A_full, B_full=B_full,
            A_lo=float(A_lo), A_hi=float(A_hi),
            B_lo=float(B_lo), B_hi=float(B_hi),
            A_boot=[float(x) for x in boot_A],
            B_boot=[float(x) for x in boot_B],
        )
        print(f"{g_tom:+7.2f}  {-g_tom:+9.2f}  {len(sub):>4}   "
              f"{A_full:+.3f} [{A_lo:+.3f}, {A_hi:+.3f}]   "
              f"{B_full:+.3f} [{B_lo:+.3f}, {B_hi:+.3f}]")

    # Diagnostic 2: linear fit A(γ) = A_0 + A_1_tom·γ_tom (our convention)
    g_arr = np.array([-g_tom for g_tom in gammas_tom])        # = γ_paper (positive)
    A_arr = np.array([per_gamma[g]["A_full"] for g in gammas_tom])
    # bootstrap uncertainty on slope & intercept by resampling the per-γ
    # A-bootstrap samples (they're already from the seed bootstrap).
    boot_samples = np.array([per_gamma[g]["A_boot"] for g in gammas_tom])
    N_boot_slopes = boot_samples.shape[1]
    coefs = []
    for k in range(N_boot_slopes):
        A_k = boot_samples[:, k]
        # Linear in γ_paper (positive):  A = A_0 + A_1_paper · γ_paper
        X = np.column_stack([np.ones_like(g_arr), g_arr])
        c, *_ = np.linalg.lstsq(X, A_k, rcond=None)
        coefs.append(c)
    coefs = np.array(coefs)
    A0_med = float(np.median(coefs[:, 0]))
    A1p_med = float(np.median(coefs[:, 1]))
    A0_lo, A0_hi = np.percentile(coefs[:, 0], [16, 84])
    A1_lo, A1_hi = np.percentile(coefs[:, 1], [16, 84])

    # Also full-data point estimate
    X_full = np.column_stack([np.ones_like(g_arr), g_arr])
    c_full, *_ = np.linalg.lstsq(X_full, A_arr, rcond=None)

    print(f"\n=== Linear fit of A(γ_paper) = A_0 + A_1_paper·γ_paper ===")
    print(f"  A_0       = {c_full[0]:+.3f}  [bs {A0_lo:+.3f}, {A0_hi:+.3f}]")
    print(f"  A_1_paper = {c_full[1]:+.3f}  [bs {A1_lo:+.3f}, {A1_hi:+.3f}]")
    A0_err = 0.5 * (A0_hi - A0_lo)
    A1_err = 0.5 * (A1_hi - A1_lo)
    print(f"  → A_0 = {c_full[0]:+.2f} ± {A0_err:.2f}")
    print(f"  → A_1_paper = {c_full[1]:+.2f} ± {A1_err:.2f}")
    print(f"  (theory-agent targets: A_0 = -0.18 ± 0.15, A_1 = +0.63 ± 0.12)")

    # Plot: per-γ A, B vs γ, with global fit overlaid.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    for g_tom in gammas_tom:
        p = per_gamma[g_tom]
        ax.errorbar(p["gamma_paper"], p["A_full"],
                     yerr=[[p["A_full"] - p["A_lo"]],
                            [p["A_hi"] - p["A_full"]]],
                     fmt="o", ms=9, color="tab:blue")
    xs = np.linspace(0.5, 1.7, 50)
    ax.plot(xs, c_full[0] + c_full[1] * xs, "--", color="tab:red",
            label=f"fit: A_0={c_full[0]:+.3f}, A_1={c_full[1]:+.3f}")
    ax.set_xlabel(r"$\gamma_{\rm paper} = -(n + 3)^{-1}\cdot$(−(n+3))")
    ax.set_ylabel(r"$A(\gamma) = A_0 + A_1\,\gamma$")
    ax.set_title("Per-γ A coefficient  (stride 1, 2LPT, quadratic fit)")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    ax = axes[1]
    for g_tom in gammas_tom:
        p = per_gamma[g_tom]
        ax.errorbar(p["gamma_paper"], p["B_full"],
                     yerr=[[p["B_full"] - p["B_lo"]],
                            [p["B_hi"] - p["B_full"]]],
                     fmt="o", ms=9, color="tab:green")
    ax.axhline(0, ls=":", color="grey")
    ax.set_xlabel(r"$\gamma_{\rm paper}$")
    ax.set_ylabel(r"$B(\gamma)$  (σ⁴ curvature)")
    ax.set_title("σ⁴ curvature coefficient B(γ)")
    ax.grid(alpha=0.3)
    fig.suptitle("Per-γ quadratic fit — 2LPT, stride 1, σ_lin ∈ [0.05, 0.60]",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig("docs/pss_per_gamma_AB.png", dpi=130)
    print("\nSaved docs/pss_per_gamma_AB.png")

    # Persist
    out = dict(
        gammas_tom=list(gammas_tom),
        per_gamma={f"{g:+.2f}": {k: v for k, v in per_gamma[g].items()
                                   if k not in ("A_boot", "B_boot")}
                    for g in gammas_tom},
        global_fit=dict(
            A_0_full=float(c_full[0]), A_1_paper_full=float(c_full[1]),
            A_0_bs_lo=float(A0_lo), A_0_bs_hi=float(A0_hi),
            A_1_bs_lo=float(A1_lo), A_1_bs_hi=float(A1_hi),
        ),
    )
    with open("docs/pss_per_gamma_AB.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved docs/pss_per_gamma_AB.json")


if __name__ == "__main__":
    main()
