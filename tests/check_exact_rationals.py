"""Compare the three exact scale-free rationals against our multi-seed 2LPT PSS.

Theory (γ=0, tree-level, scale-free, paper convention):
  Var(V)/sigma^2       = 132494/168909            ≈ 0.78442
  kappa3(V)/sigma^4    = 558630413/487696586      ≈ 1.14544
  S3^V = kappa3/kappa2 = 65359758321/35109320072  ≈ 1.86162

These are the γ=0 tree-level coefficients. Testing against the scale-free
n_s ∈ {-1.5, -2.0, -2.25} (γ_paper ∈ {1.5, 1.0, 0.75}) requires either:
  (a) extrapolating measured ratios to γ=0, OR
  (b) checking that the γ-dependent measurements approach the γ=0 value
      at small σ with γ-dependent corrections.

Implementation: at each γ, measure the three ratios as a function of σ²,
extrapolate to σ -> 0 (linear fit), extrapolate across γ -> 0 to get
the pure tree-level values.
"""

from __future__ import annotations

import json

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VAR_OVER_SIG2_TH   = 132494.0 / 168909.0
KAPPA3_OVER_SIG4_TH = 558630413.0 / 487696586.0
S3_V_TH            = 65359758321.0 / 35109320072.0


def main():
    rows = json.load(open("docs/pss_lpt_multiseed.json"))
    rows = [r for r in rows if r["lpt_order"] == 2 and r["stride"] == 1
            and r["kappa2"] > 0]

    print(f"Theory predictions (tree level, gamma=0):")
    print(f"  Var(V)/sigma^2    = {VAR_OVER_SIG2_TH:.6f}  "
          f"= 132494 / 168909")
    print(f"  kappa3(V)/sigma^4 = {KAPPA3_OVER_SIG4_TH:.6f}  "
          f"= 558630413 / 487696586")
    print(f"  S3^V              = {S3_V_TH:.6f}  "
          f"= 65359758321 / 35109320072")
    print()

    # Per-gamma-per-a averages across 5 seeds.
    summary = {}
    for g_tom in sorted(set(r["gamma"] for r in rows)):
        g_rows = [r for r in rows if abs(r["gamma"] - g_tom) < 0.01]
        a_vals = sorted(set(r["a"] for r in g_rows))
        print(f"--- gamma_tom = {g_tom:+.2f}  "
              f"(gamma_paper = {-g_tom:+.2f}) ---")
        print(f"  {'a':>5} {'sigma_lin':>10} {'Var/sig2':>15} "
              f"{'kappa3/sig4':>15} {'S3^V':>12}")
        per_a = []
        for a in a_vals:
            a_rows = [r for r in g_rows if abs(r["a"] - a) < 1e-4]
            if len(a_rows) < 3:
                continue
            sig_lin = float(np.mean([r["sigma_lin"] for r in a_rows]))
            sig2_lin = float(np.mean([r["sigma2_lin"] for r in a_rows]))
            v_over = [r["kappa2"] / r["sigma2_lin"] for r in a_rows]
            k3_over = [r["kappa3"] / r["sigma2_lin"]**2 for r in a_rows]
            s3 = [r["S3_V"] for r in a_rows]
            v_m, v_s = np.mean(v_over), np.std(v_over) / np.sqrt(len(v_over))
            k3_m, k3_s = np.mean(k3_over), np.std(k3_over) / np.sqrt(len(k3_over))
            s3_m, s3_s = np.mean(s3), np.std(s3) / np.sqrt(len(s3))
            print(f"  {a:5.2f} {sig_lin:10.4f} "
                  f"{v_m:+8.4f} +/- {v_s:5.3f}   "
                  f"{k3_m:+8.4f} +/- {k3_s:5.3f}   "
                  f"{s3_m:+6.3f} +/- {s3_s:5.3f}")
            per_a.append((sig_lin, sig2_lin, v_m, v_s, k3_m, k3_s, s3_m, s3_s))
        summary[-g_tom] = per_a      # key by gamma_paper
        print()

    # Extrapolate to sigma -> 0 per gamma, then extrapolate across gamma -> 0.
    # Use the smallest-sigma bin per gamma as "tree-level" estimate.
    print("=== sigma -> 0 extrapolation per gamma (linear in sigma^2) ===")
    print(f"  {'gamma_paper':>12} {'Var/sig2':>18} "
          f"{'kappa3/sig4':>18} {'S3^V':>16}")
    gp_vals = []; var_vals = []; k3_vals = []; s3_vals = []
    for gp, data in sorted(summary.items()):
        # Fit each ratio linearly in sigma^2, return the intercept.
        sig2 = np.array([d[1] for d in data])
        def fit_intercept(y_arr, yerr_arr):
            X = np.column_stack([np.ones_like(sig2), sig2])
            W = 1.0 / np.asarray(yerr_arr)**2
            # weighted least squares
            WX = X * W[:, None]
            beta = np.linalg.solve(WX.T @ X, WX.T @ np.asarray(y_arr))
            return float(beta[0])
        v_int = fit_intercept([d[2] for d in data], [d[3] for d in data])
        k3_int = fit_intercept([d[4] for d in data], [d[5] for d in data])
        s3_int = fit_intercept([d[6] for d in data], [d[7] for d in data])
        print(f"  {gp:+12.2f}  {v_int:15.4f}    "
              f"{k3_int:15.4f}    {s3_int:12.3f}")
        gp_vals.append(gp); var_vals.append(v_int)
        k3_vals.append(k3_int); s3_vals.append(s3_int)

    # Extrapolate across gamma_paper -> 0 (linear in gamma)
    gp_arr = np.array(gp_vals)
    print()
    print("=== gamma_paper -> 0 extrapolation (linear fit across 3 gamma) ===")
    print(f"  {'Ratio':<25} {'Measured @ gamma=0':>20} "
          f"{'Theory (exact rational)':>28} {'Delta':>10}")
    for name, y, th in [("Var(V)/sigma^2", var_vals, VAR_OVER_SIG2_TH),
                         ("kappa3(V)/sigma^4", k3_vals, KAPPA3_OVER_SIG4_TH),
                         ("S3^V", s3_vals, S3_V_TH)]:
        X = np.column_stack([np.ones_like(gp_arr), gp_arr])
        c, *_ = np.linalg.lstsq(X, np.asarray(y), rcond=None)
        meas = float(c[0])      # intercept at gamma=0
        delta = meas - th
        print(f"  {name:<25} {meas:+20.4f} "
              f"{th:+28.6f} {delta:+10.4f}")

    # Plot: each ratio vs gamma_paper with theory line + per-a points
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (idx_y, idx_err, name, th) in zip(axes, [
        (2, 3, "Var(V)/sigma^2",     VAR_OVER_SIG2_TH),
        (4, 5, "kappa3(V)/sigma^4",  KAPPA3_OVER_SIG4_TH),
        (6, 7, "S3^V",               S3_V_TH),
    ]):
        for gp, data in sorted(summary.items()):
            sig2 = [d[1] for d in data]
            ys = [d[idx_y] for d in data]
            yerr = [d[idx_err] for d in data]
            ax.errorbar(sig2, ys, yerr=yerr, fmt="o", ms=5,
                         label=fr"$\gamma_p={gp:+.2f}$")
        ax.axhline(th, ls="--", color="black",
                   label=f"theory @ gamma=0 = {th:.4f}")
        ax.set_xlabel(r"$\sigma_{\rm lin}^2$")
        ax.set_ylabel(name)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax.set_title(name)
    fig.suptitle("Scale-free 2LPT tree-level tests: measured vs exact rationals",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig("docs/exact_rational_check.png", dpi=130)
    print("\nSaved docs/exact_rational_check.png")


if __name__ == "__main__":
    main()
