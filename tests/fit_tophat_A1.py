"""Fit A_1 from the tophat-smoothed V-field moments.

Uses docs/pss_lpt_tophat_cumulants.json from tophat_smooth_V.py. Separate
fits per tophat radius R_s (s ∈ {2, 4, 8, 16}) and per LPT order.

Model: S_3^V(σ, γ) = 8/7 + (A_0 + A_1 γ) σ² + O(σ⁴).
"""

from __future__ import annotations

import json

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def fit(rows, window, order, sigma_min, sigma_max):
    sub = [r for r in rows if r.get("window") == window
            and r.get("lpt_order") == order
            and sigma_min <= r["sigma_lin"] <= sigma_max
            and r["kappa2"] > 0]
    if len(sub) < 3:
        return None
    sig2 = np.array([r["sigma_lin"]**2 for r in sub])
    gamma = np.array([r["gamma"] for r in sub])
    S3 = np.array([r["S3_V"] for r in sub])
    Y = S3 - 8.0/7.0
    X = np.column_stack([sig2, gamma * sig2])
    c, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ c
    rms = float(np.sqrt((resid**2).mean()))
    XTX_inv = np.linalg.inv(X.T @ X)
    se = rms * np.sqrt(np.diag(XTX_inv))
    return dict(A_0=float(c[0]), A_1=float(c[1]),
                se_A0=float(se[0]), se_A1=float(se[1]),
                rms=rms, n=len(sub))


def main():
    rows = json.load(open("docs/pss_lpt_tophat_cumulants.json"))
    # Windows: tet_stride1 and tophat_R{s} for s in {2, 4, 8, 16}
    windows = [("tet_stride1", 0.05, 0.6),
               ("tophat_R2",   0.02, 0.6),
               ("tophat_R4",   0.02, 0.6),
               ("tophat_R8",   0.02, 0.4),
               ("tophat_R16",  0.01, 0.3)]
    orders = [2, 3, 5]
    print(f"{'window':>14} {'σ-range':>14} {'ord':>4} {'n':>4} "
          f"{'A_0 ± se':>18} {'A_1 ± se':>18} {'rms':>6}")
    for win, lo, hi in windows:
        for order in orders:
            r = fit(rows, win, order, lo, hi)
            if r is None: continue
            print(f"{win:>14} {f'[{lo:.2f},{hi:.2f}]':>14} {order:>4} "
                  f"{r['n']:>4}  {r['A_0']:+.3f}±{r['se_A0']:.3f}    "
                  f"{r['A_1']:+.3f}±{r['se_A1']:.3f}    {r['rms']:6.3f}")

    # Compact comparison plot — focus on 2LPT.
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"tet_stride1": "tab:gray",
              "tophat_R2":   "tab:blue",
              "tophat_R4":   "tab:green",
              "tophat_R8":   "tab:orange",
              "tophat_R16":  "tab:red"}
    for win, lo, hi in windows:
        sub = [r for r in rows if r.get("window") == win
                and r["lpt_order"] == 2 and r["kappa2"] > 0]
        sub.sort(key=lambda r: r["sigma_lin"])
        # bin S_3^V by γ (each γ-slice shown as offset)
        for g, color_shift in zip((-0.75, -1.00, -1.50), (+0.1, 0.0, -0.1)):
            pts = [r for r in sub if abs(r["gamma"] - g) < 0.01]
            if not pts: continue
            xs = np.array([r["sigma_lin"]**2 for r in pts])
            ys = np.array([r["S3_V"] for r in pts])
            ax.plot(xs, ys + color_shift, "o" if win == "tet_stride1" else "s",
                     color=colors[win], alpha=0.6, ms=5,
                     label=f"{win} γ={g}" if g == -1.0 else None)
    ax.axhline(8/7, ls=":", color="black", alpha=0.7, label=r"$8/7$")
    ax.set_xlabel(r"$\sigma_{\rm lin}^2$"); ax.set_ylabel(r"$S_3^V$ (γ-offset ± 0.1)")
    ax.set_xlim(0, 0.6); ax.set_ylim(0, 2.5)
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    ax.set_title("Tophat vs tet windows, 2LPT, γ-slice offset ±0.1")
    fig.tight_layout()
    fig.savefig("docs/pss_tophat_A1_fit.png", dpi=130)
    print(f"\nSaved docs/pss_tophat_A1_fit.png")


if __name__ == "__main__":
    main()
