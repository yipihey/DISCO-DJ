"""Re-fit the A_0, A_1 coefficients on the pre-computed pure-LPT PSS table.

Uses the ``docs/pss_lpt_cumulants.json`` produced by
``analyze_pss_lpt_pure.py``. No new simulations or LPT evaluations
needed — we just vary the fit σ-window and compare the result across
LPT orders.
"""

from __future__ import annotations

import json

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LPT_ORDERS = [1, 2, 3, 5, 10]
FIT_WINDOWS = [
    ("loose",  0.10, 0.85),
    ("mid",    0.15, 0.60),
    ("tight",  0.20, 0.50),
]


def fit(rows, order, sigma_min, sigma_max):
    sub = [r for r in rows if r["lpt_order"] == order
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
    rows = json.load(open("docs/pss_lpt_cumulants.json"))

    print("Fit windows applied per LPT order:")
    print(f"{'win':>6} {'σ_range':>14} {'order':>5} {'n':>4} "
          f"{'A_0':>10} {'A_1':>10} {'rms':>6}")
    results = {}
    for win_name, lo, hi in FIT_WINDOWS:
        for order in LPT_ORDERS:
            r = fit(rows, order, lo, hi)
            if r is None: continue
            print(f"{win_name:>6} {f'[{lo:.2f},{hi:.2f}]':>14} {order:>5} "
                  f"{r['n']:>4} "
                  f"{r['A_0']:+6.3f}±{r['se_A0']:.3f} "
                  f"{r['A_1']:+6.3f}±{r['se_A1']:.3f} "
                  f"{r['rms']:6.3f}")
            results[(win_name, order)] = r

    # Best fit window — mid — compare to theory
    print(f"\n=== mid window, compared to theory (A_0 = 32/49 = {32/49:+.3f}) ===")
    for order in LPT_ORDERS:
        if ("mid", order) not in results: continue
        r = results[("mid", order)]
        dA0 = r["A_0"] - 32/49
        print(f"  order={order}: A_0 - 32/49 = {dA0:+.3f}  "
              f"({dA0/r['se_A0']:+.2f}σ),  A_1 = {r['A_1']:+.3f} ± {r['se_A1']:.3f}")

    # Plot S_3^V vs σ² for each order, overlaid three γ colors, mid-window fit shown
    fig, axes = plt.subplots(1, len(LPT_ORDERS), figsize=(4*len(LPT_ORDERS), 5),
                              sharey=True)
    colors = {-0.75: "tab:blue", -1.00: "tab:orange", -1.50: "tab:red"}
    for ax, order in zip(axes, LPT_ORDERS):
        sub = [r for r in rows if r["lpt_order"] == order]
        for g, color in colors.items():
            pts = [r for r in sub if abs(r["gamma"] - g) < 0.01]
            pts.sort(key=lambda r: r["sigma_lin"])
            x = [r["sigma_lin"]**2 for r in pts]
            y = [r["S3_V"] for r in pts]
            ax.plot(x, y, "o", color=color, alpha=0.5, ms=4,
                    label=fr"γ={g:+.2f}")
        fit_r = results.get(("mid", order))
        if fit_r is not None:
            xs = np.linspace(0, 0.85, 50)
            for g, color in colors.items():
                ys = 8/7 + (fit_r["A_0"] + fit_r["A_1"] * g) * xs
                ax.plot(xs, ys, "-", color=color, alpha=0.3)
        ax.axhline(8/7, ls="--", color="black", alpha=0.5, label="8/7")
        # Mark fit window
        ax.axvspan(0.15**2, 0.60**2, color="grey", alpha=0.1)
        ax.set_xlabel(r"$\sigma^2_{\rm lin}$"); ax.set_ylabel(r"$S_3^V$")
        ax.set_title(f"LPT order {order}")
        ax.set_xlim(0, 0.85); ax.set_ylim(0, 2.5)
        ax.grid(alpha=0.3)
        if order == LPT_ORDERS[0]:
            ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Pure LPT PSS — convergence in order at fixed (σ, γ)", fontsize=12)
    fig.tight_layout()
    fig.savefig("docs/pss_lpt_A1_refit.png", dpi=130)
    print(f"\nSaved docs/pss_lpt_A1_refit.png")

    with open("docs/pss_lpt_A1_refit.json", "w") as f:
        json.dump({f"{w}_order{o}": v for (w, o), v in results.items()}, f, indent=2)
    print("Saved docs/pss_lpt_A1_refit.json")


if __name__ == "__main__":
    main()
