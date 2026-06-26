"""Joint 4-parameter fit on multi-seed 2LPT stride-1 data:

    S_3^V - 8/7 = (A_0 + A_1 γ_paper) σ² + (B_0 + B_1 γ_paper) σ⁴

Using γ_paper = -(n + 3) (positive). This disentangles the linear-in-σ²
coefficients (A_0, A_1) — the tree-level target — from the σ⁴ curvature
(B_0, B_1), which biased earlier linear-only fits.

Bootstrap over 5 IC seeds for proper uncertainties.
"""

from __future__ import annotations

import json

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    import sys
    sigma_max = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
    rows = json.load(open("docs/pss_lpt_multiseed.json"))
    rows = [r for r in rows if r["stride"] == 1 and r["lpt_order"] == 2
            and 0.05 <= r["sigma_lin"] <= sigma_max and r["kappa2"] > 0]
    print(f"N rows = {len(rows)}   "
          f"(stride 1, 2LPT, σ_lin ∈ [0.05, {sigma_max}])")
    seeds = sorted(set(r["seed"] for r in rows))

    def build_design(sub):
        sig2 = np.array([r["sigma_lin"]**2 for r in sub])
        g_p  = np.array([-r["gamma"] for r in sub])       # γ_paper = -γ_tom
        Y    = np.array([r["S3_V"] - 8/7 for r in sub])
        X = np.column_stack([sig2, g_p * sig2, sig2**2, g_p * sig2**2])
        return X, Y

    X, Y = build_design(rows)
    c_full, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ c_full
    rms = float(np.sqrt((resid**2).mean()))
    print(f"\nFull-data fit:")
    print(f"  A_0      = {c_full[0]:+.3f}")
    print(f"  A_1      = {c_full[1]:+.3f}  (paper conv.)")
    print(f"  B_0      = {c_full[2]:+.3f}")
    print(f"  B_1      = {c_full[3]:+.3f}")
    print(f"  rms resid = {rms:.4f}")

    # Bootstrap over seeds.
    rng = np.random.default_rng(0)
    N_boot = 2000
    boots = []
    for _ in range(N_boot):
        draw = rng.choice(seeds, size=len(seeds), replace=True)
        sub = [r for r in rows if r["seed"] in draw]
        X_b, Y_b = build_design(sub)
        c, *_ = np.linalg.lstsq(X_b, Y_b, rcond=None)
        boots.append(c)
    boots = np.array(boots)
    names = ["A_0", "A_1", "B_0", "B_1"]
    print("\nBootstrap-over-seed 68% credible intervals:")
    for i, nm in enumerate(names):
        lo, hi = np.percentile(boots[:, i], [16, 84])
        print(f"  {nm} = {c_full[i]:+.3f}  [{lo:+.3f}, {hi:+.3f}]  "
              f"(±{(hi-lo)/2:.3f})")

    # Report in the theory-agent's format
    A0_lo, A0_hi = np.percentile(boots[:, 0], [16, 84])
    A1_lo, A1_hi = np.percentile(boots[:, 1], [16, 84])
    A0 = c_full[0]; A1 = c_full[1]
    A0_err = 0.5 * (A0_hi - A0_lo); A1_err = 0.5 * (A1_hi - A1_lo)
    print(f"\n=== Paper-convention headline ===")
    print(f"  A_0 = {A0:+.2f} ± {A0_err:.2f}")
    print(f"  A_1 = {A1:+.2f} ± {A1_err:.2f}")
    print(f"  (theory-agent targets: A_0 = -0.18 ± 0.15, A_1 = +0.63 ± 0.12)")
    # theory agent sigma comparison
    sigma_A0 = (A0 - (-0.18)) / np.hypot(A0_err, 0.15)
    sigma_A1 = (A1 - (+0.63)) / np.hypot(A1_err, 0.12)
    print(f"  → A_0 is {sigma_A0:+.2f}σ from target")
    print(f"  → A_1 is {sigma_A1:+.2f}σ from target")

    # Plot fit overlay for each γ
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {-0.75: "tab:blue", -1.00: "tab:orange", -1.50: "tab:red"}
    for g_tom, color in colors.items():
        sub = [r for r in rows if abs(r["gamma"] - g_tom) < 0.01]
        x = np.array([r["sigma_lin"]**2 for r in sub])
        y = np.array([r["S3_V"] for r in sub])
        ax.plot(x, y, "o", color=color, alpha=0.5, ms=5,
                label=fr"γ_paper={-g_tom:+.2f}  (data)")
        xs = np.linspace(0, 0.6, 100)
        g_p = -g_tom
        ys = 8/7 + (c_full[0] + c_full[1]*g_p)*xs + (c_full[2] + c_full[3]*g_p)*xs**2
        ax.plot(xs, ys, "-", color=color, alpha=0.7)
    ax.axhline(8/7, ls=":", color="black", alpha=0.5, label="8/7")
    ax.set_xlabel(r"$\sigma_{\rm lin}^2$"); ax.set_ylabel(r"$S_3^V$")
    ax.set_title(fr"Joint quadratic fit:  $A_0={A0:+.2f}$, "
                  fr"$A_1={A1:+.2f}$, $B_0={c_full[2]:+.2f}$, $B_1={c_full[3]:+.2f}$")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig("docs/pss_joint_ABC_fit.png", dpi=130)
    print("\nSaved docs/pss_joint_ABC_fit.png")

    # Per-γ diagnostic: at each γ fit only σ² slope (no σ⁴) — for comparison
    print("\n=== Per-γ diagnostic (same quadratic model, isolated) ===")
    per_g = {}
    for g_tom in sorted(set(r["gamma"] for r in rows)):
        sub = [r for r in rows if abs(r["gamma"] - g_tom) < 0.01]
        boot_A = []; boot_B = []
        for _ in range(N_boot):
            draw = rng.choice(seeds, size=len(seeds), replace=True)
            samp = [r for r in sub if r["seed"] in draw]
            if len(samp) < 3: continue
            x = np.array([r["sigma_lin"]**2 for r in samp])
            y = np.array([r["S3_V"] - 8/7 for r in samp])
            X_b = np.column_stack([x, x**2])
            c_b, *_ = np.linalg.lstsq(X_b, y, rcond=None)
            boot_A.append(c_b[0]); boot_B.append(c_b[1])
        x = np.array([r["sigma_lin"]**2 for r in sub])
        y = np.array([r["S3_V"] - 8/7 for r in sub])
        X_b = np.column_stack([x, x**2])
        c_full_g, *_ = np.linalg.lstsq(X_b, y, rcond=None)
        A_lo, A_hi = np.percentile(boot_A, [16, 84])
        B_lo, B_hi = np.percentile(boot_B, [16, 84])
        print(f"  γ_paper={-g_tom:+.2f}: A = {c_full_g[0]:+.3f} "
              f"[{A_lo:+.3f}, {A_hi:+.3f}]   "
              f"B = {c_full_g[1]:+.3f} [{B_lo:+.3f}, {B_hi:+.3f}]")
        per_g[f"{-g_tom:+.2f}"] = dict(
            A_full=float(c_full_g[0]), B_full=float(c_full_g[1]),
            A_lo=float(A_lo), A_hi=float(A_hi),
            B_lo=float(B_lo), B_hi=float(B_hi),
        )

    with open("docs/pss_joint_ABC_fit.json", "w") as f:
        json.dump(dict(
            A_0=float(c_full[0]), A_1=float(c_full[1]),
            B_0=float(c_full[2]), B_1=float(c_full[3]),
            A_0_bs=[float(x) for x in boots[:50, 0]],   # first 50 for sanity
            A_1_bs=[float(x) for x in boots[:50, 1]],
            A_0_err=float(A0_err), A_1_err=float(A1_err),
            B_0_err=float(0.5 * (np.percentile(boots[:, 2], 84)
                                  - np.percentile(boots[:, 2], 16))),
            B_1_err=float(0.5 * (np.percentile(boots[:, 3], 84)
                                  - np.percentile(boots[:, 3], 16))),
            rms=rms, n=len(rows), per_gamma=per_g,
        ), f, indent=2)
    print("Saved docs/pss_joint_ABC_fit.json")


if __name__ == "__main__":
    main()
