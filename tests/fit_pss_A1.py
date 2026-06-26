"""Fit A_1 from scale-free PSS data in the cosmic-signal-dominated regime.

Model: S_3^V(σ, γ) = 8/7 + (A_0 + A_1 · γ) · σ² + O(σ⁴).

Filters applied to the raw 166-row table:
  - SF sims only (scale-free; γ known exactly).
  - σ_lin ∈ [0.08, 1.0]  (above stride/grid noise floor, below PT edge).
  - n_eff ≥ 512          (drops stride=32 at N=128 and stride=64 everywhere).

Fit: 2-parameter linear least squares on the transform
    S_3^V - 8/7 = (A_0 + A_1 γ) · σ²
→ we regress Y ≡ (S_3^V - 8/7)/σ² on γ linearly, Y = A_0 + A_1 γ.

Also emit a plot of the usable subset with the fitted line overlaid, and
the "A_0 = 32/49 at γ = 0" cross-check.
"""

from __future__ import annotations

import json

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    rows = json.load(open("docs/pss_tree_cumulants.json"))
    # usable subset
    sf = [r for r in rows if r["sim"].startswith("sf_")
          and 0.08 <= r["sigma_lin"] <= 1.0
          and r["n_eff"] >= 512]
    print(f"Usable SF rows: {len(sf)} / {len(rows)}")

    sig2 = np.array([r["sigma_lin"]**2 for r in sf])
    gamma = np.array([r["gamma"] for r in sf])
    S3V  = np.array([r["S3_V"] for r in sf])

    # Per-γ S₃^V tabulation within σ²=[0.1, 0.5] band (cleanest regime).
    print("\nPer-γ S₃^V in σ²∈[0.1, 0.5]:")
    print("  γ     | n  | S₃^V mean ± std")
    for g in sorted(set(gamma.round(3))):
        m = np.isclose(gamma, g) & (0.1 <= sig2) & (sig2 <= 0.5)
        if not m.any():
            continue
        s3_sub = S3V[m]
        print(f"  {g:+.2f}  | {int(m.sum()):>2} | {s3_sub.mean():+.3f} ± {s3_sub.std():.3f}")

    # 2-parameter fit DIRECTLY on S₃^V − 8/7 vs (σ², γ·σ²). No σ² division
    # in the response (which would amplify σ→0 noise).
    Y = S3V - 8.0/7.0
    X = np.column_stack([sig2, gamma * sig2])
    coeffs, *_ = np.linalg.lstsq(X, Y, rcond=None)
    A_0, A_1 = float(coeffs[0]), float(coeffs[1])
    resid = Y - X @ coeffs
    rms = float(np.sqrt(np.mean(resid**2)))
    XTX_inv = np.linalg.inv(X.T @ X)
    se = rms * np.sqrt(np.diag(XTX_inv))
    print(f"\nFit (direct): S₃^V = 8/7 + ({A_0:+.3f} ± {se[0]:.3f} + "
          f"({A_1:+.3f} ± {se[1]:.3f}) · γ) · σ²")
    print(f"  rms residual (in S₃^V) = {rms:.3f}")
    A0_theory = 32.0/49.0
    print(f"  prediction vs theory:  A_0 pred = 32/49 = {A0_theory:.3f}  "
          f"(Δ = {A_0 - A0_theory:+.3f}, {(A_0 - A0_theory)/se[0]:+.2f}σ)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # Left: Y vs γ
    ax = axes[0]
    for g in sorted(set(gamma.round(3))):
        m = np.isclose(gamma, g)
        ax.plot(gamma[m] + 0.02 * (np.random.default_rng(42).random(m.sum()) - 0.5),
                Y[m], "o", ms=6, alpha=0.6,
                label=fr"γ={g:+.2f}  ({m.sum()} pts)")
    gs = np.linspace(-1.7, -0.6, 100)
    ax.plot(gs, A_0 + A_1 * gs, "--", color="black",
            label=fr"fit: $A_0 {A_0:+.3f},\,A_1 {A_1:+.3f}$")
    ax.axhline(32/49, ls=":", color="tab:green", alpha=0.7,
               label=r"theory: $A_0 = 32/49$")
    ax.set_xlabel(r"$\gamma$"); ax.set_ylabel(r"$(S_3^V - 8/7) / \sigma^2$")
    ax.set_title("A_1 extraction from scale-free PSS")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # Right: S_3^V vs σ² coloured by γ, with fit lines per γ
    ax = axes[1]
    colors = {-0.75: "tab:blue", -1.00: "tab:orange", -1.50: "tab:red"}
    for g, color in colors.items():
        m = np.isclose(gamma, g)
        ax.plot(sig2[m], S3V[m], "o", color=color, alpha=0.7, ms=6,
                label=fr"γ={g:+.2f}")
        xs = np.linspace(0, 2.2, 100)
        ax.plot(xs, 8/7 + (A_0 + A_1 * g) * xs, "-", color=color, alpha=0.3)
    ax.axhline(8/7, ls="--", color="black", label=r"$8/7$")
    ax.set_xlabel(r"$\sigma_{\rm lin}^2$"); ax.set_ylabel(r"$S_3^V$")
    ax.set_title(r"$S_3^V(\sigma^2)$ with fitted γ-dependence")
    ax.set_ylim(-0.5, 2.5); ax.set_xlim(0, 2.2)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("docs/pss_A1_fit.png", dpi=130)
    print("\nSaved docs/pss_A1_fit.png")

    # Persist fit
    with open("docs/pss_A1_fit.json", "w") as f:
        json.dump(dict(A_0=A_0, A_1=A_1, se_A0=float(se[0]), se_A1=float(se[1]),
                       rms=rms, n_rows=len(sf),
                       filters=dict(sigma_lin_min=0.08, sigma_lin_max=1.0,
                                     n_eff_min=512)), f, indent=2)
    print("Saved docs/pss_A1_fit.json")


if __name__ == "__main__":
    main()
