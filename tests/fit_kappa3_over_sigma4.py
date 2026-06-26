"""Clean A_0, A_1 extraction from κ_3/σ⁴_lin directly (no κ_2 denominator).

Theory tree form:
    κ_3(V)/σ_lin(R)^4 = 8/7 + (A_0 + A_1 γ_paper) σ_lin^2 + O(σ^4)

Using κ_2 in the denominator contaminates (A_0, A_1) with finite-σ
corrections to σ_v vs σ_lin. Using the analytic σ_lin^4 is cleaner
because σ_lin is known exactly from the input power spectrum.

Stride=1, 2LPT, multi-seed bootstrap. Reports paper-convention A_0, A_1.
"""

from __future__ import annotations

import json
import sys

import numpy as np


def main():
    sigma_max = float(sys.argv[1]) if len(sys.argv) > 1 else 0.45
    sigma_min = 0.05
    rows = json.load(open("docs/pss_lpt_multiseed.json"))
    rows = [r for r in rows if r["stride"] == 1 and r["lpt_order"] == 2
            and sigma_min <= r["sigma_lin"] <= sigma_max
            and r["kappa2"] > 0]
    print(f"N rows = {len(rows)}   (stride 1, 2LPT, σ_lin ∈ "
          f"[{sigma_min:.2f}, {sigma_max:.2f}])")
    seeds = sorted(set(r["seed"] for r in rows))

    def build_design(sub):
        sig2 = np.array([r["sigma_lin"]**2 for r in sub])
        g_p  = np.array([-r["gamma"] for r in sub])       # γ_paper = -γ_tom
        kappa3 = np.array([r["kappa3"] for r in sub])
        # Y = κ_3/σ⁴_lin - 8/7  =  A_0·σ²_lin + A_1 γ_paper σ²_lin + ...
        Y = kappa3 / sig2**2 - 8.0/7.0
        X = np.column_stack([sig2, g_p * sig2])           # 2-param linear
        return X, Y, sig2

    X, Y, sig2 = build_design(rows)
    c_full, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ c_full
    rms = float(np.sqrt((resid**2).mean()))
    print(f"\nFull-data 2-parameter fit   κ_3/σ⁴_lin − 8/7 = "
          f"(A_0 + A_1 γ_paper) σ²_lin:")
    print(f"  A_0 = {c_full[0]:+.3f}")
    print(f"  A_1 = {c_full[1]:+.3f}")
    print(f"  rms residual = {rms:.4f}")

    # Bootstrap over seeds
    rng = np.random.default_rng(0)
    N_boot = 3000
    boots = []
    for _ in range(N_boot):
        draw = rng.choice(seeds, size=len(seeds), replace=True)
        sub = [r for r in rows if r["seed"] in draw]
        X_b, Y_b, _ = build_design(sub)
        if len(Y_b) < 3: continue
        c_b, *_ = np.linalg.lstsq(X_b, Y_b, rcond=None)
        boots.append(c_b)
    boots = np.array(boots)
    A0_lo, A0_hi = np.percentile(boots[:, 0], [16, 84])
    A1_lo, A1_hi = np.percentile(boots[:, 1], [16, 84])
    A0_err = 0.5 * (A0_hi - A0_lo); A1_err = 0.5 * (A1_hi - A1_lo)
    print(f"\nBootstrap-over-seed 68% CL:")
    print(f"  A_0 = {c_full[0]:+.3f}  [{A0_lo:+.3f}, {A0_hi:+.3f}]  ±{A0_err:.3f}")
    print(f"  A_1 = {c_full[1]:+.3f}  [{A1_lo:+.3f}, {A1_hi:+.3f}]  ±{A1_err:.3f}")

    # Compare to theory agent targets
    print(f"\n=== vs theory-agent targets ===")
    A0_t = -0.18; A0_te = 0.15
    A1_t = +0.63; A1_te = 0.12
    s0 = (c_full[0] - A0_t) / np.hypot(A0_err, A0_te)
    s1 = (c_full[1] - A1_t) / np.hypot(A1_err, A1_te)
    print(f"  A_0: target {A0_t:+.2f}±{A0_te:.2f}, "
          f"measured {c_full[0]:+.2f}±{A0_err:.2f}  (Δ = {s0:+.2f}σ)")
    print(f"  A_1: target {A1_t:+.2f}±{A1_te:.2f}, "
          f"measured {c_full[1]:+.2f}±{A1_err:.2f}  (Δ = {s1:+.2f}σ)")

    # Include a 3-parameter extension with σ⁴·σ²_lin to verify stability
    def build_design3(sub):
        sig2 = np.array([r["sigma_lin"]**2 for r in sub])
        g_p  = np.array([-r["gamma"] for r in sub])
        kappa3 = np.array([r["kappa3"] for r in sub])
        Y = kappa3 / sig2**2 - 8.0/7.0
        X = np.column_stack([sig2, g_p * sig2, sig2**2])
        return X, Y

    X3, Y3 = build_design3(rows)
    c3, *_ = np.linalg.lstsq(X3, Y3, rcond=None)
    print(f"\n=== 3-parameter stability check (added σ⁴ term) ===")
    print(f"  A_0 = {c3[0]:+.3f}   A_1 = {c3[1]:+.3f}   "
          f"σ⁴-coef = {c3[2]:+.3f}")

    # Per-γ standalone check
    print(f"\n=== Per-γ check: fit only α₀(γ) for each γ, "
          f"then linear fit α₀(γ) vs γ_paper ===")
    per_g_A = {}
    for g_tom in sorted(set(r["gamma"] for r in rows)):
        sub = [r for r in rows if abs(r["gamma"] - g_tom) < 0.01]
        if len(sub) < 3: continue
        sig2 = np.array([r["sigma_lin"]**2 for r in sub])
        kappa3 = np.array([r["kappa3"] for r in sub])
        Y = kappa3 / sig2**2 - 8.0/7.0
        # Y = α₀·σ², single-parameter linear through origin
        alpha0 = float(np.sum(Y * sig2) / np.sum(sig2**2))
        # Bootstrap over seeds
        boots_a = []
        for _ in range(N_boot):
            draw = rng.choice(seeds, size=len(seeds), replace=True)
            samp = [r for r in sub if r["seed"] in draw]
            if len(samp) < 3: continue
            sig2_b = np.array([r["sigma_lin"]**2 for r in samp])
            kappa3_b = np.array([r["kappa3"] for r in samp])
            Y_b = kappa3_b / sig2_b**2 - 8.0/7.0
            a_b = float(np.sum(Y_b * sig2_b) / np.sum(sig2_b**2))
            boots_a.append(a_b)
        alo, ahi = np.percentile(boots_a, [16, 84])
        per_g_A[-g_tom] = (alpha0, float(alo), float(ahi))
        print(f"  γ_paper={-g_tom:+.2f}: α₀ = A_0 + A_1·γ = "
              f"{alpha0:+.3f} [{alo:+.3f}, {ahi:+.3f}]")

    # Fit linear across the 3 γ values
    g_arr = np.array(sorted(per_g_A.keys()))
    a_arr = np.array([per_g_A[g][0] for g in g_arr])
    X_lin = np.column_stack([np.ones_like(g_arr), g_arr])
    c_lin, *_ = np.linalg.lstsq(X_lin, a_arr, rcond=None)
    print(f"  → α₀(γ) = A_0 + A_1·γ :  A_0 = {c_lin[0]:+.3f}  "
          f"A_1 = {c_lin[1]:+.3f}")

    with open("docs/pss_kappa3_fit.json", "w") as f:
        json.dump(dict(
            sigma_range=[sigma_min, sigma_max],
            A_0=float(c_full[0]), A_1=float(c_full[1]),
            A_0_err=float(A0_err), A_1_err=float(A1_err),
            A_0_bs_range=[float(A0_lo), float(A0_hi)],
            A_1_bs_range=[float(A1_lo), float(A1_hi)],
            A_0_3param=float(c3[0]), A_1_3param=float(c3[1]),
            sig4_coef_3param=float(c3[2]),
            per_gamma={f"{g:+.2f}": list(v) for g, v in per_g_A.items()},
        ), f, indent=2)
    print("\nSaved docs/pss_kappa3_fit.json")


if __name__ == "__main__":
    main()
