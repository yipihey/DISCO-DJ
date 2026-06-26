"""Extract volume-distribution cumulants per k-shell and compare to ZA
closed-form; examine the 2LPT residual structure κ_m - κ_m^ZA for the
coefficients (α_m^(0), α_m^(γ)) that multiply σ^{2m+2} at next order.

Pipeline
--------
1. For each (snap, k_target, kind ∈ {R,D}), load the cached paired V_k
   and V_{k+1} arrays from ``sims/knn_cache/*.npz``.
2. Build the exact-k volume distribution via CDF subtraction on a
   log-spaced V grid, normalise, and compute moments of the centred
   v ≡ V/⟨V⟩ - 1 up to 4th order. From moments → cumulants κ_m.
3. Compute the ZA closed-form κ_m^ZA(σ_k) at σ_k = √σ²_lin(R_k), where
   σ²_lin(R) comes from the snapshot's linear P(k) at the Lagrangian
   radius R enclosing the exact-k mean volume.
4. Tabulate (σ_k, γ_k, κ_m^data, κ_m^ZA, δκ_m ≡ κ_m^data − κ_m^ZA) per
   (snap, k, kind). Plot δκ_m / σ_k^{2m+2} vs γ_k at matched σ_k to
   isolate the α_m^(γ) coefficients.

ZA closed-form (from paper eqs 37-40; u = t σ²):
    κ_2^ZA(σ) = σ^4·(1 + 4σ²/15 + 4σ⁴/225)
    κ_3^ZA(σ) = 2σ^4·(1 + 92σ²/225 + 28σ⁴/1125)
    κ_4^ZA(σ) = (56/9)σ^4·(1 + 494σ²/875 + ...)

2LPT structural ansatz (g₂ = -3/7 at EdS):
    κ_2^total = κ_2^ZA + (2g₂/3)σ⁴ + (2g₂γ/15)σ⁴ + C_2(γ,γ₂)σ⁶ + ...
    κ_3^total = κ_3^ZA + 2g₂σ⁴ + 0·γ σ⁴ + (α₃^(0) + α₃^(γ)γ)σ⁶ + ...
    κ_4^total = κ_4^ZA + (α₄^(0) + α₄^(γ)γ)σ⁶ + ...
"""

from __future__ import annotations

import json, os

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from discodj.analysis import load_sim
from discodj.analysis.spectral_slope import (
    spectral_gamma, smoothed_sigma2, reconstruct_pk,
)


K_LIST_RAW = [1, 2, 3, 4, 5, 8, 9, 16, 17, 32, 33, 64, 65, 128, 129]
K_TARGETS  = [1, 2, 3, 4, 8, 16, 32, 64, 128]
CACHE_DIR = "sims/knn_cache"
V_GRID_N  = 800    # fine enough for stable κ_4
G2_EDS    = -3.0 / 7.0


# ----------------------------------------------------------------------
# ZA closed-form cumulants (u = tσ², ψ_ZA expansion → κ_m at σ = σ_k).
# ----------------------------------------------------------------------

def kappa_ZA(m: int, sigma2: float) -> float:
    """Zel'dovich cumulants κ_m^ZA(σ) to O(σ⁸) from the paper."""
    s2 = sigma2
    if m == 2:
        return s2**2 * (1.0 + (4.0 / 15.0) * s2 + (4.0 / 225.0) * s2**2)
    if m == 3:
        return 2.0 * s2**2 * (1.0 + (92.0 / 225.0) * s2 + (28.0 / 1125.0) * s2**2)
    if m == 4:
        # Leading σ⁴ term: (56/9)σ⁴; NLO correction 494/875 as given.
        return (56.0 / 9.0) * s2**2 * (1.0 + (494.0 / 875.0) * s2)
    raise ValueError(f"κ_{m} ZA closed-form not implemented")


# ----------------------------------------------------------------------
# Exact-k moments via CDF subtraction on a log-V grid.
# ----------------------------------------------------------------------

def exact_k_cumulants(V_k: np.ndarray, V_k1: np.ndarray) -> dict:
    """Moments of V and of v = V/⟨V⟩ - 1 for the exact-k distribution."""
    V_all = np.concatenate([V_k, V_k1])
    V_lo = max(np.percentile(V_all, 0.02), 1e-12)
    V_hi = np.percentile(V_all, 99.98)
    grid = np.geomspace(V_lo, V_hi, V_GRID_N)
    F_k  = np.searchsorted(np.sort(V_k),  grid, side="right") / V_k.size
    F_k1 = np.searchsorted(np.sort(V_k1), grid, side="right") / V_k1.size
    P = np.clip(F_k - F_k1, 0.0, None)
    norm = np.trapezoid(P, grid)
    if norm <= 0:
        return None
    # Raw volume moments
    m1 = float(np.trapezoid(grid * P, grid) / norm)
    m2 = float(np.trapezoid(grid**2 * P, grid) / norm)
    m3 = float(np.trapezoid(grid**3 * P, grid) / norm)
    m4 = float(np.trapezoid(grid**4 * P, grid) / norm)
    # Central moments of V
    c2 = m2 - m1**2
    c3 = m3 - 3 * m1 * m2 + 2 * m1**3
    c4 = m4 - 4 * m1 * m3 + 6 * m1**2 * m2 - 3 * m1**4
    # Cumulants of v = V/m1 - 1 (dimensionless):
    #   κ_2(v) = c2 / m1²
    #   κ_3(v) = c3 / m1³
    #   κ_4(v) = (c4 - 3 c2²) / m1⁴
    k2 = c2 / m1**2
    k3 = c3 / m1**3
    k4 = (c4 - 3 * c2**2) / m1**4
    return dict(V_mean=m1, V_c2=float(c2), V_c3=float(c3), V_c4=float(c4),
                kappa2=float(k2), kappa3=float(k3), kappa4=float(k4),
                sigma_v=float(np.sqrt(max(k2, 0))),
                S3_V=float(k3 / k2**2) if k2 > 0 else float("nan"))


# ----------------------------------------------------------------------
# Driver: walk every cached snapshot/kind and tabulate κ_m vs (σ_lin, γ).
# ----------------------------------------------------------------------

def main():
    rows = []
    col_of = {k: i for i, k in enumerate(K_LIST_RAW)}

    for N_PM in (256, 512):
        for a_pm in (0.03, 0.05, 0.07, 0.10):
            try:
                snap = load_sim(N_PM, "nocutoff", a=a_pm)
            except FileNotFoundError:
                continue
            if abs(snap.a - a_pm) > 0.005:
                continue
            pk_k_arr, pk_Pk_arr = reconstruct_pk(snap)
            stem = f"N{N_PM}_nocut_a{int(round(snap.a*100)):03d}"
            print(f"\n=== N={N_PM} a={snap.a:.2f} ===")
            print(f"{'kind':>4} {'k':>4} {'R':>6} {'σ_lin':>6} "
                  f"{'γ':>6}  |  {'κ_2':>10} {'κ_3':>10} {'κ_4':>11}  |  "
                  f"{'κ_2^ZA':>8} {'κ_3^ZA':>8} {'κ_4^ZA':>8}")
            for kind in ("R", "D"):
                path = os.path.join(CACHE_DIR, f"{stem}_{kind}.npz")
                if not os.path.exists(path):
                    continue
                V_all = np.load(path)["V"]
                for k in K_TARGETS:
                    V_k  = V_all[:, col_of[k]]
                    V_k1 = V_all[:, col_of[k + 1]]
                    mom  = exact_k_cumulants(V_k, V_k1)
                    if mom is None: continue
                    V_bar = mom["V_mean"]
                    R_lag = (3.0 * V_bar / (4.0 * np.pi))**(1.0 / 3.0)
                    gamma  = float(spectral_gamma(pk_k_arr, pk_Pk_arr, R_lag, dln=0.1))
                    s2_lin = float(smoothed_sigma2(pk_k_arr, pk_Pk_arr, R_lag))
                    k2_ZA = kappa_ZA(2, s2_lin)
                    k3_ZA = kappa_ZA(3, s2_lin)
                    k4_ZA = kappa_ZA(4, s2_lin)
                    row = dict(
                        N_PM=N_PM, a=float(snap.a), k=k, kind=kind,
                        R_lag=R_lag, sigma2_lin=s2_lin, gamma=gamma,
                        kappa2=mom["kappa2"], kappa3=mom["kappa3"],
                        kappa4=mom["kappa4"], V_mean=V_bar,
                        kappa2_ZA=k2_ZA, kappa3_ZA=k3_ZA, kappa4_ZA=k4_ZA,
                        dk2=mom["kappa2"] - k2_ZA,
                        dk3=mom["kappa3"] - k3_ZA,
                        dk4=mom["kappa4"] - k4_ZA,
                    )
                    rows.append(row)
                    print(f"{kind:>4} {k:>4} {R_lag:6.2f} {np.sqrt(s2_lin):6.3f} "
                          f"{gamma:+6.2f}  |  {mom['kappa2']:10.4e} "
                          f"{mom['kappa3']:+10.2e} {mom['kappa4']:+11.2e}  |  "
                          f"{k2_ZA:8.2e} {k3_ZA:+8.2e} {k4_ZA:+8.2e}")

    with open("docs/knn_cumulants.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved docs/knn_cumulants.json ({len(rows)} rows).")

    # --------  Diagnostic plots  --------
    # Restrict to the physically-reliable regime: R ≥ 1 Mpc/h (drops the
    # sub-grid shot-noise corner where kNN distances are < cell size).
    rows_f = [r for r in rows if r["R_lag"] >= 1.0]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # Row 1: κ_m^data / σ^{2m} vs σ (tree-scaling test).  Horizontal line
    # at the ZA leading coefficient (1, 2, 56/9) per m.
    for col, m in enumerate((2, 3, 4)):
        ax = axes[0, col]
        ZA_leading = {2: 1.0, 3: 2.0, 4: 56.0/9.0}[m]
        for kind, marker in (("R", "o"), ("D", "s")):
            subset = [r for r in rows_f if r["kind"] == kind]
            s = np.array([np.sqrt(r["sigma2_lin"]) for r in subset])
            ratio = np.array([r[f"kappa{m}"] / (r["sigma2_lin"]**m) for r in subset])
            ax.plot(s, ratio, marker, label=f"{kind}-kNN", alpha=0.7, ms=5)
        ax.axhline(ZA_leading, ls="--", color="black",
                   label=f"ZA leading = {ZA_leading:.3f}")
        ax.set_xlabel(r"$\sigma_{\rm lin}(R_k)$")
        ax.set_ylabel(fr"$\kappa_{m}^{{\rm data}} / \sigma^{{{2*m}}}$")
        ax.set_title(fr"κ_{m} tree-scaling test")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # Row 2: δκ_m ≡ κ_m^data − κ_m^ZA, scaled by σ^{2m+2}, vs γ.
    # At leading 2LPT: δκ_m / σ^{2m+2} = α_m^(0) + α_m^(γ) γ + O(σ²).
    for col, m in enumerate((2, 3, 4)):
        ax = axes[1, col]
        for kind, marker in (("R", "o"), ("D", "s")):
            subset = [r for r in rows_f if r["kind"] == kind]
            g = np.array([r["gamma"] for r in subset])
            resid_scaled = np.array(
                [r[f"dk{m}"] / (r["sigma2_lin"]**(m + 1)) for r in subset])
            ax.plot(g, resid_scaled, marker, alpha=0.7, ms=5, label=f"{kind}-kNN")
        ax.axhline(0, ls=":", color="grey")
        # expected 2LPT tree γ-slopes (dashed ref) — only κ_2 has γ tree piece
        if m == 2:
            xs = np.linspace(min(g.min(), -2.2), max(g.max(), -0.7), 100)
            ax.plot(xs, (2 * G2_EDS / 3) + (2 * G2_EDS / 15) * xs, "--",
                    color="tab:red", alpha=0.7,
                    label=r"2LPT tree: $2g_2/3 + (2g_2/15)\gamma$")
        if m == 3:
            xs = np.linspace(-2.2, -0.7, 100)
            ax.axhline(2 * G2_EDS, ls="--", color="tab:red", alpha=0.7,
                       label=r"2LPT tree: $2g_2$ (no $\gamma$ [Prop 4.2])")
        ax.set_xlabel(r"$\gamma(R_k)$")
        ax.set_ylabel(fr"$[\kappa_{m}^{{\rm data}} - \kappa_{m}^{{\rm ZA}}]"
                       fr"/\sigma^{{{2*(m+1)}}}$")
        ax.set_title(fr"2LPT residual — $\alpha_{m}^{{(0)}} + \alpha_{m}^{{(\gamma)}}\gamma$")
        ax.grid(alpha=0.3); ax.legend(fontsize=7)

    fig.suptitle("kNN cumulants — tree-scaling (top) and 2LPT γ-residuals (bottom)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig("docs/knn_cumulants_analysis.png", dpi=130)
    print("Saved docs/knn_cumulants_analysis.png")


if __name__ == "__main__":
    main()
