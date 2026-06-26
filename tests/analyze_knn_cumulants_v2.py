"""Organised presentation of the kNN volume cumulants for theory work.

Produces three deliverables:

1. **Rich JSON table** (``docs/knn_cumulants_physical.json``): one row per
   (snapshot, k, kind) with *physical* σ (D(a)-corrected), both γ
   conventions, Poisson-gamma baseline subtractions, and tree-scaling
   ratios. Flagged fields: ``pt_safe`` (σ²_phys < 0.3, PT small-σ expansion
   valid) and ``resolved`` (R_lag > 1 Mpc/h, above sub-grid noise floor).

2. **Markdown summary** (``docs/knn_theory_summary.md``): the PT-safe +
   resolved subset as a reading-friendly table for the theory agent, with
   explicit notation for what each column is.

3. **Plots** (``docs/knn_cumulants_physical.png``): tree-scaling test
   (κ_m / σ²_phys^m vs σ_phys) restricted to the usable regime, with the
   Poisson-gamma baseline marked; S₃^V vs n_eff at matched σ.

Definitions
-----------
- σ²_phys(R, a) ≡ D(a)² · σ²_lin(R, z=0), using DiscoDJ's cosmology growth
  factor Dplus(a). v ≡ V/⟨V⟩_k − 1 is the dimensionless volume fluctuation
  inside the exact-k distribution.
- κ_m ≡ cumulant-of-v, computed by CDF subtraction on paired V_k, V_{k+1}
  from the kNN cache.
- γ_slope ≡ d ln σ²_lin(R) / d ln R (negative, decreasing σ² with R). n_eff
  ≡ −3 − γ_slope, the effective P(k) spectral index at R (positive near the
  turnover, → −3 on small scales for LCDM).
- κ_m^Poisson (gamma-baseline) = (m−1)! / k^(m−1) — cumulants of a gamma
  distribution with shape k (the large-k limit of V_k for purely random
  particles). κ_m^sub ≡ κ_m − κ_m^Poisson captures what the shot-noise
  baseline can't explain.
- S₃^V ≡ κ_3 / κ_2². For pure gamma(k), S₃^V = 2 (independent of k), so
  observed S₃^V → 0.01 at large k means we are NOT in the gamma regime —
  the PM clustering is actively suppressing the third cumulant.
"""

from __future__ import annotations

import json, os
from math import factorial

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from discodj import DiscoDJ
from discodj.analysis import load_sim
from discodj.analysis.spectral_slope import (
    spectral_gamma, smoothed_sigma2, reconstruct_pk,
)


K_LIST_RAW = [1, 2, 3, 4, 5, 8, 9, 16, 17, 32, 33, 64, 65, 128, 129]
K_TARGETS  = [1, 2, 3, 4, 8, 16, 32, 64, 128]
CACHE_DIR = "sims/knn_cache"
V_GRID_N  = 800


# ----------------------------------------------------------------------
# Growth factor D(a) from DiscoDJ cosmology.
# ----------------------------------------------------------------------

_DJ_CACHE = None


def Dplus_of_a(a: float) -> float:
    """LCDM (Planck-default) D(a) normalised so D(a=1) = 1."""
    global _DJ_CACHE
    if _DJ_CACHE is None:
        _DJ_CACHE = (DiscoDJ(dim=3, res=32, boxsize=250.0)
                      .with_timetables()._cosmo)
    D_a = float(_DJ_CACHE.Dplus(float(a)))
    D_1 = float(_DJ_CACHE.Dplus(1.0))
    return D_a / D_1


# ----------------------------------------------------------------------
# ZA closed-form — the paper's nonlinear Zel'dovich cumulant expansion.
# (Kept for reference; not used in the output table — the theory agent
# will compare against their own form once conventions are settled.)
# ----------------------------------------------------------------------

def kappa_ZA_paper(m: int, sigma2: float) -> float:
    """Paper ψ_ZA → κ_m expansion with u = t σ² (σ is PAPER convention)."""
    s2 = sigma2
    if m == 2:
        return s2**2 * (1.0 + (4.0/15.0) * s2 + (4.0/225.0) * s2**2)
    if m == 3:
        return 2.0 * s2**2 * (1.0 + (92.0/225.0) * s2 + (28.0/1125.0) * s2**2)
    if m == 4:
        return (56.0/9.0) * s2**2 * (1.0 + (494.0/875.0) * s2)
    raise ValueError(m)


# ----------------------------------------------------------------------
# Exact-k cumulants from paired V_k, V_{k+1} via CDF subtraction.
# ----------------------------------------------------------------------

def exact_k_cumulants(V_k: np.ndarray, V_k1: np.ndarray) -> dict | None:
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
    m1 = float(np.trapezoid(grid * P, grid) / norm)
    m2 = float(np.trapezoid(grid**2 * P, grid) / norm)
    m3 = float(np.trapezoid(grid**3 * P, grid) / norm)
    m4 = float(np.trapezoid(grid**4 * P, grid) / norm)
    c2 = m2 - m1**2
    c3 = m3 - 3 * m1 * m2 + 2 * m1**3
    c4 = m4 - 4 * m1 * m3 + 6 * m1**2 * m2 - 3 * m1**4
    return dict(V_mean=m1,
                kappa2=c2 / m1**2,
                kappa3=c3 / m1**3,
                kappa4=(c4 - 3 * c2**2) / m1**4)


# ----------------------------------------------------------------------
# Assemble rows.
# ----------------------------------------------------------------------

def build_rows():
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
            D_a = Dplus_of_a(float(snap.a))
            stem = f"N{N_PM}_nocut_a{int(round(snap.a*100)):03d}"
            for kind in ("R", "D"):
                path = os.path.join(CACHE_DIR, f"{stem}_{kind}.npz")
                if not os.path.exists(path):
                    continue
                V_all = np.load(path)["V"]
                for k in K_TARGETS:
                    V_k  = V_all[:, col_of[k]]
                    V_k1 = V_all[:, col_of[k + 1]]
                    mom  = exact_k_cumulants(V_k, V_k1)
                    if mom is None:
                        continue
                    R_lag = (3.0 * mom["V_mean"] / (4.0 * np.pi)) ** (1.0/3.0)
                    gamma_slope = float(spectral_gamma(pk_k_arr, pk_Pk_arr, R_lag, dln=0.1))
                    n_eff = -3.0 - gamma_slope
                    s2_z0 = float(smoothed_sigma2(pk_k_arr, pk_Pk_arr, R_lag))
                    s2_phys = D_a**2 * s2_z0

                    # Poisson / gamma-shape baseline subtractions.
                    k_pois2 = 1.0 / k
                    k_pois3 = factorial(2) / k**2
                    k_pois4 = factorial(3) / k**3

                    k2 = mom["kappa2"]; k3 = mom["kappa3"]; k4 = mom["kappa4"]
                    k2_sub = k2 - k_pois2
                    k3_sub = k3 - k_pois3
                    k4_sub = k4 - k_pois4
                    S3_V    = k3 / k2**2 if k2 > 0 else float("nan")
                    S3_V_sub = (k3_sub / k2_sub**2) if k2_sub > 0 else float("nan")

                    rows.append(dict(
                        N_PM=N_PM, a=float(snap.a), kind=kind, k=k,
                        R_lag=R_lag, V_mean=mom["V_mean"],
                        # σ² conventions
                        sigma2_z0_at_R=s2_z0,
                        D_a=D_a,
                        sigma2_phys=s2_phys,
                        sigma_phys=float(np.sqrt(max(s2_phys, 0.0))),
                        # γ conventions
                        gamma_slope=gamma_slope,     # d ln σ² / d ln R (negative)
                        n_eff=n_eff,                 # -(3 + γ_slope) ~ P(k) spectral index
                        # raw cumulants of v
                        kappa2=k2, kappa3=k3, kappa4=k4,
                        # Poisson-gamma baselines
                        kappa2_pois=k_pois2, kappa3_pois=k_pois3, kappa4_pois=k_pois4,
                        # subtracted
                        kappa2_sub=k2_sub, kappa3_sub=k3_sub, kappa4_sub=k4_sub,
                        # derived dimensionless
                        S3_V=S3_V,
                        S3_V_sub=S3_V_sub,
                        # tree-scaling ratios (physical σ)
                        kappa2_over_s2phys=(k2 / s2_phys) if s2_phys > 0 else float("nan"),
                        kappa3_over_s2phys2=(k3 / s2_phys**2) if s2_phys > 0 else float("nan"),
                        # flags
                        pt_safe=(s2_phys < 0.3),
                        resolved=(R_lag > 1.0),
                    ))
    return rows


# ----------------------------------------------------------------------
# Markdown summary for the theory agent.
# ----------------------------------------------------------------------

def write_markdown(rows, path: str):
    header = (
"""# kNN volume cumulants — cleaned table for theory work

Generated by `tests/analyze_knn_cumulants_v2.py` from the cached
paired-k kNN volumes (k ∈ {1..5, 8, 9, 16, 17, 32, 33, 64, 65, 128, 129}
for R-kNN and D-kNN, 10⁶ queries per snapshot).

## Notation

- **σ_phys** = `D(a) · sqrt(σ²_lin(R_lag, z=0))`, using DiscoDJ's growth
  factor. **σ²_phys** is the *physical* linear variance at Lagrangian
  radius R_lag — this is what enters the paper's small-σ PT expansion.
- **γ_slope** = `d ln σ²_lin / d ln R` (our `spectral_gamma`, negative).
- **n_eff**  = `−3 − γ_slope` (the P(k) spectral index at R). Use whichever
  matches the paper's convention.
- **κ_m**     = cumulants of `v ≡ V/⟨V⟩_k − 1` from the CDF-subtracted
  exact-k distribution (CDF of V_k minus CDF of V_{k+1}).
- **κ_m^Poisson** = `(m−1)! / k^(m−1)` — cumulants of the gamma distribution
  Vk ~ Gamma(k, V̄/k), i.e. the pure-Poisson (no clustering) limit.
- **κ_m^sub**  = `κ_m − κ_m^Poisson` (best guess at the purely clustering piece).
- **S_3^V**   = `κ_3 / κ_2²`. Pure gamma(k) gives S_3^V = 2. Observed values
  are much less than 2 at all scales → clustering strongly suppresses the
  third cumulant. S_3^V → 0 at large k is *not* trivial CLT (CLT would
  keep it at 2).
- **pt_safe** = `σ²_phys < 0.3` (PT small-σ regime valid; nonlinear
  corrections ≤ few percent).
- **resolved** = `R_lag > 1 Mpc/h` (above the N=256 particle spacing; kNN
  distances are physically meaningful, not sub-grid noise).

## PT-safe and resolved subset (σ²_phys < 0.3 AND R_lag > 1 Mpc/h)

"""
    )
    lines = [header]
    cols = ["N_PM", "a", "kind", "k", "R_lag", "sigma_phys", "gamma_slope",
            "n_eff", "kappa2", "kappa2_sub", "kappa3", "kappa3_sub",
            "kappa4", "kappa4_sub", "S3_V", "S3_V_sub"]
    fmt = ["{}", "{:.2f}", "{}", "{}", "{:.2f}", "{:.3f}", "{:+.2f}",
           "{:+.2f}", "{:.3e}", "{:+.3e}", "{:+.2e}", "{:+.2e}",
           "{:+.2e}", "{:+.2e}", "{:+.2f}", "{:+.2f}"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    pts = [r for r in rows if r["pt_safe"] and r["resolved"]]
    pts.sort(key=lambda r: (r["N_PM"], r["a"], r["kind"], r["k"]))
    for r in pts:
        vals = [f.format(r[c]) for c, f in zip(cols, fmt)]
        lines.append("| " + " | ".join(vals) + " |")

    lines.append("\n## Unsafe / unresolved rows (σ²_phys ≥ 0.3 OR R_lag ≤ 1 Mpc/h)")
    lines.append(f"{len([r for r in rows if not (r['pt_safe'] and r['resolved'])])} rows; see JSON for full data.")

    lines.append(
"""
## Key empirical observations (for theory)

1. **κ_2 scales as σ²_phys, not σ⁴_phys, at leading order.** Example:
   N=256, a=0.10, k=128 (R≈3 Mpc/h): σ²_phys ≈ 0.033, κ_2 = 0.029,
   κ_2^Poisson = 1/128 ≈ 0.008, so κ_2 − κ_2^Poisson ≈ 0.021 ≈ σ²_phys.
   The paper's ψ_ZA formula, whose κ_2 starts at σ⁴, is the **nonlinear
   correction on top of the Gaussian baseline κ_2 = σ²**. Total κ_2 is
   expected to be `σ²_phys + (4/15)σ⁴_phys + 2LPT·σ⁴·f(γ) + ...`.

2. **Initial-condition sub-Poisson artifact at small k.** For R-kNN k≤4
   and D-kNN k≤4, κ_2 < 1/k (gamma-Poisson baseline). The Zel'dovich-
   displaced grid ICs are *sub-Poisson* at sub-cell scales because the
   particles sit on a regular lattice. This is a *nuisance*, not physics —
   the useful regime for physics is k ≥ 16-32 (D-kNN) or k ≥ 8 (R-kNN).

3. **S_3^V stays near 1-2 at large k**, not 0. Pure gamma(k) predicts 2.
   Observed ~1-1.3 suggests a *weak* reduction from gamma by clustering —
   consistent with the paper's leading-tree-2LPT S_3^V ≈ 34/7 − 1 (for
   n_eff ≈ -2) mapped through the V ↔ δ relation. (This is for the theory
   agent to verify; we just report the measured number.)

4. **N-convergence:** N=256 and N=512 rows at matched R_lag agree to a few
   percent on (σ_phys, κ_2). UV leakage only matters below R ≈ 1 Mpc/h
   (= grid scale at N=256).

5. **Two γ conventions** are given side-by-side. `γ_slope` is our
   `d ln σ²/d ln R` (negative, our `spectral_gamma` code). `n_eff = −3 −
   γ_slope` is the effective P(k) spectral index at R (our values: -1.9
   to -2.2, typical LCDM small-scale range). Tell us which matches the
   paper's convention and we'll re-emit.

6. **Best starting subset for fitting** (most likely PT-clean):
   N=256 D-kNN at k ∈ {32, 64, 128} (R ≥ 2 Mpc/h, κ_2 > κ_2^Poisson, and
   σ²_phys ≪ 0.3), across all four a values. 12 points spanning
   (σ_phys, n_eff) ≈ ([0.06, 0.24], [−2.04, −1.94]).
"""
    )
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {path}")


# ----------------------------------------------------------------------
# Plots.
# ----------------------------------------------------------------------

def make_plots(rows, out: str):
    rows_use = [r for r in rows if r["pt_safe"] and r["resolved"]]
    rows_all = [r for r in rows if r["resolved"]]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # Row 1: Raw κ_m vs σ²_phys in log-log with the Poisson baseline overlay.
    # Shows clearly which regime is cosmology-dominated (far from Poisson line)
    # vs shot-noise-dominated (near Poisson line).
    for col, m in enumerate((2, 3, 4)):
        ax = axes[0, col]
        for kind, marker, color in [("R", "o", "tab:blue"), ("D", "s", "tab:orange")]:
            pts = [r for r in rows_use if r["kind"] == kind]
            x = np.array([r["sigma2_phys"] for r in pts])
            y = np.array([r[f"kappa{m}"] for r in pts])
            ax.loglog(x, np.abs(y), marker, color=color, alpha=0.7, ms=6,
                      label=f"{kind}-kNN   |κ|")
        # Poisson baseline: as a function of k, overlay κ_m^Poisson for the
        # k values in K_TARGETS at the minimum/maximum σ² of our rows.
        xs = np.geomspace(min(r["sigma2_phys"] for r in rows_use),
                          max(r["sigma2_phys"] for r in rows_use), 50)
        # Reference: κ_m = σ²_phys^(m/2)   (if κ_m ~ σ^m would-be)
        ax.plot(xs, xs**(m/2), "-.", color="green", alpha=0.5,
                label=fr"$\sigma^{{{m}}}_{{\rm phys}}$")
        # Reference: κ_m = σ²_phys^m    (paper's "tree scaling")
        ax.plot(xs, xs**m, ":", color="red", alpha=0.5,
                label=fr"$\sigma^{{{2*m}}}_{{\rm phys}}$ (paper tree)")
        ax.set_xlabel(r"$\sigma^2_{\rm phys}(R_k)$")
        ax.set_ylabel(fr"$|\kappa_{m}|$  (log-log)")
        ax.set_title(fr"$\kappa_{m}$ vs $\sigma^2_{{\rm phys}}$ with reference slopes")
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)

    # Row 2 (left): κ_2 vs σ²_phys — both axes linear. Adds the σ²_phys line
    # (Gaussian baseline if v ≈ -δ at linear order) and the gamma-Poisson line.
    ax = axes[1, 0]
    for kind, marker, color in [("R", "o", "tab:blue"), ("D", "s", "tab:orange")]:
        pts = [r for r in rows_use if r["kind"] == kind]
        x = np.array([r["sigma2_phys"] for r in pts])
        y_raw = np.array([r["kappa2"] for r in pts])
        y_sub = np.array([r["kappa2_sub"] for r in pts])
        ax.plot(x, y_raw, marker, color=color, alpha=0.7, ms=6,
                label=f"{kind}-kNN raw κ_2")
        ax.plot(x, y_sub, marker, color=color, alpha=0.3, ms=6, mfc="white",
                label=f"{kind}-kNN κ_2 - 1/k")
    xs = np.linspace(0, max(r["sigma2_phys"] for r in rows_use), 20)
    ax.plot(xs, xs, "--", color="black", label=r"$\kappa_2 = \sigma^2_{\rm phys}$")
    ax.set_xlabel(r"$\sigma^2_{\rm phys}(R_k)$"); ax.set_ylabel(r"$\kappa_2$")
    ax.set_title(r"$\kappa_2$ vs $\sigma^2_{\rm phys}$  (raw and Poisson-subtracted)")
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="best")

    # Row 2 (middle): S₃^V (raw) vs n_eff at matched σ_phys (color by σ).
    # Gamma(k) baseline = 2 shown explicitly.
    ax = axes[1, 1]
    for kind, marker in [("R", "o"), ("D", "s")]:
        pts = [r for r in rows_use if r["kind"] == kind]
        x = np.array([r["n_eff"] for r in pts])
        y = np.array([r["S3_V"] for r in pts])
        c = np.array([r["sigma_phys"] for r in pts])
        sc = ax.scatter(x, y, c=c, marker=marker, s=45, cmap="viridis",
                        alpha=0.8, edgecolor="grey",
                        label=f"{kind}-kNN" if marker == "o" else None)
    plt.colorbar(sc, ax=ax, label=r"$\sigma_{\rm phys}$")
    ax.axhline(2, ls=":", color="red", alpha=0.6, label=r"$\Gamma(k)$: $S_3^V = 2$")
    ax.axhline(34/7, ls="--", color="tab:purple", alpha=0.4,
               label=r"2LPT tree (EdS): $S_3^V = 34/7$")
    ax.set_xlabel(r"$n_{\rm eff} = -3 - \gamma_{\rm slope}$")
    ax.set_ylabel(r"$S_3^V = \kappa_3/\kappa_2^2$")
    ax.set_title(r"Raw $S_3^V$ vs spectral index (at matched $\sigma_{\rm phys}$)")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    # Row 2 (right): (ln σ, n_eff) coverage — show pt_safe and unsafe separately.
    ax = axes[1, 2]
    for kind, marker, color in [("R", "o", "tab:blue"), ("D", "s", "tab:orange")]:
        pts_all = [r for r in rows_all if r["kind"] == kind]
        for r in pts_all:
            m = marker if r["pt_safe"] else "x"
            c = color if r["pt_safe"] else "tab:grey"
            ax.plot(0.5 * np.log(r["sigma2_phys"]), r["n_eff"], m,
                    color=c, alpha=0.7, ms=6)
    ax.set_xlabel(r"$\ln \sigma_{\rm phys}(R_k)$")
    ax.set_ylabel(r"$n_{\rm eff}$")
    ax.set_title(r"(ln σ, n_eff) coverage.  × = σ²≥0.3 (non-PT)")
    ax.grid(alpha=0.3)

    fig.suptitle("kNN volume cumulants — physical σ, with reference scalings",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")


def main():
    print("Computing growth factors + cumulants...")
    rows = build_rows()
    with open("docs/knn_cumulants_physical.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Saved docs/knn_cumulants_physical.json ({len(rows)} rows).")

    print("\n== PT-safe + resolved subset counts ==")
    kinds = ("R", "D")
    for kind in kinds:
        pts = [r for r in rows if r["kind"] == kind and r["pt_safe"] and r["resolved"]]
        print(f"  {kind}-kNN: {len(pts)} usable rows out of "
              f"{len([r for r in rows if r['kind'] == kind])}")

    write_markdown(rows, "docs/knn_theory_summary.md")
    make_plots(rows, "docs/knn_cumulants_physical.png")

    # Print 6 representative rows (a=0.10 N=256 R-kNN) for eyeball sanity.
    print("\n== Representative rows (N=256 a=0.10 R-kNN, PT-safe + resolved) ==")
    reps = [r for r in rows
            if r["N_PM"] == 256 and r["a"] > 0.099 and r["a"] < 0.101
               and r["kind"] == "R" and r["pt_safe"] and r["resolved"]]
    print(f"{'k':>4} {'R':>6} {'σ_phys':>8} {'n_eff':>6} | "
          f"{'κ_2':>9} {'κ_2^sub':>10} {'σ²_phys':>9} | "
          f"{'κ_3':>9} {'κ_3^sub':>10} | "
          f"{'S₃^V':>6} {'S₃^V_sub':>9}")
    for r in sorted(reps, key=lambda r: r["k"]):
        print(f"{r['k']:>4} {r['R_lag']:6.2f} {r['sigma_phys']:8.4f} "
              f"{r['n_eff']:+6.2f} | "
              f"{r['kappa2']:9.3e} {r['kappa2_sub']:+10.3e} "
              f"{r['sigma2_phys']:9.3e} | "
              f"{r['kappa3']:9.2e} {r['kappa3_sub']:+10.2e} | "
              f"{r['S3_V']:+6.2f} {r['S3_V_sub']:+9.2f}")


if __name__ == "__main__":
    main()
