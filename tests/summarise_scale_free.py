"""Post-analysis of the scale-free kNN data for the theory agent.

Focus: clean, reliable subset and its S_3^V dependence on γ at matched σ.

Key filters:
  - σ_lin > 0.25 (above the Poisson/grid noise floor seen in the raw plots)
  - σ_lin < 1.0  (stay in PT-applicable regime)
  - κ_2 > 1 / k  (cosmic signal above the gamma(k) shot-noise baseline)

Outputs:
  - docs/scale_free_theory_summary.md   — pared-down markdown table + reading
  - docs/scale_free_cleaned.png         — plots restricted to the usable subset
"""

from __future__ import annotations

import json

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_rows():
    return json.load(open("docs/scale_free_cumulants.json"))


def filter_usable(rows):
    """Restrict to the cosmic-signal-dominated subset."""
    out = []
    for r in rows:
        if not (0.25 < r["sigma_lin"] < 1.0):
            continue
        if r["kappa2"] <= 1.0 / r["k"]:   # below gamma baseline
            continue
        out.append(r)
    return out


def write_markdown(rows_all, rows_use, path):
    lines = []
    lines.append("# Scale-free kNN cumulants — theory summary\n")
    lines.append(
        "Three EdS PM sims at 128³ with power-law P(k) ∝ k^n. 6 snapshots per "
        "sim (a ∈ {0.05, 0.1, 0.2, 0.35, 0.6, 1.0}). kNN paired-k + CDF "
        "subtraction at k ∈ {1..5, 8,9, 16,17, 32,33, 64,65, 128,129} for "
        "both R (random) and D (data) queries, 10⁶ queries each.\n"
    )
    lines.append("## γ setup\n")
    lines.append("| n    | γ (= -(n+3)) | sigma8 | σ²_lin(R=d, a=1) |")
    lines.append("| ---  | ---         | ---   | ---              |")
    lines.append("| -2.25 | -0.75       | 0.4585 | 1.00            |")
    lines.append("| -2.00 | -1.00       | 0.3536 | 1.00            |")
    lines.append("| -1.50 | -1.50       | 0.2102 | 1.00            |\n")

    lines.append("## Diagnostics\n")
    lines.append(
        "* Self-similarity **works** above σ_lin ≳ 0.25 in each sim — "
        "all (snap × k) at matched σ_lin within each n collapse to a single "
        "S_3^V(σ) curve (see `scale_free_selfsim.png`).\n"
        "* Below σ_lin ≈ 0.1 the estimator is dominated by Poisson + grid "
        "discreteness (shot noise). Useful regime: σ_lin ∈ [0.25, 1] and "
        "κ_2 > κ_2^Poisson = 1/k.\n"
        "* γ-slice separation across the three sims is **at the noise floor "
        "of the 128³ suite** — the three γ curves mostly overlap in the "
        "usable regime (σ_lin > 0.25). A modest γ-dependence is visible at "
        "high σ (see cleaned plot) but ≲ 0.2 in S_3^V.\n"
    )

    lines.append(f"## Usable subset ({len(rows_use)}/{len(rows_all)} rows, filtered)\n")
    cols = ["n_tag", "gamma_true", "a", "kind", "k", "R_shell_exact",
            "sigma_lin", "kappa2", "kappa3", "kappa4", "S3_V"]
    fmt = ["{}", "{:+.2f}", "{:.2f}", "{}", "{}", "{:.2f}", "{:.3f}",
           "{:.3e}", "{:+.2e}", "{:+.2e}", "{:+.2f}"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for r in sorted(rows_use, key=lambda r: (r["n_tag"], r["a"], r["kind"], r["k"])):
        lines.append("| " + " | ".join(f.format(r[c]) for c, f in zip(cols, fmt)) + " |")

    lines.append("")
    lines.append("## Per-n, large-σ anchor values (σ_lin ≈ 0.3–0.5)\n")
    lines.append("Mean S_3^V in the σ_lin ∈ [0.3, 0.5] band:\n")
    lines.append("| n    | γ     | R-kNN S₃^V  | D-kNN S₃^V |")
    lines.append("| --- | ---  | ---         | ---        |")
    for n_tag in ("-2.25", "-2.0", "-1.5"):
        sub = [r for r in rows_use if r["n_tag"] == n_tag and 0.3 <= r["sigma_lin"] <= 0.5]
        r_vals = [r["S3_V"] for r in sub if r["kind"] == "R"]
        d_vals = [r["S3_V"] for r in sub if r["kind"] == "D"]
        r_mean = np.mean(r_vals) if r_vals else float("nan")
        d_mean = np.mean(d_vals) if d_vals else float("nan")
        r_std  = np.std(r_vals) if r_vals else float("nan")
        d_std  = np.std(d_vals) if d_vals else float("nan")
        gamma_val = -(float(n_tag) + 3.0)
        lines.append(f"| {n_tag} | {gamma_val:+.2f} | {r_mean:+.2f} ± {r_std:.2f} "
                     f"({len(r_vals)} pts) | {d_mean:+.2f} ± {d_std:.2f} ({len(d_vals)} pts) |")

    lines.append("")
    lines.append("## Notation\n")
    lines.append(
        "* **R_shell_exact** ≡ (k · V_per_particle · 3/(4π))^(1/3) — Lagrangian "
        "radius enclosing k particles at the mean density. Exact, no "
        "estimator needed.\n"
        "* **σ_lin** ≡ sqrt(σ²_lin(R, a)) with σ²_lin(R, a) = sigma8² · "
        "(8/R)^(n+3) · a² for this EdS power-law setup. Values below 1 mean "
        "the standard Bernardeau-Fry PT expansion should apply.\n"
        "* **γ_true = −(n+3)** is exact and known per sim (no measurement "
        "uncertainty). R_lag varies within each sim but γ is constant.\n"
        "* **κ_m** = cumulants of v = V/⟨V⟩_k − 1 via CDF-subtraction on the "
        "paired (V_k, V_{k+1}) arrays.\n"
        "* **S_3^V ≡ κ_3 / κ_2²**. Gamma(k) baseline = 2; 2LPT tree EdS ≈ 34/7 "
        "≈ 4.86.\n"
    )

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {path}")


def plot_cleaned(rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"-2.25": "tab:blue", "-2.0": "tab:orange", "-1.5": "tab:red"}
    markers = {"R": "o", "D": "s"}
    for ax, kind in zip(axes, ("R", "D")):
        for n_tag in ("-2.25", "-2.0", "-1.5"):
            pts = [r for r in rows if r["n_tag"] == n_tag and r["kind"] == kind]
            x = [r["sigma_lin"] for r in pts]
            y = [r["S3_V"] for r in pts]
            n_val = float(n_tag)
            ax.plot(x, y, markers[kind], color=colors[n_tag], alpha=0.75, ms=6,
                    label=fr"$n={n_val:+.2f}$, $\gamma={-(n_val+3):+.2f}$")
        ax.axhline(2, ls=":", color="grey", alpha=0.5, label=r"$\Gamma(k)$: $S_3^V = 2$")
        ax.axhline(34/7, ls="--", color="tab:purple", alpha=0.3,
                   label="2LPT EdS: 34/7")
        ax.set_xscale("log")
        ax.set_xlabel(r"$\sigma_{\rm lin}$")
        ax.set_ylabel(r"$S_3^V$")
        ax.set_title(f"{kind}-kNN (usable subset: $\\sigma_{{\\rm lin}} \\in [0.25, 1]$, $\\kappa_2 > 1/k$)")
        ax.grid(alpha=0.3, which="both")
        ax.set_ylim(-0.5, 4)
        ax.legend(fontsize=8)
    fig.suptitle("Scale-free γ-slice separation (cleaned subset)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")


def main():
    rows_all = load_rows()
    rows_use = filter_usable(rows_all)
    print(f"Rows: total {len(rows_all)}, usable {len(rows_use)}")
    write_markdown(rows_all, rows_use, "docs/scale_free_theory_summary.md")
    plot_cleaned(rows_use, "docs/scale_free_cleaned.png")


if __name__ == "__main__":
    main()
