"""Joint β(ln σ, γ) fit on v5 — combined N=256 + N=512 nocut snapshots.

The single-variable β(ln σ) fit differs by ~2σ between N=256 and N=512
(see refit_beta_N512.py). If the apparent N-dependence is really the
spectral slope γ at the Lagrangian smoothing scale, the two-parameter
    β = β₀ + β_σ · ln σ + β_γ · γ
should absorb the offset with β_γ ≠ 0 and collapse both suites onto one law.

Reuses β_best values already computed in:
  - docs/beta_nocut_v5.json (N=256, 4 snapshots)
  - docs/beta_nocut_N512.json (N=512, 4 snapshots)

Only new computation: γ per snap via spectral_slope.gamma_for_snapshot.
"""

import json

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from discodj.analysis import load_sim
from discodj.analysis.spectral_slope import gamma_for_snapshot


def gather():
    """Collect (ln σ, γ, β_best) from the two saved JSONs."""
    data = {}
    for N_PM, path in [(256, "docs/beta_nocut_v5.json"),
                       (512, "docs/beta_nocut_N512.json")]:
        d = json.load(open(path))
        rows = []
        for r in d["rows"]:
            snap = load_sim(N_PM, "nocutoff", a=r["a"])
            s2, R, g = gamma_for_snapshot(snap, sigma2_trG=r["sigma2"])
            rows.append(dict(N_PM=N_PM, a=r["a"], sigma2=r["sigma2"],
                              gamma=g, R_eff=R,
                              beta_best=r["beta_best_S3"]))
            print(f"  N={N_PM} a={r['a']:.2f}  σ²={s2:.4f}  γ={g:+.3f}  "
                  f"β_best={r['beta_best_S3']:+.4f}")
        data[N_PM] = rows
    return data


def fit_1d(rows):
    ln_sig = np.array([0.5 * np.log(r["sigma2"]) for r in rows])
    y = np.array([r["beta_best"] for r in rows])
    A = np.column_stack([np.ones_like(ln_sig), ln_sig])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ c
    rms = float(np.sqrt(np.mean(resid**2)))
    return float(c[0]), float(c[1]), rms, resid


def fit_2d(rows):
    ln_sig = np.array([0.5 * np.log(r["sigma2"]) for r in rows])
    gamma = np.array([r["gamma"] for r in rows])
    y = np.array([r["beta_best"] for r in rows])
    A = np.column_stack([np.ones_like(ln_sig), ln_sig, gamma])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ c
    rms = float(np.sqrt(np.mean(resid**2)))
    # covariance (σ² ≈ rms² · (AᵀA)⁻¹)
    cov = np.linalg.inv(A.T @ A) * rms**2
    se = np.sqrt(np.diag(cov))
    return dict(b0=float(c[0]), b_sigma=float(c[1]), b_gamma=float(c[2]),
                se_b0=float(se[0]), se_sigma=float(se[1]), se_gamma=float(se[2]),
                rms=rms, resid=resid.tolist())


def main():
    data = gather()
    all_rows = data[256] + data[512]

    print("\n=== 1-parameter fits per suite ===")
    for N_PM, rows in data.items():
        b0, bs, rms, _ = fit_1d(rows)
        print(f"N_PM={N_PM}  β = {b0:+.4f} + ({bs:+.4f})·ln σ   rms = {rms:.4f}")

    print("\n=== 1-parameter fit on combined N=256 + N=512 ===")
    b0, bs, rms, res = fit_1d(all_rows)
    print(f"Combined  β = {b0:+.4f} + ({bs:+.4f})·ln σ   rms = {rms:.4f}")
    print("Per-snap residuals (combined 1-param fit):")
    for r, dr in zip(all_rows, res):
        print(f"  N={r['N_PM']} a={r['a']:.2f}  σ²={r['sigma2']:.4f}  "
              f"γ={r['gamma']:+.3f}  Δβ = {dr:+.4f}")

    print("\n=== 2-parameter fit β = β₀ + β_σ·ln σ + β_γ·γ  (combined) ===")
    f = fit_2d(all_rows)
    print(f"β = {f['b0']:+.4f}(±{f['se_b0']:.4f}) + "
          f"({f['b_sigma']:+.4f}±{f['se_sigma']:.4f})·ln σ + "
          f"({f['b_gamma']:+.4f}±{f['se_gamma']:.4f})·γ")
    print(f"rms = {f['rms']:.4f}  (vs 1-param rms = {rms:.4f})")
    print(f"β_γ vs 0: {f['b_gamma']/f['se_gamma']:+.2f}σ")
    print("Per-snap residuals (2-param fit):")
    for r, dr in zip(all_rows, f["resid"]):
        print(f"  N={r['N_PM']} a={r['a']:.2f}  σ²={r['sigma2']:.4f}  "
              f"γ={r['gamma']:+.3f}  Δβ = {dr:+.4f}")

    # Plot: β vs ln σ, colour by γ, show 1-param fits per N and 2-param combined law
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, fit_kind in zip(axes, ("1-param", "2-param")):
        for rows, color, lbl in [(data[256], "tab:blue", "N=256"),
                                 (data[512], "tab:red", "N=512")]:
            ln_s = np.array([0.5 * np.log(r["sigma2"]) for r in rows])
            y = np.array([r["beta_best"] for r in rows])
            ax.plot(ln_s, y, "o", ms=9, color=color, label=lbl)
        xs = np.linspace(-2.3, -0.6, 100)
        if fit_kind == "1-param":
            # per-suite lines
            for rows, color in [(data[256], "tab:blue"), (data[512], "tab:red")]:
                b0_s, bs_s, _, _ = fit_1d(rows)
                ax.plot(xs, b0_s + bs_s * xs, ":", color=color, alpha=0.7)
            # combined
            ax.plot(xs, b0 + bs * xs, "--", color="black",
                    label=f"combined: β = {b0:+.3f} + ({bs:+.3f})·ln σ")
            ax.set_title(f"1-param fit  rms = {rms:.4f}")
        else:
            # 2-param: slice at mean γ
            gm = float(np.mean([r["gamma"] for r in all_rows]))
            ax.plot(xs, f["b0"] + f["b_sigma"] * xs + f["b_gamma"] * gm, "--",
                    color="black",
                    label=fr"2-param at $\bar\gamma={gm:+.2f}$:")
            # γ extremes to show spread
            g_lo = float(min(r["gamma"] for r in all_rows))
            g_hi = float(max(r["gamma"] for r in all_rows))
            ax.fill_between(xs,
                            f["b0"] + f["b_sigma"] * xs + f["b_gamma"] * g_lo,
                            f["b0"] + f["b_sigma"] * xs + f["b_gamma"] * g_hi,
                            color="grey", alpha=0.2,
                            label=fr"γ ∈ [{g_lo:+.2f}, {g_hi:+.2f}] band")
            ax.set_title(f"2-param fit  rms = {f['rms']:.4f}")
        ax.set_xlabel(r"$\ln \sigma$")
        ax.set_ylabel(r"$\beta_{\rm nl, best}$")
        ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="lower right")

    fig.suptitle("v5 cascade β(ln σ) — 1-param vs 2-param (combined N=256+512 nocut)")
    fig.tight_layout()
    fig.savefig("docs/beta_sigma_gamma_v5.png", dpi=130)
    print("\nSaved docs/beta_sigma_gamma_v5.png")

    with open("docs/beta_sigma_gamma_v5.json", "w") as f_out:
        json.dump(dict(combined_1param=dict(b0=b0, b_sigma=bs, rms=rms),
                       combined_2param=f,
                       data=all_rows), f_out, indent=2)
    print("Saved docs/beta_sigma_gamma_v5.json")


if __name__ == "__main__":
    main()
