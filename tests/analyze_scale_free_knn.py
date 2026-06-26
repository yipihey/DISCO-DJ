"""Post-process the scale-free suite: kNN paired-k + cumulants + self-similarity check.

For each (sim, snapshot, k, kind) shell:
  - V_k distribution via kNN (1e6 queries, both R and D)
  - cumulants κ_2, κ_3, κ_4 of v = V/⟨V⟩ - 1 via paired CDF subtraction
  - R_lag_shell = (k · V_per_particle · 3/(4π))^(1/3)  (exact — no density estimator needed)
  - σ²_lin(R, a) analytic from P(k) = A k^n:
        σ²_lin(R, a) = sigma8² · D(a)² · (8/R)^(n+3) · [window correction cancels]
    where D(a) = a for EdS.
  - γ exactly known: γ = -(n+3) per sim (constant across scales for power-law).

Outputs:
  - docs/scale_free_cumulants.json   — one row per (sim, snap, k, kind)
  - docs/scale_free_selfsim.png      — S_3^V vs σ_lin within each sim
                                        (should collapse to one curve for
                                        each n — self-similarity test)
  - docs/scale_free_gamma_slice.png  — S_3^V vs σ_lin across sims
                                        (γ-slice ordering test)
"""

from __future__ import annotations

import json, os, time
from math import factorial

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


K_LIST_RAW = [1, 2, 3, 4, 5, 8, 9, 16, 17, 32, 33, 64, 65, 128, 129]
K_TARGETS  = [1, 2, 3, 4, 8, 16, 32, 64, 128]
N_QUERIES  = 1_000_000
V_GRID_N   = 800
SIMS = [("-2.25", -2.25), ("-2.0", -2.0), ("-1.5", -1.5)]


def tree_build(x, L):
    return cKDTree(np.asarray(x, dtype=np.float64), boxsize=L + 1e-6,
                   leafsize=32, compact_nodes=True, balanced_tree=True)


def rknn_volumes(tree, L, k_max, n_queries, seed):
    rng = np.random.default_rng(seed)
    q = rng.uniform(0, L, size=(n_queries, 3)).astype(np.float32)
    dists, _ = tree.query(q, k=k_max, workers=-1)
    return (4.0/3.0) * np.pi * dists**3


def dknn_volumes(tree, x, k_max, n_queries, seed):
    rng = np.random.default_rng(seed + 1)
    N = x.shape[0]
    idx = rng.choice(N, size=n_queries, replace=False)
    q = np.asarray(x[idx], dtype=np.float64)
    dists, _ = tree.query(q, k=k_max + 1, workers=-1)
    return (4.0/3.0) * np.pi * dists[:, 1:]**3


def exact_k_cumulants(V_k, V_k1):
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
    return dict(V_mean=m1, kappa2=c2/m1**2, kappa3=c3/m1**3,
                kappa4=(c4 - 3*c2**2)/m1**4)


def sigma2_lin_power_law(R: float, a: float, n: float, sigma8: float) -> float:
    """σ²_lin(R, a) for P(k) = A k^n in EdS, with A fixed by sigma8 convention.

    Pure-scaling identity: σ²(R) = σ²(8) · 8^(n+3) / R^(n+3) = sigma8² · (8/R)^(n+3).
    Growth factor D(a) = a in EdS.
    """
    return sigma8**2 * (8.0 / R)**(n + 3.0) * a**2


def sigma8_for_unit_variance_at_d(n: float) -> float:
    return 8.0**(-(n + 3.0) / 2.0)


def process_sim(n_tag: str, n: float):
    path = f"sims/N128_sf_n{n_tag}.npz"
    if not os.path.exists(path):
        print(f"[skip] {path} not found"); return []
    data = np.load(path, allow_pickle=True)
    meta = json.loads(str(data["meta"]))
    N = meta["N"]; L = meta["L"]; sigma8 = meta["sigma8"]
    gamma_true = -(n + 3.0)
    print(f"\n=== n={n}  γ={gamma_true:+.2f}  sigma8={sigma8:.4f}  L={L} ===")

    rows = []
    xs = data["x"]          # (n_snap, N**3, 3)
    as_ = data["a"]
    col_of = {k: i for i, k in enumerate(K_LIST_RAW)}
    V_per_particle = L**3 / (N**3)

    for snap_idx, a_val in enumerate(as_):
        print(f"  a={float(a_val):.3f}")
        x = xs[snap_idx]
        tree = tree_build(x, L)
        for kind in ("R", "D"):
            if kind == "R":
                V_all = rknn_volumes(tree, L, max(K_LIST_RAW), N_QUERIES, seed=5000+snap_idx)
            else:
                V_all = dknn_volumes(tree, x, max(K_LIST_RAW), N_QUERIES, seed=6000+snap_idx)
            # Keep only K_LIST_RAW columns
            keep = [k - 1 for k in K_LIST_RAW]
            V_cache = V_all[:, keep].astype(np.float32)

            for k in K_TARGETS:
                V_k  = V_cache[:, col_of[k]]
                V_k1 = V_cache[:, col_of[k + 1]]
                mom = exact_k_cumulants(V_k, V_k1)
                if mom is None: continue
                R_shell_exact = (k * V_per_particle * 3.0 / (4.0 * np.pi))**(1.0/3.0)
                R_shell_meas  = (3.0 * mom["V_mean"] / (4.0 * np.pi))**(1.0/3.0)
                s2_lin = sigma2_lin_power_law(R_shell_exact, float(a_val), n, sigma8)
                k2_pois = 1.0 / k
                k3_pois = factorial(2) / k**2
                k4_pois = factorial(3) / k**3
                rows.append(dict(
                    sim_n=n, n_tag=n_tag, gamma_true=gamma_true, sigma8=sigma8,
                    a=float(a_val), k=k, kind=kind,
                    R_shell_exact=R_shell_exact, R_shell_meas=R_shell_meas,
                    V_mean=mom["V_mean"], V_bar=k*V_per_particle,
                    sigma2_lin=s2_lin,
                    sigma_lin=float(np.sqrt(s2_lin)),
                    kappa2=mom["kappa2"], kappa3=mom["kappa3"], kappa4=mom["kappa4"],
                    kappa2_sub=mom["kappa2"] - k2_pois,
                    kappa3_sub=mom["kappa3"] - k3_pois,
                    kappa4_sub=mom["kappa4"] - k4_pois,
                    S3_V=(mom["kappa3"] / mom["kappa2"]**2) if mom["kappa2"] > 0 else float("nan"),
                ))
    return rows


def plot_selfsim(rows, out: str):
    """Within each sim, S_3^V vs σ_lin should collapse across (snap × k)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, (n_tag, n) in zip(axes, SIMS):
        rs = [r for r in rows if r["n_tag"] == n_tag]
        for kind, marker, color in (("R", "o", "tab:blue"),
                                    ("D", "s", "tab:orange")):
            by_a = {}
            for r in [r for r in rs if r["kind"] == kind]:
                by_a.setdefault(r["a"], []).append(r)
            for a_val, pts in by_a.items():
                pts = sorted(pts, key=lambda r: r["k"])
                x = [r["sigma_lin"] for r in pts]
                y = [r["S3_V"] for r in pts]
                ax.plot(x, y, "-", color=color, alpha=0.25, lw=1)
                ax.plot(x, y, marker, color=color, alpha=0.7, ms=5,
                        label=f"{kind}-kNN a={a_val:.2f}" if a_val == 0.10 else None)
        ax.set_xscale("log")
        ax.set_xlabel(r"$\sigma_{\rm lin}(R_k, a)$")
        ax.set_ylabel(r"$S_3^V = \kappa_3/\kappa_2^2$")
        ax.set_title(fr"$n={n:+.2f}$, $\gamma={-(n+3):+.2f}$")
        ax.grid(alpha=0.3, which="both"); ax.axhline(2, ls=":", color="red", alpha=0.5)
        ax.axhline(34/7, ls="--", color="tab:purple", alpha=0.4)
        if ax is axes[0]:
            ax.legend(fontsize=7, loc="best")
    fig.suptitle(r"Self-similarity test — all (snap, k) should form one curve per $n$",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")


def plot_gamma_slice(rows, out: str):
    """Overlay all three sims' S_3^V(σ); spread at matched σ ↔ γ dependence."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"-2.25": "tab:blue", "-2.0": "tab:orange", "-1.5": "tab:red"}
    for ax_idx, kind in enumerate(("R", "D")):
        ax = axes[ax_idx]
        for n_tag, color in colors.items():
            rs = [r for r in rows if r["n_tag"] == n_tag and r["kind"] == kind]
            x = [r["sigma_lin"] for r in rs]
            y = [r["S3_V"] for r in rs]
            n_val = float(n_tag)
            ax.plot(x, y, "o", color=color, alpha=0.6, ms=5,
                    label=fr"$n={n_val:+.2f}$, $\gamma={-(n_val+3):+.2f}$")
        ax.set_xscale("log")
        ax.set_xlabel(r"$\sigma_{\rm lin}$"); ax.set_ylabel(r"$S_3^V$")
        ax.set_title(f"{kind}-kNN: γ-slice overlay")
        ax.grid(alpha=0.3, which="both")
        ax.axhline(2, ls=":", color="grey", alpha=0.5, label="gamma(k)=2")
        ax.axhline(34/7, ls="--", color="tab:purple", alpha=0.3,
                   label="2LPT EdS tree = 34/7")
        ax.legend(fontsize=7)
    fig.suptitle("γ-slice separation across scale-free sims", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")


def main():
    t0 = time.time()
    rows = []
    for n_tag, n in SIMS:
        rows += process_sim(n_tag, n)
    with open("docs/scale_free_cumulants.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved docs/scale_free_cumulants.json ({len(rows)} rows) in {time.time()-t0:.0f}s")

    if not rows:
        print("No data — nothing to plot."); return
    plot_selfsim(rows, "docs/scale_free_selfsim.png")
    plot_gamma_slice(rows, "docs/scale_free_gamma_slice.png")


if __name__ == "__main__":
    main()
