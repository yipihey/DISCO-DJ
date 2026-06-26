"""Multi-scale volume statistics via paired-k kNN + CDF subtraction.

For each PM snapshot: place ~10^6 random queries (R-kNN) AND take 10^6
random particles (D-kNN). For k ∈ {1,2,3,4,5, 8,9, 16,17, 32,33, 64,65,
128,129} find the k-th nearest particle (with periodic BCs) and record
V_k = (4π/3) r_k³.

Using k-and-k+1 pairs, the CDF subtraction
    P_exact_k(V) ≡ Prob(exactly k particles in sphere of volume V around
                   a query) = CDF_k(V) − CDF_{k+1}(V)
gives the exact-k mass-scale distribution at each target k ∈ {1,2,3,4,
8,16,32,64,128}. Moments thereof:
    ⟨V⟩_k      = ∫ V · P_exact_k(V) dV / ∫ P_exact_k(V) dV
    σ_V²(M_k) = ⟨V²⟩_k − ⟨V⟩_k²
    κ_3(M_k)   = ⟨(V − ⟨V⟩_k)³⟩_k
    S_3^V(M_k) = κ_3(M_k) / σ_V(M_k)⁴

The raw per-query V_k arrays are persisted compressed (one npz per snap
per kNN type) so moment re-analyses need not re-run kNN.

R-kNN: random query points in [0,L)³.
D-kNN: queries are random particles; the self-distance is skipped (query
        the (k+1)-th nearest including self and take that as the k-th
        other particle).
"""

from __future__ import annotations

import json, os, time

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from discodj.analysis import load_sim
from discodj.analysis.spectral_slope import (
    spectral_gamma, smoothed_sigma2, reconstruct_pk,
)


K_LIST_RAW = [1, 2, 3, 4, 5, 8, 9, 16, 17, 32, 33, 64, 65, 128, 129]
K_TARGETS  = [1, 2, 3, 4, 8, 16, 32, 64, 128]
N_QUERIES  = 1_000_000
OUT_DIR = "sims/knn_cache"


def _build_tree(x, L):
    return cKDTree(np.asarray(x, dtype=np.float64), boxsize=L + 1e-6,
                   leafsize=32, compact_nodes=True, balanced_tree=True)


def rknn_volumes(tree, L, k_max, n_queries, seed):
    rng = np.random.default_rng(seed)
    q = rng.uniform(0, L, size=(n_queries, 3)).astype(np.float32)
    dists, _ = tree.query(q, k=k_max, workers=-1)
    return (4.0 / 3.0) * np.pi * dists**3           # (n_queries, k_max)


def dknn_volumes(tree, x, k_max, n_queries, seed):
    """Query from random particles; return V_1..V_{k_max} of OTHER particles."""
    rng = np.random.default_rng(seed + 1)
    N = x.shape[0]
    idx = rng.choice(N, size=n_queries, replace=False)
    q = np.asarray(x[idx], dtype=np.float64)
    # k+1 so we can drop the self (distance 0)
    dists, _ = tree.query(q, k=k_max + 1, workers=-1)
    return (4.0 / 3.0) * np.pi * dists[:, 1:]**3     # (n_queries, k_max)


def exact_k_moments(V_k: np.ndarray, V_k1: np.ndarray):
    """Moments of the volume distribution containing exactly k particles.

    Uses CDF subtraction: we build empirical CDFs F_k and F_{k+1} on a
    common log-V grid; P_exact_k(V) = F_k(V) − F_{k+1}(V) is the
    probability at V of exactly k particles. Normalise and compute
    ⟨V⟩, σ_V², κ_3, S_3^V, and the standard δ_V skewness.
    """
    # Common grid: percentile-based edges over the combined range.
    V_all = np.concatenate([V_k, V_k1])
    V_lo = max(np.percentile(V_all, 0.01), 1e-12)
    V_hi = np.percentile(V_all, 99.99)
    grid = np.geomspace(V_lo, V_hi, 400)

    F_k  = np.searchsorted(np.sort(V_k),  grid, side="right") / V_k.size
    F_k1 = np.searchsorted(np.sort(V_k1), grid, side="right") / V_k1.size
    P = np.clip(F_k - F_k1, 0.0, None)
    # Use midpoint values V at each bin.
    V_mid = grid
    norm = np.trapezoid(P, V_mid)
    if norm <= 0:
        return None
    # Moments — integrate using the trapezoid rule on P(V) as a measure.
    m1 = float(np.trapezoid(V_mid * P, V_mid) / norm)
    m2 = float(np.trapezoid((V_mid - m1)**2 * P, V_mid) / norm)
    m3 = float(np.trapezoid((V_mid - m1)**3 * P, V_mid) / norm)
    sigma_V = float(np.sqrt(max(m2, 0)))
    S3_V    = float(m3 / m2**2) if m2 > 0 else float("nan")
    # δ_V = V/V̄ − 1 with V̄ = m1 (the exact-k mean volume)
    sigma_d = sigma_V / m1
    S3_std  = float(m3 / m2**1.5) if m2 > 0 else float("nan")   # κ_3 / σ_V³
    return dict(V_mean=m1, V_var=m2, V_k3=m3, sigma_V=sigma_V,
                S3_V=S3_V, sigma_d=float(sigma_d), S3_std=S3_std,
                P_V=P.tolist(), V_grid=V_mid.tolist())


def process_snapshot(snap, cache_stem: str):
    """Returns a list of rows — one per (k_target, kind)."""
    pk_k_arr, pk_Pk_arr = reconstruct_pk(snap)
    N_part = snap.x.shape[0]
    V_box = snap.L**3
    V_per_particle = V_box / N_part
    tree = _build_tree(snap.x, snap.L)
    k_max = max(K_LIST_RAW)

    rows = []
    for kind in ("R", "D"):
        path_cache = os.path.join(OUT_DIR, f"{cache_stem}_{kind}.npz")
        if os.path.exists(path_cache):
            print(f"  [cache] {path_cache}")
            V_all = np.load(path_cache)["V"]
        else:
            print(f"  [compute] {kind}-kNN (k_max={k_max}, queries={N_QUERIES})")
            t0 = time.time()
            if kind == "R":
                V_all = rknn_volumes(tree, snap.L, k_max, N_QUERIES, seed=42)
            else:
                V_all = dknn_volumes(tree, snap.x, k_max, N_QUERIES, seed=43)
            # Keep only the k values in K_LIST_RAW to save disk.
            keep_idx = [k - 1 for k in K_LIST_RAW]
            V_all = V_all[:, keep_idx].astype(np.float32)
            os.makedirs(OUT_DIR, exist_ok=True)
            np.savez_compressed(path_cache, V=V_all, k_list=np.asarray(K_LIST_RAW))
            print(f"    done in {time.time()-t0:.0f}s, "
                  f"size {os.path.getsize(path_cache)/1e6:.0f} MB")
        # V_all columns correspond to K_LIST_RAW order.
        col_of = {k: i for i, k in enumerate(K_LIST_RAW)}

        for k in K_TARGETS:
            V_k  = V_all[:, col_of[k]]
            V_k1 = V_all[:, col_of[k + 1]]
            mom = exact_k_moments(V_k, V_k1)
            if mom is None: continue
            V_bar = mom["V_mean"]
            R_lag = (3.0 * V_bar / (4.0 * np.pi))**(1.0/3.0)
            gamma = float(spectral_gamma(pk_k_arr, pk_Pk_arr, R_lag, dln=0.1))
            s2_lin = float(smoothed_sigma2(pk_k_arr, pk_Pk_arr, R_lag))
            mom_lite = {k: mom[k] for k in
                        ("V_mean", "V_var", "V_k3", "sigma_V", "S3_V",
                         "sigma_d", "S3_std")}
            rows.append(dict(
                N_PM=snap.N, a=float(snap.a), k=k, kind=kind,
                V_per_particle=V_per_particle,
                R_lag=R_lag, gamma=gamma, sigma2_lin_at_R=s2_lin,
                **mom_lite,
            ))
            print(f"  {kind}-kNN k={k:>3}  R={R_lag:6.2f}  "
                  f"σ_δV={mom['sigma_d']:.3f}  S₃^V={mom['S3_V']:+.2f}  "
                  f"γ={gamma:+.2f}")
    return rows


def main():
    t_all = time.time()
    rows = []
    for N_PM in (256, 512):
        for a_pm in (0.03, 0.05, 0.07, 0.10):
            try:
                snap = load_sim(N_PM, "nocutoff", a=a_pm)
            except FileNotFoundError:
                continue
            if abs(snap.a - a_pm) > 0.005:
                continue
            stem = f"N{N_PM}_nocut_a{int(round(snap.a*100)):03d}"
            print(f"\n=== N={N_PM} a={snap.a:.2f}  N_part={snap.x.shape[0]:,}")
            rows += process_snapshot(snap, stem)

    with open("docs/knn_multiscale_moments.json", "w") as fo:
        json.dump(rows, fo, indent=2)
    print(f"\nSaved docs/knn_multiscale_moments.json ({len(rows)} rows).")

    # Coverage plot (one row for R-kNN, one for D-kNN)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for row_idx, kind in enumerate(("R", "D")):
        rs = [r for r in rows if r["kind"] == kind]
        for ax, xk, ylabel in [(axes[row_idx, 0], "gamma", r"$\gamma(R_{\rm lag})$"),
                                (axes[row_idx, 1], "S3_V", r"$S_3^V$")]:
            for N_PM, color in [(256, "tab:blue"), (512, "tab:red")]:
                by_a = {}
                for r in [r for r in rs if r["N_PM"] == N_PM]:
                    by_a.setdefault(r["a"], []).append(r)
                for a, pts in by_a.items():
                    pts = sorted(pts, key=lambda r: r["k"])
                    ln_s = [0.5 * np.log(r["sigma_d"]**2) for r in pts]
                    y = [r[xk] for r in pts]
                    ax.plot(ln_s, y, "-", color=color, alpha=0.3)
                    ax.plot(ln_s, y, "o", color=color, ms=6, alpha=0.7,
                            label=(f"N={N_PM} a={a:.2f}"
                                   if xk == "gamma" and row_idx == 0 else None))
            ax.set_xlabel(r"$\ln\sigma_{\delta V}(M_k)$")
            ax.set_ylabel(f"{ylabel}  ({kind}-kNN)"); ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7, ncol=2)
    axes[0, 0].set_title("R-kNN γ-coverage")
    axes[0, 1].set_title("R-kNN S₃^V vs ln σ_{δV}")
    axes[1, 0].set_title("D-kNN γ-coverage")
    axes[1, 1].set_title("D-kNN S₃^V vs ln σ_{δV}")
    fig.tight_layout()
    fig.savefig("docs/knn_multiscale_coverage.png", dpi=130)
    print("Saved docs/knn_multiscale_coverage.png")
    print(f"Total wall time: {time.time()-t_all:.0f}s")


if __name__ == "__main__":
    main()
