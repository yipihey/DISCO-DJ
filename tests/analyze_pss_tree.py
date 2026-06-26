"""Multi-scale PSS analysis with overlapping sampling + parent-child tree.

By default the analysis uses **overlapping** Kuhn-cube sampling at every
Lagrangian origin, giving N³ cubes per stride instead of (N/s)³. The
cubes overlap at their shared particle vertices, so samples are
correlated, but moments and cumulants still benefit from the ~s³ boost
in effective count — essential for reducing κ_3 noise at the large
strides where non-overlapping gave only 10²–10³ tets.

For each (sim, snapshot, stride) we record:

* Marginal cube moments (κ_2, κ_3, κ_4) of v = V_cube/⟨V⟩ − 1.
* 200-bin log-V histogram + negative-V count.
* Effective sample size n_eff = n_cubes / s³ (uncorrelated-cube proxy).
* Analytic σ²_lin(R_lag) at the Lagrangian radius.

Parent-child tree (stride_parent s, child stride s/2, both overlapping):

* Conservation rel. error ‖V_parent − Σ V_child‖ / |V_parent| (probes
  shell-crossing).
* ⟨V_c | V_parent⟩, Var(V_c | V_parent), skew(V_c | V_parent) binned
  into 40 log-V_parent bins.
* Mean pairwise child-child covariance ⟨ V_{c1} V_{c2} ⟩ - ⟨V_{c1}⟩⟨V_{c2}⟩
  averaged over all (c1, c2) pairs and all parents.

Outputs:
  - docs/pss_tree_cumulants.json   — marginal + conditional summaries
  - docs/pss_tree_sf_<n>.npz       — per-sim histograms + conditional binning
  - docs/pss_selfsim_sf.png        — SF self-similarity (restricted to
                                      reliable σ regime)
  - docs/pss_S3_collapse.png       — γ-slice overlay, A_1 diagnostic
"""

from __future__ import annotations

import json, os, time

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from discodj.analysis import (
    load_sim, kuhn_cube_volumes, parent_child_cube_pairs,
    smoothed_sigma2, spectral_gamma, reconstruct_pk,
)
from discodj.analysis.sim_store import SimSnapshot
from discodj import DiscoDJ


SF_N_VALUES  = (("-2.25", -2.25), ("-2.0", -2.0), ("-1.5", -1.5))
LCDM_CASES   = [(N_PM, a) for N_PM in (256,)
                 for a in (0.03, 0.05, 0.07, 0.10)]
# For N=512 we skip high strides due to memory (overlapping is 4 GB at stride>=2).
LCDM_CASES  += [(512, a) for a in (0.03, 0.05, 0.07, 0.10)]

STRIDES_SF   = [1, 2, 4, 8, 16, 32, 64]
STRIDES_LCDM256 = [2, 4, 8, 16, 32, 64]
STRIDES_LCDM512 = [2, 4, 8, 16]    # cap by memory
V_HIST_BINS  = 200
COND_BINS    = 40


# ----------------------------------------------------------------------
# Marginal and conditional summaries
# ----------------------------------------------------------------------


def marginal_moments(V_cube: np.ndarray, stride: int) -> dict:
    """Cumulants of v = V/⟨V⟩ − 1 and log-V histogram.  Effective sample
    size is ``n_cubes / s³`` (uncorrelated-cube proxy at overlap factor s³).
    """
    V = np.asarray(V_cube, dtype=np.float64).ravel()
    m = float(V.mean())
    v = V / m - 1.0
    c2 = float((v**2).mean())
    c3 = float((v**3).mean())
    c4 = float((v**4).mean()) - 3.0 * c2**2
    pos = V[V > 0]
    if pos.size == 0:
        V_edges = np.array([0.0, 1.0])
        hist = np.zeros(V_HIST_BINS, dtype=np.int64)
    else:
        V_lo = max(pos.min(), 1e-12)
        V_hi = pos.max()
        V_edges = np.geomspace(V_lo, V_hi, V_HIST_BINS + 1)
        hist, _ = np.histogram(pos, bins=V_edges)
    n_neg = int((V <= 0).sum())
    n_cubes = V.size
    n_eff = n_cubes / max(stride**3, 1)
    return dict(V_mean=m, kappa2=c2, kappa3=c3, kappa4=c4,
                sigma_v=float(np.sqrt(max(c2, 0))),
                S3_V=float(c3 / c2**2) if c2 > 0 else float("nan"),
                frac_neg=float(n_neg) / V.size,
                n_cubes=int(n_cubes), n_eff=float(n_eff),
                V_hist=hist.tolist(),
                V_edges=V_edges.tolist())


def conditional_moments(V_parent: np.ndarray, V_child: np.ndarray,
                          stride_parent: int, n_bins: int = COND_BINS) -> dict:
    parents, children = parent_child_cube_pairs(
        V_parent, V_child, stride_parent=stride_parent)
    sum_children = children.sum(axis=1)
    rel_err = float(np.median(np.abs(parents - sum_children) /
                               np.maximum(np.abs(parents), 1e-12)))
    pos = parents > 0
    P = parents[pos]; C = children[pos]
    if P.size == 0:
        return dict(conservation_rel_err=rel_err, n_bins=n_bins,
                    bin_edges=[], bin_centers=[],
                    mean_child=[], var_child=[], skew_child=[],
                    cross_cov_mean=float("nan"),
                    cross_cov_norm=float("nan"),
                    n_per_bin=[])
    edges = np.geomspace(P.min(), P.max(), n_bins + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    idx = np.searchsorted(edges, P, side="right") - 1
    idx = np.clip(idx, 0, n_bins - 1)
    mean_child = np.full(n_bins, np.nan)
    var_child  = np.full(n_bins, np.nan)
    skew_child = np.full(n_bins, np.nan)
    n_per_bin  = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        Cb = C[mask].ravel().astype(np.float64)
        m_b = float(Cb.mean())
        v2 = float(((Cb - m_b)**2).mean())
        v3 = float(((Cb - m_b)**3).mean())
        mean_child[b] = m_b
        var_child[b]  = v2
        skew_child[b] = v3 / v2**1.5 if v2 > 0 else np.nan
        n_per_bin[b]  = int(mask.sum())

    # child-child pairwise covariance (28 pairs, averaged).
    C_all = C.astype(np.float64)
    cross = 0.0; count = 0
    for a in range(8):
        for b in range(a + 1, 8):
            cov_ab = np.mean(C_all[:, a] * C_all[:, b]) \
                      - np.mean(C_all[:, a]) * np.mean(C_all[:, b])
            cross += cov_ab; count += 1
    cross /= max(count, 1)
    # normalise by ⟨V̄_child⟩² to make it dimensionless-ish
    V_bar_child = float(C_all.mean())
    cross_norm = cross / V_bar_child**2 if V_bar_child != 0 else float("nan")
    return dict(conservation_rel_err=rel_err, n_bins=n_bins,
                bin_edges=edges.tolist(),
                bin_centers=centers.tolist(),
                n_per_bin=n_per_bin.tolist(),
                mean_child=mean_child.tolist(),
                var_child=var_child.tolist(),
                skew_child=skew_child.tolist(),
                cross_cov_mean=float(cross),
                cross_cov_norm=float(cross_norm))


# ----------------------------------------------------------------------
# Scale-free processing
# ----------------------------------------------------------------------


def process_scale_free(n_tag: str, n_s: float):
    path = f"sims/N128_sf_n{n_tag}.npz"
    if not os.path.exists(path):
        return [], None
    data = np.load(path, allow_pickle=True)
    meta = json.loads(str(data["meta"]))
    sigma8 = float(meta["sigma8"])
    N = meta["N"]; L = meta["L"]
    gamma = -(n_s + 3.0)
    print(f"\n=== SF n={n_s}  γ={gamma:+.2f}  sigma8={sigma8:.4f}  "
          f"strides={STRIDES_SF} (overlapping) ===")

    rows = []
    hist_store = {}
    xs = data["x"]; as_ = data["a"]
    for i in range(len(as_)):
        snap = SimSnapshot(x=xs[i], v=np.zeros_like(xs[i]),
                            a=float(as_[i]), N=N, L=L, meta=meta)
        print(f" a={snap.a:.3f}")
        cubes = {}
        for s in STRIDES_SF:
            if s >= N: continue
            t0 = time.time()
            cubes[s] = kuhn_cube_volumes(snap, stride=s, overlapping=True)
            dt = time.time() - t0
            d_c = L / (N // s)
            R_lag = (d_c**3 * 3/(4*np.pi))**(1/3)
            s2_lin = sigma8**2 * (8.0 / R_lag)**(n_s + 3.0) * snap.a**2
            mom = marginal_moments(cubes[s], s)
            row = dict(sim=f"sf_n{n_tag}", n_s=n_s, gamma=gamma,
                       a=float(snap.a), stride=s, N_PM=N, L=L,
                       R_lag=R_lag, sigma2_lin=float(s2_lin),
                       sigma_lin=float(np.sqrt(max(s2_lin, 0))),
                       dt_compute=float(dt),
                       **{k: v for k, v in mom.items()
                          if k not in ("V_hist", "V_edges")})
            rows.append(row)
            hist_store.setdefault(s, []).append(
                (float(snap.a), np.asarray(mom["V_edges"]),
                 np.asarray(mom["V_hist"])))
            print(f"   stride={s:>3}  n_cubes={mom['n_cubes']:>10}  "
                  f"n_eff={mom['n_eff']:>9.0f}  R={R_lag:6.3f}  "
                  f"σ_lin={np.sqrt(s2_lin):6.3f}  σ_v={mom['sigma_v']:.4f}  "
                  f"S₃^V={mom['S3_V']:+.3f}  ({dt:.1f}s)")
        # Conditional over all consecutive stride pairs (parent, child=parent/2).
        for sp in STRIDES_SF:
            sc = sp // 2
            if sc == 0 or sc not in cubes or sp not in cubes:
                continue
            t0 = time.time()
            cond = conditional_moments(cubes[sp], cubes[sc], stride_parent=sp)
            dt = time.time() - t0
            print(f"   cond parent={sp:>2} child={sc:>2}  "
                  f"cons_med_rel_err={cond['conservation_rel_err']:.2e}  "
                  f"cross_cov_norm={cond['cross_cov_norm']:+.3e}  ({dt:.1f}s)")
            for r in rows:
                if r["sim"] == f"sf_n{n_tag}" and r["a"] == float(snap.a) \
                        and r["stride"] == sp:
                    r["conditional_child_stride"] = sc
                    r["conditional_cross_cov_norm"] = cond["cross_cov_norm"]
                    r["conditional_rel_err"] = cond["conservation_rel_err"]
                    break
    return rows, hist_store


# ----------------------------------------------------------------------
# LCDM processing (marginal only)
# ----------------------------------------------------------------------


def process_lcdm(N_PM: int, a_pm: float):
    try:
        snap = load_sim(N_PM, "nocutoff", a=a_pm)
    except FileNotFoundError:
        return []
    if abs(snap.a - a_pm) > 0.005:
        return []
    pk_k, pk_Pk = reconstruct_pk(snap)
    dj_cosmo = (DiscoDJ(dim=3, res=32, boxsize=snap.L)
                 .with_timetables())._cosmo
    D_a = float(dj_cosmo.Dplus(snap.a) / dj_cosmo.Dplus(1.0))
    rows = []
    strides = STRIDES_LCDM256 if N_PM == 256 else STRIDES_LCDM512
    for s in strides:
        if s >= snap.N:
            continue
        t0 = time.time()
        V_cube = kuhn_cube_volumes(snap, stride=s, overlapping=True)
        dt = time.time() - t0
        d_c = snap.L / (snap.N // s)
        R_lag = (d_c**3 * 3/(4*np.pi))**(1/3)
        s2_z0  = float(smoothed_sigma2(pk_k, pk_Pk, R_lag))
        s2_phys = D_a**2 * s2_z0
        gamma  = float(spectral_gamma(pk_k, pk_Pk, R_lag, dln=0.1))
        mom = marginal_moments(V_cube, s)
        row = dict(sim="lcdm_nocut", n_s=float("nan"), gamma=gamma,
                   a=float(snap.a), stride=s, N_PM=snap.N, L=snap.L,
                   R_lag=R_lag, sigma2_lin=s2_phys,
                   sigma_lin=float(np.sqrt(max(s2_phys, 0))),
                   dt_compute=float(dt),
                   **{k: v for k, v in mom.items()
                      if k not in ("V_hist", "V_edges")})
        print(f"  LCDM N={N_PM} a={snap.a:.2f}  s={s:>3}  n_cubes={mom['n_cubes']:>10}  "
              f"n_eff={mom['n_eff']:>9.0f}  R={R_lag:6.2f}  "
              f"σ_phys={np.sqrt(s2_phys):6.3f}  γ={gamma:+.2f}  "
              f"σ_v={mom['sigma_v']:.4f}  S₃^V={mom['S3_V']:+.3f}  ({dt:.1f}s)")
        rows.append(row)
    return rows


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main():
    os.makedirs("docs", exist_ok=True)
    all_rows = []

    print("=== Scale-free (N=128, overlapping) ===")
    for n_tag, n_s in SF_N_VALUES:
        rows, hist_store = process_scale_free(n_tag, n_s)
        all_rows += rows
        if hist_store:
            out = {}
            for s, lst in hist_store.items():
                for a_val, edges, hist in lst:
                    out[f"s{s}_a{a_val:.3f}_edges"] = edges
                    out[f"s{s}_a{a_val:.3f}_hist"]  = hist
            np.savez_compressed(f"docs/pss_tree_sf_{n_tag}.npz", **out)
            print(f"Saved docs/pss_tree_sf_{n_tag}.npz")

    print("\n=== LCDM nocut low-a (overlapping) ===")
    for N_PM, a_pm in LCDM_CASES:
        all_rows += process_lcdm(N_PM, a_pm)

    with open("docs/pss_tree_cumulants.json", "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nSaved docs/pss_tree_cumulants.json ({len(all_rows)} rows).")

    # ----- Self-similarity per SF sim -----
    sf_rows = [r for r in all_rows if r["sim"].startswith("sf_")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, (n_tag, n_s) in zip(axes, SF_N_VALUES):
        rs = [r for r in sf_rows if abs(r["n_s"] - n_s) < 0.01]
        by_a = {}
        for r in rs:
            by_a.setdefault(r["a"], []).append(r)
        for a_val, pts in by_a.items():
            pts = sorted(pts, key=lambda r: r["stride"])
            ax.plot([r["sigma_lin"] for r in pts],
                     [r["S3_V"] for r in pts],
                     "o-", alpha=0.7, ms=6,
                     label=(f"a={a_val:.2f}" if ax is axes[0] else None))
        ax.set_xscale("log")
        ax.set_xlabel(r"$\sigma_{\rm lin}(R_{\rm lag})$")
        ax.set_ylabel(r"$S_3^V$"); ax.grid(alpha=0.3, which="both")
        ax.set_title(fr"SF $n={n_s:+.2f}$, γ={-(n_s+3):+.2f}")
        ax.axhline(8/7, ls=":", color="red", alpha=0.7,
                   label=r"theory tree: $S_3^V = 8/7$")
        ax.axhline(34/7, ls="--", color="tab:purple", alpha=0.4,
                   label=r"2LPT δ-tree: 34/7")
        ax.set_ylim(-2, 4)
        if ax is axes[0]: ax.legend(fontsize=8)
    fig.suptitle("PSS (overlapping) self-similarity — (snap, stride) per n",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig("docs/pss_selfsim_sf.png", dpi=130)
    print("Saved docs/pss_selfsim_sf.png")

    # ----- S_3^V collapse plot: γ-slice A_1 diagnostic -----
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"-2.25": "tab:blue", "-2.0": "tab:orange", "-1.5": "tab:red"}
    for n_tag, color in colors.items():
        rs = [r for r in sf_rows if abs(r["n_s"] - float(n_tag)) < 0.01]
        # Keep all strides where σ_lin ≤ 1.5 (PT edge)
        rs = [r for r in rs if 0.01 <= r["sigma_lin"] <= 1.5]
        x = [r["sigma_lin"]**2 for r in rs]
        y = [r["S3_V"] for r in rs]
        ax.plot(x, y, "o", color=color, alpha=0.7, ms=6,
                label=fr"SF γ={-(float(n_tag)+3):+.2f}")
    xs = np.linspace(0, 1.5, 100)
    ax.axhline(8/7, ls="--", color="black",
               label=r"theory tree $S_3^V = 8/7$")
    ax.plot(xs, 8/7 + (32/49) * xs, ":", color="green", alpha=0.5,
            label=r"$8/7 + (32/49)\,\sigma^2$  (A_0 = 32/49)")
    ax.set_xlabel(r"$\sigma_{\rm lin}^2$"); ax.set_ylabel(r"$S_3^V$")
    ax.set_title(r"$S_3^V(\sigma^2)$ across γ-slices — overlapping PSS")
    ax.set_ylim(-1.5, 4)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("docs/pss_S3_collapse.png", dpi=130)
    print("Saved docs/pss_S3_collapse.png")


if __name__ == "__main__":
    main()
