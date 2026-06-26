"""Phase-space-sheet (PSS) multi-scale cumulant analysis.

For each PM snapshot (LCDM low-a + N=512 + scale-free), compute the
signed-volume distribution of Kuhn tets at a range of strides and
tabulate (σ_v, S_3^V, κ_3, κ_4) per stride. This delivers a Poisson-
noise-free and grid-aliasing-free counterpart to the kNN measurement.

Output: ``docs/pss_cumulants.json`` with all (sim, snap, stride) rows
and ``docs/pss_selfsim.png``, ``docs/pss_kNN_overlay.png``.
"""

from __future__ import annotations

import json, os, time
from math import factorial

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from discodj.analysis import (
    load_sim, kuhn_tet_volumes,
    smoothed_sigma2, spectral_gamma, reconstruct_pk,
)
from discodj import DiscoDJ


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

# For LCDM N=256 and N=512 suites
LCDM_STRIDES = [2, 4, 8, 16, 32]   # skip stride=1 (large memory, R_lag sub-cell)

# For scale-free N=128 suite
SF_STRIDES   = [1, 2, 4, 8, 16]

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def pss_cumulants(V: np.ndarray) -> dict:
    """Cumulants of v = V / V̄ − 1 for a flat signed-volume array."""
    V = np.asarray(V, dtype=np.float64)
    m = float(V.mean())
    v = V / m - 1.0
    c2 = float((v**2).mean())
    c3 = float((v**3).mean())
    c4 = float((v**4).mean()) - 3.0 * c2**2
    return dict(V_mean=m, kappa2=c2, kappa3=c3, kappa4=c4,
                sigma_v=float(np.sqrt(max(c2, 0))),
                S3_V=float(c3 / c2**2) if c2 > 0 else float("nan"),
                frac_neg=float((V < 0).mean()))


def process_snap_lcdm(snap, strides=LCDM_STRIDES):
    """For an LCDM snap, compute PSS cumulants at each stride."""
    pk_k, pk_Pk = reconstruct_pk(snap)
    N = snap.N; L = snap.L
    # Physical D(a) from DiscoDJ cosmo (LCDM normalised at z=0)
    dj = (DiscoDJ(dim=3, res=32, boxsize=L)
          .with_timetables())._cosmo
    D_a = float(dj.Dplus(float(snap.a)) / dj.Dplus(1.0))

    rows = []
    for s in strides:
        if N % s != 0:
            continue
        N_c = N // s
        d_c = L / N_c
        V_lag = d_c**3 / 6.0
        R_lag = (V_lag * 3.0 / (4.0 * np.pi))**(1.0/3.0)
        t0 = time.time()
        V = kuhn_tet_volumes(snap, stride=s)
        dt_compute = time.time() - t0
        mom = pss_cumulants(V)
        gamma = float(spectral_gamma(pk_k, pk_Pk, R_lag, dln=0.1))
        s2_z0  = float(smoothed_sigma2(pk_k, pk_Pk, R_lag))
        s2_phys = D_a**2 * s2_z0
        rows.append(dict(
            sim="lcdm_nocut", N_PM=N, a=float(snap.a), stride=s,
            n_tet=int(V.size), dt_compute=dt_compute,
            R_lag=R_lag, V_lag=V_lag, D_a=D_a,
            sigma2_z0_at_R=s2_z0, sigma2_phys=s2_phys,
            sigma_phys=float(np.sqrt(max(s2_phys, 0))),
            gamma=gamma, n_eff=-3.0 - gamma,
            **mom,
        ))
        print(f"  N={N} a={snap.a:.2f} s={s:>3}  n={V.size:>10}  "
              f"R={R_lag:6.2f}  σ_phys={np.sqrt(s2_phys):6.3f}  "
              f"γ={gamma:+.2f}  σ_v={mom['sigma_v']:.4f}  S₃^V={mom['S3_V']:+.2f}  "
              f"({dt_compute:.1f}s)")
    return rows


def process_snap_scale_free(snap, n_s: float, sigma8: float, strides=SF_STRIDES):
    """For a scale-free snap, compute PSS cumulants at each stride.

    γ = -(n+3) exactly. σ²_lin(R, a) = sigma8² · (8/R)^(n+3) · a² (EdS).
    """
    N = snap.N; L = snap.L
    gamma = -(n_s + 3.0)
    rows = []
    for s in strides:
        if N % s != 0:
            continue
        N_c = N // s
        d_c = L / N_c
        V_lag = d_c**3 / 6.0
        R_lag = (V_lag * 3.0 / (4.0 * np.pi))**(1.0/3.0)
        # Analytic σ²_lin(R, a) for power-law P(k)
        s2_lin = sigma8**2 * (8.0 / R_lag)**(n_s + 3.0) * float(snap.a)**2
        t0 = time.time()
        V = kuhn_tet_volumes(snap, stride=s)
        dt_compute = time.time() - t0
        mom = pss_cumulants(V)
        rows.append(dict(
            sim=f"sf_n{n_s:+.2f}", N_PM=N, a=float(snap.a), n_s=float(n_s),
            stride=s, n_tet=int(V.size), dt_compute=dt_compute,
            R_lag=R_lag, V_lag=V_lag,
            sigma2_lin=float(s2_lin),
            sigma_lin=float(np.sqrt(max(s2_lin, 0))),
            gamma=float(gamma), n_eff=float(n_s),
            **mom,
        ))
        print(f"  SF n={n_s:+.2f} a={snap.a:.2f} s={s:>3}  R={R_lag:6.3f}  "
              f"σ_lin={np.sqrt(s2_lin):6.3f}  σ_v={mom['sigma_v']:.4f}  "
              f"S₃^V={mom['S3_V']:+.2f}  ({dt_compute:.1f}s)")
    return rows


def main():
    rows_lcdm, rows_sf = [], []

    # --- LCDM low-a (N=256, 512) ---
    print("=== LCDM nocut low-a ===")
    for N_PM in (256, 512):
        for a_pm in (0.03, 0.05, 0.07, 0.10):
            try:
                snap = load_sim(N_PM, "nocutoff", a=a_pm)
            except FileNotFoundError:
                continue
            if abs(snap.a - a_pm) > 0.005:
                continue
            rows_lcdm += process_snap_lcdm(snap)

    # --- Scale-free N=128 ---
    print("\n=== Scale-free ===")
    import json as _json
    for n_tag, n_s in (("-2.25", -2.25), ("-2.0", -2.0), ("-1.5", -1.5)):
        path = f"sims/N128_sf_n{n_tag}.npz"
        if not os.path.exists(path):
            print(f"[skip] {path} not found"); continue
        data = np.load(path, allow_pickle=True)
        meta = _json.loads(str(data["meta"]))
        sigma8 = float(meta["sigma8"])
        xs = data["x"]; as_ = data["a"]
        # Build a SimSnapshot-like object per snap
        from discodj.analysis import SimSnapshot
        for i in range(len(as_)):
            snap = SimSnapshot(x=xs[i], v=np.zeros_like(xs[i]),
                                a=float(as_[i]), N=meta["N"], L=meta["L"],
                                meta=meta)
            rows_sf += process_snap_scale_free(snap, n_s, sigma8)

    all_rows = rows_lcdm + rows_sf
    with open("docs/pss_cumulants.json", "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nSaved docs/pss_cumulants.json ({len(all_rows)} rows).")

    # --- Self-similarity plot for scale-free ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, (n_tag, n_s) in zip(axes, (("-2.25", -2.25), ("-2.0", -2.0), ("-1.5", -1.5))):
        rs = [r for r in rows_sf if r["n_s"] == n_s]
        by_a = {}
        for r in rs:
            by_a.setdefault(r["a"], []).append(r)
        for a_val, pts in by_a.items():
            pts = sorted(pts, key=lambda r: r["stride"])
            x = [r["sigma_lin"] for r in pts]
            y = [r["S3_V"] for r in pts]
            ax.plot(x, y, "o-", alpha=0.6, ms=6,
                    label=(f"a={a_val:.2f}" if ax is axes[0] else None))
        ax.set_xscale("log")
        ax.set_xlabel(r"$\sigma_{\rm lin}(R_{\rm lag})$")
        ax.set_ylabel(r"$S_3^V$"); ax.grid(alpha=0.3, which="both")
        ax.set_title(fr"SF $n={n_s:+.2f}$, γ$={-(n_s+3):+.2f}$")
        ax.axhline(2, ls=":", color="red", alpha=0.5)
        ax.axhline(34/7, ls="--", color="tab:purple", alpha=0.4)
        if ax is axes[0]: ax.legend(fontsize=8)
    fig.suptitle("PSS self-similarity — (snap, stride) should form one curve per n", fontsize=12)
    fig.tight_layout()
    fig.savefig("docs/pss_selfsim.png", dpi=130)
    print("Saved docs/pss_selfsim.png")

    # --- γ-slice overlay (scale-free) + LCDM overlay ---
    fig, ax = plt.subplots(figsize=(9, 6))
    sf_colors = {"-2.25": "tab:blue", "-2.0": "tab:orange", "-1.5": "tab:red"}
    for n_tag, color in sf_colors.items():
        rs = [r for r in rows_sf if abs(r["n_s"] - float(n_tag)) < 0.01]
        x = [r["sigma_lin"] for r in rs]; y = [r["S3_V"] for r in rs]
        n_v = float(n_tag)
        ax.plot(x, y, "o", color=color, alpha=0.7, ms=6,
                label=fr"SF $n={n_v:+.2f}$, γ={-(n_v+3):+.2f}")
    # LCDM grey
    x_l = [r["sigma_phys"] for r in rows_lcdm]
    y_l = [r["S3_V"] for r in rows_lcdm]
    ax.plot(x_l, y_l, "x", color="grey", alpha=0.6, ms=5, label="LCDM (multi-scale, strides)")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\sigma_{\rm lin}$"); ax.set_ylabel(r"$S_3^V = \kappa_3/\kappa_2^2$")
    ax.axhline(2, ls=":", color="grey", alpha=0.5, label=r"$\Gamma$: $S_3^V = 2$")
    ax.axhline(34/7, ls="--", color="tab:purple", alpha=0.4,
               label="2LPT EdS tree 34/7")
    ax.set_title("PSS tet volumes — S_3^V vs σ_lin (multi-scale, multi-γ)")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("docs/pss_gamma_slice.png", dpi=130)
    print("Saved docs/pss_gamma_slice.png")


if __name__ == "__main__":
    main()
