"""Tophat-smooth the stride-1 Jacobian cube field and compare to tet-window moments.

For each scale-free LPT snapshot, build the stride-1 V_cube field
(per-Lagrangian-cell signed volume) and FFT-convolve with a spherical
tophat of radius ``R_s`` matched to each stride-s cube:
    R_s = (s · d_grid) · (3/(4π))^{1/3}  ≈  0.620 · s · d_grid.

Moments κ_m of ``v_R = V_R / ⟨V_R⟩ − 1`` over all 128³ Lagrangian
smoothing centres. Output is parallel to ``pss_lpt_cumulants.json`` but
measures the tophat window (what theory directly predicts) rather than
the Kuhn-tet window.

Purpose: cross-check that ``A_1^tophat`` extracted from tophat moments
matches the theory agent's tree-level analytic rational. Any discrepancy
between tet-window S_3^V(R_s) and tophat-window S_3^V(R_s) at the same
R is a **window-function correction**, not a physics mismatch.
"""

from __future__ import annotations

import json, os, time

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax, jax.numpy as jnp
from functools import partial

from discodj import DiscoDJ
from discodj.analysis import (
    kuhn_cube_volumes_from_positions,
)
from discodj.analysis.sim_store import SimSnapshot


SF_N_VALUES = (("-2.25", -2.25), ("-2.0", -2.0), ("-1.5", -1.5))
A_TARGETS  = [0.02, 0.05, 0.10, 0.20, 0.35, 0.55, 0.85]
N_PART     = 128
BOXSIZE    = 128.0
SEED       = 1234
# Stride-equivalent radii to probe: R_s = (s · d_grid) · (3/4π)^{1/3}.
STRIDE_EQUIV = [2, 4, 8, 16]
LPT_ORDERS   = [1, 2, 3, 5]


# ----------------------------------------------------------------------
# LPT + positions → stride-1 V_cube field
# ----------------------------------------------------------------------


def sigma8_for_unit_variance_at_d(n: float) -> float:
    return 8.0 ** (-(n + 3.0) / 2.0)


def build_cosmo(n_s: float, sigma8: float) -> dict:
    return dict(Omega_c=1.0, Omega_b=0.0, Omega_k=0.0,
                h=1.0, n_s=float(n_s), sigma8=float(sigma8),
                w0=-1.0, wa=0.0)


def make_dj(n_s: float, sigma8: float, seed: int, n_max: int) -> DiscoDJ:
    return (DiscoDJ(dim=3, res=N_PART, boxsize=BOXSIZE,
                     cosmo=build_cosmo(n_s, sigma8))
            .with_timetables()
            .with_linear_ps(transfer_function="none")
            .with_ics(seed=seed, white_noise_space="fourier",
                      k_order_fourier="stable")
            .with_lpt(n_order=n_max))


def lpt_positions_mesh(dj: DiscoDJ, a: float, order: int) -> jnp.ndarray:
    """(N, N, N, 3) periodic-wrapped Eulerian positions, min-image unwrapped."""
    L = float(dj._boxsize)
    N = int(dj._res)
    x_flat = dj.evaluate_lpt_pos_at_a(float(a), n_order=order)
    # Build unwrapped positions using the original Lagrangian grid.
    x_mesh = np.asarray(x_flat).reshape(N, N, N, 3).astype(np.float32)
    ix, iy, iz = np.meshgrid(np.arange(N), np.arange(N), np.arange(N),
                              indexing="ij")
    h = L / N
    q = np.stack([ix * h, iy * h, iz * h], axis=-1).astype(np.float32)
    # Min-image wrap of displacement (matches analysis.sim_store.displacement_field).
    psi = x_mesh - q
    psi = psi - L * np.round(psi / L)
    return jnp.asarray(q + psi)


# ----------------------------------------------------------------------
# FFT tophat
# ----------------------------------------------------------------------


@partial(jax.jit, static_argnames=("N",))
def _tophat_convolve_jit(V_cube: jnp.ndarray, L: float, R: float, N: int):
    kx = 2 * jnp.pi * jnp.fft.fftfreq(N, d=L / N)
    ky = kx
    kz = 2 * jnp.pi * jnp.fft.rfftfreq(N, d=L / N)
    KX, KY, KZ = jnp.meshgrid(kx, ky, kz, indexing="ij")
    kmag = jnp.sqrt(KX**2 + KY**2 + KZ**2)
    kR = kmag * R
    small = kR < 1e-3
    x = jnp.where(small, 1e-3, kR)
    W = 3.0 * (jnp.sin(x) - x * jnp.cos(x)) / x**3
    W = jnp.where(small, 1.0, W)
    V_f = jnp.fft.rfftn(V_cube)
    V_R = jnp.fft.irfftn(V_f * W, s=(N, N, N))
    return V_R


def tophat_convolve(V_cube: np.ndarray, L: float, R: float) -> np.ndarray:
    N = V_cube.shape[0]
    return np.asarray(_tophat_convolve_jit(
        jnp.asarray(V_cube), float(L), float(R), int(N)))


# ----------------------------------------------------------------------
# Moment helper
# ----------------------------------------------------------------------


def moments(arr: np.ndarray) -> dict:
    x = np.asarray(arr, dtype=np.float64).ravel()
    m = float(x.mean())
    v = x / m - 1.0
    c2 = float((v**2).mean())
    c3 = float((v**3).mean())
    c4 = float((v**4).mean()) - 3.0 * c2**2
    return dict(V_mean=m, kappa2=c2, kappa3=c3, kappa4=c4,
                sigma_v=float(np.sqrt(max(c2, 0))),
                S3_V=float(c3 / c2**2) if c2 > 0 else float("nan"))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main():
    os.makedirs("docs", exist_ok=True)
    d_grid = BOXSIZE / N_PART
    R_values = {s: (s * d_grid) * (3.0 / (4.0 * np.pi)) ** (1.0/3.0)
                for s in STRIDE_EQUIV}
    print(f"Tophat radii (matched to stride-s cube volume):")
    for s, R in R_values.items():
        print(f"  s={s:>2}: R = {R:.3f}")

    rows = []
    for n_tag, n_s in SF_N_VALUES:
        sigma8 = sigma8_for_unit_variance_at_d(n_s)
        gamma_true = -(n_s + 3.0)
        print(f"\n=== SF n={n_s}  γ={gamma_true:+.2f} ===")
        dj = make_dj(n_s, sigma8, SEED, max(LPT_ORDERS))
        for a in A_TARGETS:
            for order in LPT_ORDERS:
                t0 = time.time()
                x_mesh = lpt_positions_mesh(dj, a, order)
                # Stride-1 per-cube V_cube field, shape (N, N, N).
                V_cube = np.asarray(kuhn_cube_volumes_from_positions(
                    x_mesh, float(BOXSIZE), stride=1, overlapping=True))
                dt_pss = time.time() - t0
                # Self-moment (stride-1, no smoothing) for reference.
                m0 = moments(V_cube)
                sigma2_lin_at_grid = sigma8**2 * (8.0 / ((d_grid**3)
                                        * 3 / (4 * np.pi)) ** (1/3))**(n_s + 3.0) * a**2
                # Sanity: sigma2_lin should be computed at the *grid-cube* R_lag.
                R_grid = (d_grid**3 * 3 / (4 * np.pi)) ** (1/3)
                sigma2_lin_grid = (sigma8**2 * (8.0 / R_grid)**(n_s + 3.0)
                                    * a**2)
                rows.append(dict(
                    sim=f"sf_n{n_tag}", n_s=n_s, gamma=gamma_true,
                    a=float(a), lpt_order=order, window="tet_stride1",
                    R_lag=R_grid, sigma2_lin=float(sigma2_lin_grid),
                    sigma_lin=float(np.sqrt(sigma2_lin_grid)),
                    **m0,
                ))
                for s, R in R_values.items():
                    V_R = tophat_convolve(V_cube, BOXSIZE, R)
                    m = moments(V_R)
                    sig2 = sigma8**2 * (8.0 / R)**(n_s + 3.0) * a**2
                    rows.append(dict(
                        sim=f"sf_n{n_tag}", n_s=n_s, gamma=gamma_true,
                        a=float(a), lpt_order=order,
                        window=f"tophat_R{s}", stride_equiv=int(s),
                        R_lag=float(R), sigma2_lin=float(sig2),
                        sigma_lin=float(np.sqrt(max(sig2, 0))),
                        **m,
                    ))
                dt_all = time.time() - t0
                print(f"  a={a:.3f}  order={order}  stride-1 pss={dt_pss:.2f}s  "
                      f"total={dt_all:.1f}s  "
                      f"(S₃^V stride-1={m0['S3_V']:+.2f}, "
                      f"tophat R_8={[r for r in rows if r.get('stride_equiv')==8][-1]['S3_V']:+.2f})")

    with open("docs/pss_lpt_tophat_cumulants.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved docs/pss_lpt_tophat_cumulants.json ({len(rows)} rows)")

    # Diagnostic plot: S_3^V vs σ² at matched R_lag, colour by γ, two line styles
    # tet-window (stride-matched from existing file) vs tophat (this run).
    try:
        tet_rows = json.load(open("docs/pss_lpt_cumulants.json"))
    except FileNotFoundError:
        tet_rows = []

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    for ax, s_eq in zip(axes, STRIDE_EQUIV):
        R_s = R_values[s_eq]
        # tophat rows at this R
        th = [r for r in rows if r.get("window") == f"tophat_R{s_eq}"
              and r["lpt_order"] == 2]
        # tet rows at the corresponding stride and order=2
        tet = [r for r in tet_rows if r.get("stride") == s_eq
                and r.get("lpt_order") == 2]
        colors = {-0.75: "tab:blue", -1.00: "tab:orange", -1.50: "tab:red"}
        for g, color in colors.items():
            pts_th  = [r for r in th  if abs(r["gamma"] - g) < 0.01]
            pts_tet = [r for r in tet if abs(r["gamma"] - g) < 0.01]
            if pts_th:
                pts_th.sort(key=lambda r: r["sigma_lin"])
                ax.plot([r["sigma_lin"]**2 for r in pts_th],
                         [r["S3_V"] for r in pts_th],
                         "o-", color=color, alpha=0.8, ms=6,
                         label=(fr"tophat, γ={g:+.2f}" if s_eq == 2 else None))
            if pts_tet:
                pts_tet.sort(key=lambda r: r["sigma_lin"])
                ax.plot([r["sigma_lin"]**2 for r in pts_tet],
                         [r["S3_V"] for r in pts_tet],
                         "s--", color=color, alpha=0.4, ms=5,
                         label=(fr"tet,    γ={g:+.2f}" if s_eq == 2 else None))
        ax.axhline(8/7, ls=":", color="black", alpha=0.7, label="8/7")
        ax.set_xlabel(r"$\sigma^2_{\rm lin}$"); ax.set_ylabel(r"$S_3^V$")
        ax.set_title(fr"$R_{s_eq} \approx {R_s:.2f}$ cells (2LPT)")
        ax.set_ylim(-0.5, 2.5)
        ax.grid(alpha=0.3)
        if s_eq == 2: ax.legend(fontsize=7)
    fig.suptitle("Tophat (solid) vs tet (dashed) windows at matched R — 2LPT",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig("docs/pss_tophat_vs_tet.png", dpi=130)
    print("Saved docs/pss_tophat_vs_tet.png")


if __name__ == "__main__":
    main()
