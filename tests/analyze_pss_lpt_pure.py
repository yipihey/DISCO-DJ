"""Pure-nLPT PSS: use 2LPT displacements directly (no PM) for clean low-σ data.

Motivation: the A_1 extraction from the PM scale-free runs is noise-limited
at low σ because PM integrator + particle discreteness + small-residual shell
crossings add irreducible noise. Since the theoretical prediction
``S_3^V = 8/7 + (A_0 + A_1 γ) σ² + O(σ⁴)``
is a *2LPT statement*, the cleanest data to test it is 2LPT itself.

Pipeline:
  1. For each scale-free cosmology (n_s ∈ {-2.25, -2.0, -1.5}, EdS), build a
     single DiscoDJ object with fixed seed, compute ICs and 2LPT.
  2. For each target scale factor a in a dense low-σ grid, evaluate the
     pure 2LPT position field ``x = q + ψ_2LPT(a)``.
  3. Run the existing JAX PSS cube-volume pipeline at multiple strides and
     record (σ_v, κ_2, κ_3, κ_4, S_3^V, tr Σ², …).
  4. Fit ``S_3^V = 8/7 + (A_0 + A_1 γ) σ²`` on the usable subset.

Also records 1LPT for comparison (Zel'dovich limit) so we can see the 2LPT
correction directly.

Output:
  - ``docs/pss_lpt_cumulants.json``
  - ``docs/pss_lpt_A1_fit.json`` / ``.png``
"""

from __future__ import annotations

import json, os, time

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax, jax.numpy as jnp

from discodj import DiscoDJ
from discodj.analysis.sim_store import SimSnapshot
from discodj.analysis.phase_space_sheet import (
    kuhn_cube_volumes,
    kuhn_tet_deformation_tensor_from_positions,
    strain_invariants_from_F,
)


SF_N_VALUES = (("-2.25", -2.25), ("-2.0", -2.0), ("-1.5", -1.5))
A_TARGETS  = [0.01, 0.02, 0.04, 0.07, 0.1, 0.15, 0.22, 0.35, 0.55, 0.85]
N_PART     = 128
BOXSIZE    = 128.0
SEED       = 1234
STRIDES    = [1, 2, 4, 8, 16]
# Run 1LPT (Zel'dovich), 2LPT (theory ansatz target), then 3,5,10 for
# convergence. If 2LPT agrees with 10LPT on S_3^V at our σ range, the
# theory's 2LPT-truncated prediction is self-consistent. If it drifts,
# higher orders matter.
LPT_ORDERS = [1, 2, 3, 5, 10]


def sigma8_for_unit_variance_at_d(n: float) -> float:
    return 8.0 ** (-(n + 3.0) / 2.0)


def build_cosmo(n_s: float, sigma8: float) -> dict:
    return dict(Omega_c=1.0, Omega_b=0.0, Omega_k=0.0,
                h=1.0, n_s=float(n_s), sigma8=float(sigma8),
                w0=-1.0, wa=0.0)


def make_dj(n_s: float, sigma8: float, seed: int) -> DiscoDJ:
    return (DiscoDJ(dim=3, res=N_PART, boxsize=BOXSIZE,
                     cosmo=build_cosmo(n_s, sigma8))
            .with_timetables()
            .with_linear_ps(transfer_function="none")
            .with_ics(seed=seed, white_noise_space="fourier",
                      k_order_fourier="stable")
            .with_lpt(n_order=max(LPT_ORDERS)))


def snap_from_positions(x_any: np.ndarray, a: float,
                          cosmo: dict, L: float, N: int) -> SimSnapshot:
    """Wrap LPT positions into a SimSnapshot shape for PSS.

    Accepts either mesh ``(N, N, N, 3)`` or flat ``(N**3, 3)`` positions.
    """
    arr = np.asarray(x_any).astype(np.float32)
    if arr.ndim == 4:
        arr = arr.reshape(N**3, 3)
    return SimSnapshot(x=arr,
                        v=np.zeros_like(arr),
                        a=float(a), N=N, L=L,
                        meta=dict(N=N, L=L, scale_free=True, cosmo=cosmo))


def moments(V_cube: np.ndarray) -> dict:
    V = np.asarray(V_cube, dtype=np.float64).ravel()
    m = float(V.mean())
    v = V / m - 1.0
    c2 = float((v**2).mean())
    c3 = float((v**3).mean())
    c4 = float((v**4).mean()) - 3.0 * c2**2
    return dict(V_mean=m, kappa2=c2, kappa3=c3, kappa4=c4,
                sigma_v=float(np.sqrt(max(c2, 0))),
                S3_V=float(c3 / c2**2) if c2 > 0 else float("nan"),
                frac_neg=float((V <= 0).mean()))


def invariant_moments_from_F(F: jnp.ndarray) -> dict:
    """Aggregate moments over all tets/cells of the invariants."""
    inv = strain_invariants_from_F(F)
    # Flatten over (6, N, N, N) for each invariant
    out = {}
    for key in ("I1", "I2", "I3", "trG2", "trS2", "trS3", "antisym_sq"):
        arr = np.asarray(inv[key]).ravel()
        out[f"{key}_mean"] = float(arr.mean())
        out[f"{key}_var"]  = float(arr.var())
    # Cross moment ⟨I1²·I2⟩ at leading σ gives a specific rational
    I1 = np.asarray(inv["I1"]).ravel()
    I2 = np.asarray(inv["I2"]).ravel()
    trS2 = np.asarray(inv["trS2"]).ravel()
    out["I1sq_I2_mean"] = float(np.mean(I1**2 * I2))
    out["I1_trG2_mean"] = float(np.mean(I1 * np.asarray(inv["trG2"]).ravel()))
    out["I1sq_trS2_mean"] = float(np.mean(I1**2 * trS2))
    return out


def process_sf_lpt(n_tag: str, n_s: float):
    sigma8 = sigma8_for_unit_variance_at_d(n_s)
    gamma_true = -(n_s + 3.0)
    print(f"\n=== SF n={n_s}  γ={gamma_true:+.2f}  sigma8={sigma8:.4f}  "
          f"LPT orders={LPT_ORDERS} ===")
    dj = make_dj(n_s, sigma8, SEED)
    rows = []
    for a in A_TARGETS:
        for order in LPT_ORDERS:
            t0 = time.time()
            x_flat = dj.evaluate_lpt_pos_at_a(float(a), n_order=order)
            jax.block_until_ready(x_flat)
            snap = snap_from_positions(x_flat, a, {}, BOXSIZE, N_PART)
            for s in STRIDES:
                V = kuhn_cube_volumes(snap, stride=s, overlapping=True)
                mom = moments(V)
                d_c = BOXSIZE / (N_PART // s)
                R_lag = (d_c**3 * 3/(4*np.pi))**(1/3)
                s2_lin = sigma8**2 * (8.0 / R_lag)**(n_s + 3.0) * a**2
                row = dict(
                    sim=f"sf_n{n_tag}", n_s=n_s, gamma=gamma_true,
                    a=float(a), lpt_order=order, stride=s,
                    R_lag=R_lag, sigma2_lin=float(s2_lin),
                    sigma_lin=float(np.sqrt(max(s2_lin, 0))),
                    **mom,
                )
                rows.append(row)
            dt = time.time() - t0
            print(f"  a={a:6.3f}  order={order}  ({dt:.1f}s)")
    return rows


def fit_A1(rows, order: int, sigma_min: float = 0.15, sigma_max: float = 0.7):
    """Fit S_3^V = 8/7 + (A_0 + A_1 γ)·σ² on the LPT subset.

    Default range σ_lin ∈ [0.15, 0.7] skips the Kuhn-tet-anisotropy-dominated
    small-σ regime and the PT-breakdown large-σ edge.
    """
    sub = [r for r in rows if r["lpt_order"] == order
            and sigma_min <= r["sigma_lin"] <= sigma_max
            and r["kappa2"] > 0]
    if not sub: return None
    sig2 = np.array([r["sigma_lin"]**2 for r in sub])
    gamma = np.array([r["gamma"] for r in sub])
    S3 = np.array([r["S3_V"] for r in sub])
    Y = S3 - 8.0/7.0
    X = np.column_stack([sig2, gamma * sig2])
    c, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ c
    rms = float(np.sqrt((resid**2).mean()))
    XTX_inv = np.linalg.inv(X.T @ X)
    se = rms * np.sqrt(np.diag(XTX_inv))
    return dict(order=order, n_rows=len(sub),
                A_0=float(c[0]), A_1=float(c[1]),
                se_A0=float(se[0]), se_A1=float(se[1]),
                rms=rms, sigma_min=sigma_min, sigma_max=sigma_max)


def main():
    os.makedirs("docs", exist_ok=True)
    rows = []
    for n_tag, n_s in SF_N_VALUES:
        rows += process_sf_lpt(n_tag, n_s)
    with open("docs/pss_lpt_cumulants.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved docs/pss_lpt_cumulants.json ({len(rows)} rows).")

    # Fit for each LPT order separately.
    fits = {}
    for order in LPT_ORDERS:
        fit = fit_A1(rows, order)
        if fit is None: continue
        fits[order] = fit
        print(f"\n[LPT order={order}, n={fit['n_rows']}]")
        print(f"  S_3^V = 8/7 + ({fit['A_0']:+.3f}±{fit['se_A0']:.3f} + "
              f"({fit['A_1']:+.3f}±{fit['se_A1']:.3f})·γ)·σ²  rms={fit['rms']:.3f}")
        A0_theory = 32/49
        print(f"  A_0 vs 32/49={A0_theory:+.3f}: Δ={fit['A_0']-A0_theory:+.3f} "
              f"({(fit['A_0']-A0_theory)/fit['se_A0']:+.2f}σ)")
    with open("docs/pss_lpt_A1_fit.json", "w") as f:
        json.dump(fits, f, indent=2)
    print("Saved docs/pss_lpt_A1_fit.json")

    # Diagnostic: S_3^V vs σ² colored by γ, one panel per LPT order.
    fig, axes = plt.subplots(1, len(LPT_ORDERS), figsize=(7 * len(LPT_ORDERS), 5),
                              sharey=True)
    if len(LPT_ORDERS) == 1: axes = [axes]
    for ax, order in zip(axes, LPT_ORDERS):
        sub = [r for r in rows if r["lpt_order"] == order]
        colors = {-0.75: "tab:blue", -1.00: "tab:orange", -1.50: "tab:red"}
        for g, color in colors.items():
            pts = [r for r in sub if abs(r["gamma"] - g) < 0.01]
            pts.sort(key=lambda r: r["sigma_lin"])
            x = [r["sigma_lin"]**2 for r in pts]
            y = [r["S3_V"] for r in pts]
            ax.plot(x, y, "o", color=color, alpha=0.7, ms=5,
                    label=fr"γ={g:+.2f}")
        # Overlay fit at mid-γ
        fit = fits.get(order, None)
        if fit is not None:
            xs = np.linspace(0, 1.0, 50)
            for g, color in colors.items():
                ys = 8/7 + (fit["A_0"] + fit["A_1"] * g) * xs
                ax.plot(xs, ys, "-", color=color, alpha=0.3)
        ax.axhline(8/7, ls="--", color="black", alpha=0.5, label="8/7")
        ax.set_xlabel(r"$\sigma_{\rm lin}^2$"); ax.set_ylabel(r"$S_3^V$")
        ax.set_title(f"LPT order = {order}")
        ax.set_xlim(0, 0.85); ax.set_ylim(0.0, 2.2)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Pure LPT PSS: S_3^V(σ²) across γ — A_1 extraction",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig("docs/pss_lpt_A1_fit.png", dpi=130)
    print("Saved docs/pss_lpt_A1_fit.png")


if __name__ == "__main__":
    main()
