"""Multi-seed pure-LPT PSS + invariant moments for variance reduction.

Repeats the pure-LPT measurement across N_SEEDS independent IC realisations
of each scale-free cosmology, then combines into a single table with an
extra ``seed`` column. Each (sim, a, order, stride) gets N_SEEDS independent
measurements — bootstrap across them for empirical error bars.

Pipeline:
  1. For each (seed, n_tag) build a DiscoDJ object with that seed and LPT
     order up to max(LPT_ORDERS).
  2. Loop over a, order: evaluate LPT positions; compute per-cube V and
     all invariant moments at every stride.
  3. Save to docs/pss_lpt_multiseed.json.
  4. Fit A_0, A_1 with bootstrap-over-seeds errors.
"""

from __future__ import annotations

import json, os, time

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax, jax.numpy as jnp

from discodj import DiscoDJ
from discodj.analysis import (
    kuhn_cube_volumes_from_positions,
    kuhn_tet_deformation_tensor_from_positions,
    strain_invariants_from_F,
)
from discodj.analysis.sim_store import SimSnapshot


SF_N_VALUES = (("-2.25", -2.25), ("-2.0", -2.0), ("-1.5", -1.5))
A_TARGETS  = [0.05, 0.10, 0.20, 0.35, 0.55]
N_PART     = 128
BOXSIZE    = 128.0
SEEDS      = [1234, 2345, 3456, 4567, 5678]
STRIDES    = [1, 2, 4, 8]
LPT_ORDERS = [2, 3, 5]


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


def lpt_positions_mesh(dj, a: float, order: int) -> jnp.ndarray:
    x_flat = dj.evaluate_lpt_pos_at_a(float(a), n_order=order)
    L = float(dj._boxsize); N = int(dj._res)
    x_mesh = np.asarray(x_flat).reshape(N, N, N, 3).astype(np.float32)
    ix, iy, iz = np.meshgrid(np.arange(N), np.arange(N), np.arange(N),
                              indexing="ij")
    h = L / N
    q = np.stack([ix * h, iy * h, iz * h], axis=-1).astype(np.float32)
    psi = x_mesh - q
    psi = psi - L * np.round(psi / L)
    return jnp.asarray(q + psi)


def summary_from_V(V_cube: np.ndarray) -> dict:
    V = np.asarray(V_cube, dtype=np.float64).ravel()
    m = float(V.mean())
    v = V / m - 1.0
    c2 = float((v**2).mean())
    c3 = float((v**3).mean())
    c4 = float((v**4).mean()) - 3.0 * c2**2
    return dict(V_mean=m, kappa2=c2, kappa3=c3, kappa4=c4,
                sigma_v=float(np.sqrt(max(c2, 0))),
                S3_V=float(c3 / c2**2) if c2 > 0 else float("nan"))


def invariant_moments(F: jnp.ndarray) -> dict:
    inv = strain_invariants_from_F(F)
    I1 = np.asarray(inv["I1"]).ravel()
    I2 = np.asarray(inv["I2"]).ravel()
    I3 = np.asarray(inv["I3"]).ravel()
    trG2 = np.asarray(inv["trG2"]).ravel()
    trS2 = np.asarray(inv["trS2"]).ravel()
    trS3 = np.asarray(inv["trS3"]).ravel()
    asq = np.asarray(inv["antisym_sq"]).ravel()
    out = dict(
        I1_mean=float(I1.mean()),
        I1sq_mean=float((I1**2).mean()),
        I1_cubed_mean=float((I1**3).mean()),
        I2_mean=float(I2.mean()),
        I3_mean=float(I3.mean()),
        trG2_mean=float(trG2.mean()),
        trS2_mean=float(trS2.mean()),
        trS3_mean=float(trS3.mean()),
        I1_I2_mean=float((I1 * I2).mean()),
        I1sq_I2_mean=float((I1 * I1 * I2).mean()),
        I2sq_mean=float((I2 * I2).mean()),
        I1sq_trG2_mean=float((I1 * I1 * trG2).mean()),
        I1sq_trS2_mean=float((I1 * I1 * trS2).mean()),
        I1_I3_mean=float((I1 * I3).mean()),
        I1_I2sq_mean=float((I1 * I2 * I2).mean()),
        I1sq_I3_mean=float((I1 * I1 * I3).mean()),
        I2_I3_mean=float((I2 * I3).mean()),
        I2_cubed_mean=float((I2 * I2 * I2).mean()),
        I3sq_mean=float((I3 * I3).mean()),
        I1_I2_I3_mean=float((I1 * I2 * I3).mean()),
        antisym_sq_mean=float(asq.mean()),
    )
    return out


def main():
    os.makedirs("docs", exist_ok=True)
    rows = []
    n_max = max(LPT_ORDERS)

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n###### SEED {seed_idx + 1}/{len(SEEDS)} (value={seed}) ######")
        for n_tag, n_s in SF_N_VALUES:
            sigma8 = sigma8_for_unit_variance_at_d(n_s)
            gamma_true = -(n_s + 3.0)
            print(f"\n== SF n={n_s}  γ={gamma_true:+.2f}  seed={seed} ==")
            dj = make_dj(n_s, sigma8, seed, n_max)
            for a in A_TARGETS:
                for order in LPT_ORDERS:
                    t0 = time.time()
                    x = lpt_positions_mesh(dj, a, order)
                    for s in STRIDES:
                        V = kuhn_cube_volumes_from_positions(
                            x, float(BOXSIZE), stride=s, overlapping=True)
                        F = kuhn_tet_deformation_tensor_from_positions(
                            x, float(BOXSIZE), stride=s, overlapping=True)
                        mom_V = summary_from_V(np.asarray(V))
                        mom_I = invariant_moments(F)
                        d_c = BOXSIZE / (N_PART // s)
                        R_lag = (d_c**3 * 3/(4*np.pi))**(1/3)
                        s2_lin = sigma8**2 * (8.0/R_lag)**(n_s+3.0) * a**2
                        rows.append(dict(
                            seed=int(seed), sim=f"sf_n{n_tag}",
                            n_s=n_s, gamma=gamma_true,
                            a=float(a), lpt_order=order, stride=s,
                            R_lag=R_lag, sigma2_lin=float(s2_lin),
                            sigma_lin=float(np.sqrt(s2_lin)),
                            **mom_V, **mom_I,
                        ))
                    dt = time.time() - t0
                    print(f"  a={a:.2f}  order={order}  ({dt:.1f}s)")
            del dj

    with open("docs/pss_lpt_multiseed.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved docs/pss_lpt_multiseed.json ({len(rows)} rows, "
          f"{len(SEEDS)} seeds × {len(SF_N_VALUES)} sims × "
          f"{len(A_TARGETS)} a × {len(LPT_ORDERS)} orders × "
          f"{len(STRIDES)} strides)")

    # --- bootstrap A_0, A_1 per (window = stride) with seeds as resampling units ---
    print("\n=== Bootstrap A_0, A_1 per stride (order=2, σ_lin∈[0.10,0.60]) ===")
    def fit_rows(subset):
        if len(subset) < 3: return None
        sig2 = np.array([r["sigma_lin"]**2 for r in subset])
        gamma = np.array([r["gamma"] for r in subset])
        S3 = np.array([r["S3_V"] for r in subset])
        X = np.column_stack([sig2, gamma * sig2])
        Y = S3 - 8/7
        c, *_ = np.linalg.lstsq(X, Y, rcond=None)
        return float(c[0]), float(c[1])

    rng = np.random.default_rng(0)
    N_boot = 500
    for stride in STRIDES:
        sub = [r for r in rows if r["lpt_order"] == 2 and r["stride"] == stride
               and 0.10 <= r["sigma_lin"] <= 0.60]
        if not sub: continue
        # group by seed for bootstrap
        seeds_avail = list({r["seed"] for r in sub})
        boot_A0 = []; boot_A1 = []
        for _ in range(N_boot):
            draw = rng.choice(seeds_avail, size=len(seeds_avail), replace=True)
            sample = [r for r in sub if r["seed"] in draw]
            fit = fit_rows(sample)
            if fit is not None:
                boot_A0.append(fit[0]); boot_A1.append(fit[1])
        # point estimate from full data
        A0_full, A1_full = fit_rows(sub)
        A0_med = float(np.median(boot_A0)); A1_med = float(np.median(boot_A1))
        A0_lo, A0_hi = np.percentile(boot_A0, [16, 84])
        A1_lo, A1_hi = np.percentile(boot_A1, [16, 84])
        print(f"  stride={stride}  n={len(sub)}  "
              f"A_0 = {A0_full:+.3f}  [bs {A0_lo:+.3f}, {A0_hi:+.3f}]  "
              f"A_1 = {A1_full:+.3f}  [bs {A1_lo:+.3f}, {A1_hi:+.3f}]   "
              f"→ A_1_paper ∈ [{-A1_hi:+.3f}, {-A1_lo:+.3f}]")


if __name__ == "__main__":
    main()
