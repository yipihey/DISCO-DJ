"""Head-to-head: FEM N-body vs DISCO-DJ FastPM at matched ICs.

Not a pytest — interactive comparison driver. Scans a_end values and reports
trajectory differences (median |Δx|, |ψ| magnitude ratio) plus timings.

Usage::

    python tests/compare_nbody_vs_pm.py                       # defaults
    python tests/compare_nbody_vs_pm.py --N 32 --a_ini 0.02 --a_end 0.1 0.2 0.3
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from discodj import DiscoDJ
from discodj.nbody_fem import fem_nbody_cosmo, grid_positions, ics_from_discodj


def min_image(a, b, L):
    return (a - b) - L * np.round((a - b) / L)


def compare_one(dj, x0, p0, a_ini, a_end, n_steps, N, L, cg_tol, cg_maxiter):
    t0 = time.time()
    res_fem = fem_nbody_cosmo(x0, p0, dj.cosmo, a_ini, a_end, n_steps=n_steps,
                              boxsize=L, cg_tol=cg_tol, cg_maxiter=cg_maxiter)
    jax.block_until_ready(res_fem.state.x)
    t_fem = time.time() - t0

    t0 = time.time()
    x_pm, _, _ = dj.run_nbody(a_ini=a_ini, a_end=a_end, n_steps=n_steps,
                              stepper="fastpm", method="pm", res_pm=N,
                              time_var="log_a", ic_method="lpt",
                              nlpt_order_ics=1)
    t_pm = time.time() - t0
    x_pm = np.asarray(x_pm).reshape(N**3, 3)

    q = np.asarray(grid_positions(N, L))
    x_fem = np.asarray(res_fem.state.x)

    dx = min_image(x_fem, x_pm, L)
    per_part = np.linalg.norm(dx, axis=-1)
    cell = L / N

    psi_fem = min_image(x_fem, q, L)
    psi_pm = min_image(x_pm, q, L)
    mean_fem = np.mean(np.linalg.norm(psi_fem, axis=-1))
    mean_pm = np.mean(np.linalg.norm(psi_pm, axis=-1))

    return {
        "a_end": a_end,
        "median_dx_cells": float(np.median(per_part) / cell),
        "p95_dx_cells": float(np.percentile(per_part, 95) / cell),
        "max_dx_cells": float(np.max(per_part) / cell),
        "amp_ratio": float(mean_fem / mean_pm),
        "mean_psi_fem": float(mean_fem),
        "mean_psi_pm": float(mean_pm),
        "t_fem": t_fem,
        "t_pm": t_pm,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=16)
    p.add_argument("--boxsize", type=float, default=250.0)
    p.add_argument("--a_ini", type=float, default=0.05)
    p.add_argument("--a_end", type=float, nargs="+",
                   default=[0.1, 0.15, 0.2, 0.3, 0.5])
    p.add_argument("--n_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cg_tol", type=float, default=1e-8)
    p.add_argument("--cg_maxiter", type=int, default=15)
    args = p.parse_args()

    dj = (DiscoDJ(dim=3, res=args.N, boxsize=args.boxsize)
          .with_timetables()
          .with_linear_ps()
          .with_ics(seed=args.seed, white_noise_space="fourier")
          .with_lpt(n_order=1))
    x0, p0 = ics_from_discodj(dj, a_ini=args.a_ini)

    print(f"N={args.N}  L={args.boxsize} Mpc/h  a_ini={args.a_ini}  n_steps={args.n_steps}  seed={args.seed}")
    print()
    hdr = f"{'a_end':>6} | {'med Δx':>9} | {'p95 Δx':>9} | {'max Δx':>9} | {'|ψ| ratio':>9} | {'t FEM':>6} | {'t PM':>6}"
    print(hdr)
    print(f"{'':>6} | {'(cells)':>9} | {'(cells)':>9} | {'(cells)':>9} | {'FEM/PM':>9} | {'(s)':>6} | {'(s)':>6}")
    print("-" * len(hdr))
    for a_end in args.a_end:
        r = compare_one(dj, x0, p0, args.a_ini, a_end, args.n_steps,
                        args.N, args.boxsize, args.cg_tol, args.cg_maxiter)
        print(f"{r['a_end']:6.3f} | "
              f"{r['median_dx_cells']:9.4f} | {r['p95_dx_cells']:9.4f} | {r['max_dx_cells']:9.4f} | "
              f"{r['amp_ratio']:9.4f} | {r['t_fem']:6.1f} | {r['t_pm']:6.1f}")


if __name__ == "__main__":
    main()
