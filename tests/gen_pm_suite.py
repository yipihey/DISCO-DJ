"""Generate the 6-simulation PM cascade-validation suite.

Produces ``sims/N{N}_{cutoff_mode}.npz`` for N ∈ {64, 128, 256} and
cutoff_mode ∈ {nocutoff, cutoff}. Each file contains snapshots at
a = [0.1, 0.3, 0.5, 0.7, 1.0] with positions (x), velocities (v), and
metadata (N, L, seed, k_cut, ...).

Cutoff: Gaussian exp(−½(k/k_cut)²) applied to fphi_ini with
k_cut = π·N=64/(2L) = 0.40 h/Mpc (half-Nyquist of N=64). The smoothing
scale is R ~ 2.5 Mpc/h — roughly 2/3 of a cell at N=64, so the cutoff
is just barely resolved in the coarsest sim and well-resolved at N≥128.

Skip files that already exist so the script is safely restartable.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from discodj import DiscoDJ
from discodj.core.grids import get_fourier_grid


def apply_cutoff(dj: DiscoDJ, k_cut: float) -> DiscoDJ:
    """Multiply the IC Fourier-space potential by a Gaussian cutoff."""
    res = dj._res
    L = dj._boxsize
    kd = get_fourier_grid([res] * 3, boxsize=L, sparse_k_vecs=True,
                          full=False, dtype_num=dj.dtype_num, with_jax=True)
    kernel = jnp.exp(-0.5 * (kd["|k|"] / k_cut) ** 2)
    return dj.update(ics={**dj._ics, "fphi": dj._ics["fphi"] * kernel})


def run_one(N: int, cutoff_mode: str, *, L: float, seed: int,
            a_ini: float, n_steps: int, a_save: list[float],
            k_cut: float, out_dir: str) -> None:
    tag = f"N{N}_{cutoff_mode}"
    out_path = os.path.join(out_dir, f"{tag}.npz")
    if os.path.exists(out_path):
        print(f"  [skip] {tag} already exists ({os.path.getsize(out_path)/1e6:.0f} MB)")
        return

    print(f"  [run]  {tag} ...")
    t0 = time.time()
    dj = (DiscoDJ(dim=3, res=N, boxsize=L)
          .with_timetables()
          .with_linear_ps()
          .with_ics(seed=seed, white_noise_space="fourier",
                    k_order_fourier="stable"))
    if cutoff_mode == "cutoff":
        dj = apply_cutoff(dj, k_cut)
    dj = dj.with_lpt(n_order=2)

    # Run a separate sim per snapshot to sidestep DiscoDJ's collect_all bug
    # at current code. Each run shares the same seed/state so the trajectory
    # up to each a_save matches what a continuous run would produce.
    x_list, v_list, a_list = [], [], []
    for a_t in a_save:
        x_t, v_t, _ = dj.run_nbody(
            a_ini=a_ini, a_end=float(a_t), n_steps=n_steps,
            stepper="fastpm", method="pm", res_pm=N,
            time_var="log_a", ic_method="lpt", nlpt_order_ics=2,
        )
        jax.block_until_ready(x_t)
        x_arr = np.asarray(x_t).reshape(N**3, 3).astype(np.float32)
        v_arr = np.asarray(v_t).reshape(N**3, 3).astype(np.float32)
        x_list.append(x_arr); v_list.append(v_arr); a_list.append(float(a_t))
    x_snap = np.stack(x_list)
    v_snap = np.stack(v_list)
    a_snap = np.asarray(a_list, dtype=np.float32)

    meta = dict(
        N=N, L=L, seed=seed, cutoff_mode=cutoff_mode,
        k_cut=(float(k_cut) if cutoff_mode == "cutoff" else 0.0),
        a_ini=a_ini, n_steps=n_steps,
        stepper="fastpm", method="pm", nlpt_order_ics=2,
        k_order_fourier="stable",
    )
    np.savez_compressed(
        out_path,
        x=x_snap, v=v_snap, a=a_snap,
        meta=np.array(json.dumps(meta)),
    )
    dt = time.time() - t0
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  [done] {tag}  {dt:.0f}s  x.shape={x_snap.shape}  "
          f"a={np.round(a_snap, 3).tolist()}  {size_mb:.0f} MB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=float, default=250.0, help="Mpc/h")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--a_ini", type=float, default=0.02)
    p.add_argument("--n_steps", type=int, default=30)
    p.add_argument("--a_save", type=float, nargs="+",
                   default=[0.1, 0.3, 0.5, 0.7, 1.0])
    p.add_argument("--N", type=int, nargs="+", default=[64, 128, 256])
    p.add_argument("--out_dir", type=str, default="sims")
    args = p.parse_args()

    k_cut = 0.5 * np.pi * 64 / args.L
    os.makedirs(args.out_dir, exist_ok=True)

    # Drop a README with the suite-level metadata
    readme = os.path.join(args.out_dir, "README.md")
    if not os.path.exists(readme):
        with open(readme, "w") as f:
            f.write(
                f"# PM cascade-validation suite\n\n"
                f"- Box L = {args.L} Mpc/h, seed = {args.seed}, "
                f"k_order_fourier = 'stable'\n"
                f"- a_ini = {args.a_ini}, a_save = {args.a_save}\n"
                f"- stepper = fastpm, method = pm, res_pm = N, "
                f"n_steps = {args.n_steps}\n"
                f"- ICs: 2LPT, LPT order 2\n"
                f"- Cutoff: Gaussian exp(-½(k/k_cut)²) with "
                f"k_cut = {k_cut:.4f} h/Mpc "
                f"(= half-Nyquist of N=64 → R ≈ {1/k_cut:.2f} Mpc/h)\n\n"
                f"Files: N{{N}}_{{cutoff_mode}}.npz for "
                f"N ∈ {args.N} × {{nocutoff, cutoff}}.\n"
                f"Load via numpy.load(path); keys: x, v, a, meta "
                f"(meta is a json-encoded dict stored as a 0-d numpy array).\n"
            )

    print(f"k_cut = {k_cut:.4f} h/Mpc  "
          f"(R ~ {1/k_cut:.2f} Mpc/h; h_N64 = {args.L/64:.2f} Mpc/h)")
    for N in args.N:
        print(f"N = {N}")
        for cutoff_mode in ("nocutoff", "cutoff"):
            run_one(N, cutoff_mode, L=args.L, seed=args.seed,
                    a_ini=args.a_ini, n_steps=args.n_steps,
                    a_save=args.a_save, k_cut=k_cut, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
