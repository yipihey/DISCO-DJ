"""Add low-a (high-z) snapshots to the existing sim suite.

Runs short PM evolutions from a_ini=0.02 to a_end ∈ {0.03, 0.05, 0.07}
on the same seed/config as `gen_pm_suite.py`, for the N=256 case only
(that's the converged resolution). Output: `sims/N256_{cutoff_mode}_lowa.npz`
with snapshots at these low scale factors.

These low-a sims probe σ² ≲ 0.01 where cascade-PT agreement should be
essentially exact up to finite-sample noise.
"""

import os, time, json
import numpy as np
import jax
import jax.numpy as jnp
from discodj import DiscoDJ
from discodj.core.grids import get_fourier_grid


def apply_cutoff(dj, k_cut):
    res = dj._res; L = dj._boxsize
    kd = get_fourier_grid([res]*3, boxsize=L, sparse_k_vecs=True, full=False,
                           dtype_num=dj.dtype_num, with_jax=True)
    kernel = jnp.exp(-0.5 * (kd["|k|"] / k_cut) ** 2)
    return dj.update(ics={**dj._ics, "fphi": dj._ics["fphi"] * kernel})


def main():
    L = 250.0
    seed = 42
    a_ini = 0.02
    a_save = [0.03, 0.05, 0.07]
    n_steps = 30
    k_cut = 0.5 * np.pi * 64 / L

    out_dir = "sims"
    os.makedirs(out_dir, exist_ok=True)
    N = 256
    for cutoff_mode in ("nocutoff", "cutoff"):
        tag = f"N{N}_{cutoff_mode}_lowa"
        out_path = os.path.join(out_dir, f"{tag}.npz")
        if os.path.exists(out_path):
            print(f"[skip] {tag}"); continue
        print(f"[run] {tag}")
        t0 = time.time()
        dj = (DiscoDJ(dim=3, res=N, boxsize=L)
              .with_timetables().with_linear_ps()
              .with_ics(seed=seed, white_noise_space="fourier",
                        k_order_fourier="stable"))
        if cutoff_mode == "cutoff":
            dj = apply_cutoff(dj, k_cut)
        dj = dj.with_lpt(n_order=2)
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
        meta = dict(N=N, L=L, seed=seed, cutoff_mode=cutoff_mode,
                    k_cut=(float(k_cut) if cutoff_mode == "cutoff" else 0.0),
                    a_ini=a_ini, n_steps=n_steps, stepper="fastpm",
                    method="pm", nlpt_order_ics=2,
                    k_order_fourier="stable")
        np.savez_compressed(out_path,
                            x=np.stack(x_list), v=np.stack(v_list),
                            a=np.asarray(a_list, np.float32),
                            meta=np.array(json.dumps(meta)))
        dt = time.time() - t0
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  done in {dt:.0f}s, {size_mb:.0f} MB, a={a_list}")


if __name__ == "__main__":
    main()
