"""Generate N=512 nocut PM snapshots at low a for β(σ) universality test.

Mirrors :mod:`gen_low_a_snapshots` at N=512 (nocutoff only) at
a ∈ {0.03, 0.05, 0.07, 0.10}. Same cosmology, seed, and boxsize as the
main suite — only the grid resolution changes. At N=512 the Nyquist is
k_Nyq ≈ 6.4 h/Mpc, so the nocut suite picks up more small-scale power
and σ² shifts up vs N=256 at the same ``a``.

Goal: at the new (larger) σ values, run the cascade scan and check
whether β_best still sits on the β = (3/5)·ln(σ/e) line — i.e.
whether the 3/5 law is **resolution-independent** or a truncation
artifact of N=256.

Output: ``sims/N512_nocutoff_lowa.npz`` with snapshots at a=[0.03,0.05,
0.07,0.10], matching the gen_low_a_snapshots format so load_sim picks
it up via the ``_lowa`` companion.
"""

import os, time, json
import numpy as np
import jax
import jax.numpy as jnp

from discodj import DiscoDJ


def main():
    L = 250.0
    seed = 42
    a_ini = 0.02
    a_save = [0.03, 0.05, 0.07, 0.10]
    n_steps = 30
    N = 512

    out_dir = "sims"
    os.makedirs(out_dir, exist_ok=True)
    tag = f"N{N}_nocutoff_lowa"
    out_path = os.path.join(out_dir, f"{tag}.npz")
    if os.path.exists(out_path):
        print(f"[skip] {tag}"); return

    print(f"[run] {tag}  (N={N}, 4 snapshots)")
    t0 = time.time()

    x_list, v_list, a_list = [], [], []
    for a_t in a_save:
        print(f"  a_end = {a_t}  ...", end=" ", flush=True)
        ts = time.time()
        dj = (DiscoDJ(dim=3, res=N, boxsize=L)
              .with_timetables().with_linear_ps()
              .with_ics(seed=seed, white_noise_space="fourier",
                        k_order_fourier="stable"))
        dj = dj.with_lpt(n_order=2)
        x_t, v_t, _ = dj.run_nbody(
            a_ini=a_ini, a_end=float(a_t), n_steps=n_steps,
            stepper="fastpm", method="pm", res_pm=N,
            time_var="log_a", ic_method="lpt", nlpt_order_ics=2,
        )
        jax.block_until_ready(x_t)
        x_arr = np.asarray(x_t).reshape(N**3, 3).astype(np.float32)
        v_arr = np.asarray(v_t).reshape(N**3, 3).astype(np.float32)
        x_list.append(x_arr); v_list.append(v_arr); a_list.append(float(a_t))
        del dj, x_t, v_t
        print(f"done in {time.time()-ts:.0f}s")

    meta = dict(N=N, L=L, seed=seed, cutoff_mode="nocutoff", k_cut=0.0,
                a_ini=a_ini, n_steps=n_steps, stepper="fastpm", method="pm",
                nlpt_order_ics=2, k_order_fourier="stable")
    np.savez_compressed(out_path,
                        x=np.stack(x_list), v=np.stack(v_list),
                        a=np.asarray(a_list, np.float32),
                        meta=np.array(json.dumps(meta)))
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"Total: {time.time()-t0:.0f}s, {size_mb:.0f} MB, a={a_list}")


if __name__ == "__main__":
    main()
