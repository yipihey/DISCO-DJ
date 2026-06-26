"""Generate a PM sim suite at an unrealistic cosmology — robustness test.

Deliberately pushed past observational edges:
  Ω_c = 0.45  (vs Planck ~0.26 — over 70% extra dark matter)
  n_s = 1.25  (vs Planck 0.97 — blue-tilted, extra small-scale power)
  σ_8 = 1.00  (vs Planck 0.81 — high amplitude)

These shift P(k)'s shape and normalisation substantially. If the cascade
with the same β_nl still matches PM J-PDFs at matched σ², the effective
RG-flowed coupling is cosmology-independent — which is what we'd want
from a physics-first model.

Output: sims/N256_cutoff_extreme.npz, sims/N256_nocutoff_extreme.npz with
snapshots at a = 0.1, 0.3. Same seed/boxsize as the main suite.
"""

import os, time, json
import numpy as np
import jax, jax.numpy as jnp

from discodj import DiscoDJ
from discodj.core.grids import get_fourier_grid

EXTREME_COSMO = dict(
    Omega_c=0.45,    # vs ~0.26 Planck
    Omega_b=0.05,    # ~normal
    Omega_k=0.0,
    h=0.67,
    n_s=1.25,        # vs 0.97 Planck — strongly blue
    sigma8=1.0,      # vs 0.81 Planck
    w0=-1.0,
    wa=0.0,
)


def apply_cutoff(dj, k_cut):
    res = dj._res; L = dj._boxsize
    kd = get_fourier_grid([res]*3, boxsize=L, sparse_k_vecs=True, full=False,
                           dtype_num=dj.dtype_num, with_jax=True)
    return dj.update(ics={**dj._ics,
                          "fphi": dj._ics["fphi"]
                                  * jnp.exp(-0.5 * (kd["|k|"] / k_cut) ** 2)})


def main():
    L = 250.0
    seed = 42
    a_ini = 0.02
    a_save = [0.1, 0.3]
    n_steps = 30
    N = 256
    k_cut = 0.5 * np.pi * 64 / L

    os.makedirs("sims", exist_ok=True)
    for cutoff_mode in ("nocutoff", "cutoff"):
        tag = f"N{N}_{cutoff_mode}_extreme"
        out_path = f"sims/{tag}.npz"
        if os.path.exists(out_path):
            print(f"[skip] {tag}")
            continue
        print(f"[run] {tag}")
        t0 = time.time()
        dj = (DiscoDJ(dim=3, res=N, boxsize=L, cosmo=EXTREME_COSMO)
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
            x_list.append(np.asarray(x_t).reshape(N**3, 3).astype(np.float32))
            v_list.append(np.asarray(v_t).reshape(N**3, 3).astype(np.float32))
            a_list.append(float(a_t))

        meta = dict(N=N, L=L, seed=seed, cutoff_mode=cutoff_mode,
                    k_cut=(float(k_cut) if cutoff_mode == "cutoff" else 0.0),
                    a_ini=a_ini, n_steps=n_steps, cosmo=EXTREME_COSMO,
                    stepper="fastpm", method="pm", nlpt_order_ics=2,
                    k_order_fourier="stable")
        np.savez_compressed(out_path,
                            x=np.stack(x_list), v=np.stack(v_list),
                            a=np.asarray(a_list, np.float32),
                            meta=np.array(json.dumps(meta)))
        print(f"  done in {time.time()-t0:.0f}s, "
              f"{os.path.getsize(out_path)/1e6:.0f} MB, a={a_list}")


if __name__ == "__main__":
    main()
