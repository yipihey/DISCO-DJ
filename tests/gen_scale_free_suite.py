"""Generate scale-free (power-law) PM sims for 2LPT theory comparison.

Three 128³ runs in Einstein-de Sitter with power-law P(k) = A k^n and n in
{-1.5, -2.0, -2.25}. γ = -(n+3) = {-1.5, -1.0, -0.75} brackets the ΛCDM
multi-scale γ range from the low-a kNN measurement (γ ∈ [-1.06, -0.64]) so
we can isolate spectral-slope dependence cleanly.

Six snapshots per sim span σ²_lin(R=d) from ~0.0025 at a=0.05 up to ~1 at
a=1 (EdS: D(a) ∝ a, so σ² ∝ a²). Box: L = 128 in arbitrary units so the
inter-particle spacing d = 1 — all our k-shell R_lag values then read
directly in units of d.

Amplitude normalisation: sigma8 set so σ²(R=1, a=1) ≈ 1. For pure power-law,
σ²(R)/σ²(8) = 8^(n+3) → sigma8 = 1/sqrt(8^(n+3)).

Output per sim: sims/N128_sf_n{n_tag}.npz with x, v, a for each snapshot.
"""

import os, time, json
import numpy as np
import jax, jax.numpy as jnp

from discodj import DiscoDJ


BOXSIZE = 128.0            # arbitrary units; d = L/N = 1
N_PART  = 128
A_INI   = 0.02
A_SAVE  = [0.05, 0.1, 0.2, 0.35, 0.6, 1.0]
N_STEPS = 30
SEED    = 1234

# Three scale-free sims: (n, tag)
COSMOS = [
    ("-2.25", -2.25, "-0.75"),
    ("-2.0",  -2.00, "-1.00"),
    ("-1.5",  -1.50, "-1.50"),
]


def build_cosmo(n_s: float, sigma8: float) -> dict:
    """EdS with custom spectral index and sigma8."""
    return dict(
        Omega_c=1.0, Omega_b=0.0, Omega_k=0.0,
        h=1.0, n_s=float(n_s), sigma8=float(sigma8),
        w0=-1.0, wa=0.0,
    )


def sigma8_for_unit_variance_at_d(n: float) -> float:
    """Pick sigma8 such that σ²(R=d=1, a=1) = 1 for P(k) = A k^n."""
    # σ²(R₁)/σ²(R₂) = (R₁/R₂)^{-(n+3)}  → σ²(1)/σ²(8) = 8^{n+3}
    # Target σ²(1) = 1 → σ²(8) = 8^{-(n+3)} → sigma8 = 8^{-(n+3)/2}.
    return 8.0 ** (-(n + 3.0) / 2.0)


def run_one(n_tag, n_s, gamma_val):
    out_path = f"sims/N{N_PART}_sf_n{n_tag}.npz"
    if os.path.exists(out_path):
        print(f"[skip] {out_path}"); return
    sigma8 = sigma8_for_unit_variance_at_d(n_s)
    cosmo = build_cosmo(n_s, sigma8)
    print(f"[run] n={n_s}  γ={gamma_val}  sigma8={sigma8:.4f}")
    t0 = time.time()

    x_list, v_list, a_list = [], [], []
    for a_t in A_SAVE:
        print(f"  a_end={a_t}  ...", end=" ", flush=True)
        ts = time.time()
        dj = (DiscoDJ(dim=3, res=N_PART, boxsize=BOXSIZE, cosmo=cosmo)
              .with_timetables()
              .with_linear_ps(transfer_function="none")      # P(k) = A k^n
              .with_ics(seed=SEED, white_noise_space="fourier",
                        k_order_fourier="stable"))
        dj = dj.with_lpt(n_order=2)
        x_t, v_t, _ = dj.run_nbody(
            a_ini=A_INI, a_end=float(a_t), n_steps=N_STEPS,
            stepper="fastpm", method="pm", res_pm=N_PART,
            time_var="log_a", ic_method="lpt", nlpt_order_ics=2,
        )
        jax.block_until_ready(x_t)
        x_arr = np.asarray(x_t).reshape(N_PART**3, 3).astype(np.float32)
        v_arr = np.asarray(v_t).reshape(N_PART**3, 3).astype(np.float32)
        x_list.append(x_arr); v_list.append(v_arr); a_list.append(float(a_t))
        del dj, x_t, v_t
        print(f"done in {time.time()-ts:.0f}s")

    meta = dict(N=N_PART, L=BOXSIZE, seed=SEED,
                scale_free=True, n_s=float(n_s), gamma_lin=float(gamma_val),
                sigma8=float(sigma8), cosmo=cosmo, a_ini=A_INI,
                n_steps=N_STEPS, stepper="fastpm", method="pm",
                nlpt_order_ics=2, k_order_fourier="stable")
    np.savez_compressed(out_path,
                        x=np.stack(x_list), v=np.stack(v_list),
                        a=np.asarray(a_list, np.float32),
                        meta=np.array(json.dumps(meta)))
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  total {time.time()-t0:.0f}s, {size_mb:.0f} MB, a={a_list}")


def main():
    os.makedirs("sims", exist_ok=True)
    for n_tag, n_s, gamma_val in COSMOS:
        run_one(n_tag, n_s, gamma_val)


if __name__ == "__main__":
    main()
