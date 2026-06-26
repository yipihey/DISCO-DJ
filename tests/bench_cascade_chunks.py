"""Sweep chunk_size to find the sweet spot for fem_cascade_v3_batch on CPU.

Memory-bound vmap at large batch size can underperform the scalar loop on
CPU; a moderate chunk_size (fits L2/L3) may win.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

from discodj.analysis import load_sim, deformation_gradient_jacobian
from discodj.analysis.spectral_slope import reconstruct_pk
from discodj.cascade import fem_cascade_v3, fem_cascade_v3_batch


N, N_STEPS, BOXSIZE = 64, 10, 250.0


def main():
    snap = load_sim(256, "nocutoff", a=0.07)
    G_pm, _ = deformation_gradient_jacobian(snap)
    s2 = float(jnp.var(jnp.einsum("...ii->...", G_pm)))
    k_np, Pk_np = reconstruct_pk(snap)
    pk_k, pk_Pk = jnp.asarray(k_np), jnp.asarray(Pk_np)

    betas = np.arange(-2.30, -1.09, 0.05)
    seeds = 5
    B = len(betas) * seeds
    beta_batch = jnp.asarray(np.repeat(betas, seeds), dtype=jnp.float32)
    keys = jax.vmap(jax.random.PRNGKey)(jnp.arange(5200, 5200 + B))

    # Warm jit for each chunk size we test (one compile per chunk size).
    chunk_sizes = [1, 2, 5, 10, 25, 125]
    print(f"B = {B}, N = {N}")
    print(f"{'chunk':>6} | {'time(s)':>8} | {'ms/call':>8} | {'×loop':>6}")

    # Baseline: scalar Python loop.
    _ = fem_cascade_v3(sigma2=s2, n_steps=N_STEPS, N=N, boxsize=BOXSIZE,
                       key=keys[0], beta_nl=float(beta_batch[0]),
                       pk_k=pk_k, pk_Pk=pk_Pk)  # warm
    t0 = time.time()
    for i in range(B):
        r = fem_cascade_v3(sigma2=s2, n_steps=N_STEPS, N=N, boxsize=BOXSIZE,
                           key=keys[i], beta_nl=float(beta_batch[i]),
                           pk_k=pk_k, pk_Pk=pk_Pk)
        jax.block_until_ready(r.J)
    t_loop = time.time() - t0
    print(f"{'loop':>6} | {t_loop:8.1f} | {t_loop/B*1000:8.0f} | {1.0:6.2f}")

    # Batched with various chunks.
    for cs in chunk_sizes:
        # warm the JIT for this chunk size
        _warm = fem_cascade_v3_batch(
            sigma2=s2, beta_nl=beta_batch[:min(cs, B)],
            keys=keys[:min(cs, B)],
            n_steps=N_STEPS, N=N, boxsize=BOXSIZE,
            pk_k=pk_k, pk_Pk=pk_Pk, chunk_size=cs)
        jax.block_until_ready(_warm.J)

        t0 = time.time()
        r = fem_cascade_v3_batch(
            sigma2=s2, beta_nl=beta_batch, keys=keys,
            n_steps=N_STEPS, N=N, boxsize=BOXSIZE,
            pk_k=pk_k, pk_Pk=pk_Pk, chunk_size=cs)
        jax.block_until_ready(r.J)
        t = time.time() - t0
        print(f"{cs:>6} | {t:8.1f} | {t/B*1000:8.0f} | "
              f"{t_loop/max(t,1e-6):6.2f}")


if __name__ == "__main__":
    main()
