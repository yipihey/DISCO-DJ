"""Wall-time benchmark for the cascade v3 batch API.

Compares: (a) legacy Python-looped scalar calls and (b)
``fem_cascade_v3_batch`` at the canonical scan config (N=64, 25 β × 5
seeds). Old scan config under a=0.07 nocut.
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

    betas = np.arange(-2.30, -1.09, 0.05)       # 25 points
    seeds = 5
    B = len(betas) * seeds

    beta_batch = jnp.asarray(np.repeat(betas, seeds), dtype=jnp.float32)
    keys = jax.vmap(jax.random.PRNGKey)(jnp.arange(5200, 5200 + B))

    print(f"Snapshot a=0.07  σ²={s2:.4f}")
    print(f"Batch size B = {B}  (25 β × 5 seeds)")

    # --- warm jit (compile cost outside timing) ---
    print("Warming JIT...")
    t0 = time.time()
    r_warm = fem_cascade_v3_batch(
        sigma2=s2, beta_nl=beta_batch[:2], keys=keys[:2],
        n_steps=N_STEPS, N=N, boxsize=BOXSIZE,
        pk_k=pk_k, pk_Pk=pk_Pk)
    jax.block_until_ready(r_warm.J)
    print(f"  batch warm: {time.time()-t0:.1f}s")

    t0 = time.time()
    r_warm = fem_cascade_v3(sigma2=s2, n_steps=N_STEPS, N=N, boxsize=BOXSIZE,
                            key=keys[0], beta_nl=float(beta_batch[0]),
                            pk_k=pk_k, pk_Pk=pk_Pk)
    jax.block_until_ready(r_warm.J)
    print(f"  scalar warm: {time.time()-t0:.1f}s")

    # --- full-batch run ---
    t0 = time.time()
    r_b = fem_cascade_v3_batch(
        sigma2=s2, beta_nl=beta_batch, keys=keys,
        n_steps=N_STEPS, N=N, boxsize=BOXSIZE,
        pk_k=pk_k, pk_Pk=pk_Pk)
    jax.block_until_ready(r_b.J)
    t_batch = time.time() - t0

    # --- legacy loop for comparison ---
    t0 = time.time()
    for i in range(B):
        r = fem_cascade_v3(
            sigma2=s2, n_steps=N_STEPS, N=N, boxsize=BOXSIZE,
            key=keys[i], beta_nl=float(beta_batch[i]),
            pk_k=pk_k, pk_Pk=pk_Pk)
        jax.block_until_ready(r.J)
    t_loop = time.time() - t0

    print(f"\n  batch run: {t_batch:.1f}s ({t_batch/B*1000:.0f} ms/call)")
    print(f"  loop  run: {t_loop:.1f}s ({t_loop/B*1000:.0f} ms/call)")
    print(f"  speedup: ×{t_loop/max(t_batch, 1e-6):.2f}")

    # --- parity spot check on first 3 batch elements ---
    print("\nParity check (first 3 batch elements, J Frobenius):")
    for i in range(3):
        r_s = fem_cascade_v3(
            sigma2=s2, n_steps=N_STEPS, N=N, boxsize=BOXSIZE,
            key=keys[i], beta_nl=float(beta_batch[i]),
            pk_k=pk_k, pk_Pk=pk_Pk)
        diff = float(jnp.linalg.norm(r_b.J[i] - r_s.J))
        norm = float(jnp.linalg.norm(r_s.J))
        print(f"  i={i}:  ||ΔJ||/||J|| = {diff/norm:.2e}")


if __name__ == "__main__":
    main()
