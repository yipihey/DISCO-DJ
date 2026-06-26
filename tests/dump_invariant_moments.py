"""Dump raw invariant moments of the deformation gradient for the theory agent.

For each (scale-free sim, snapshot a, LPT order, stride) compute the per-tet
deformation gradient ``F = ∂x/∂q`` and aggregate the ensemble moments of
the standard tensor invariants and their products:

  first-moment (leading σ² check):
    ⟨I_1⟩      ≈ 0
    ⟨I_1²⟩    = σ²_lin(R)         [Gaussian-LPT leading order]
    ⟨tr(G²)⟩
    ⟨tr(Σ²)⟩  (shear)

  second-moment (tree-level σ⁴ checks — the rationals the theory agent wants):
    ⟨I_1·I_2⟩
    ⟨I_1²·I_2⟩
    ⟨I_2²⟩
    ⟨I_1²·tr(G²)⟩
    ⟨I_1²·tr(Σ²)⟩
    ⟨I_1·I_3⟩
    ⟨tr(Σ³)⟩
    ⟨antisym_sq⟩        (= 0 for potential flow; diagnostic only)

Output:
  docs/pss_invariant_moments.json   — full table (all rows)
  docs/pss_invariant_moments.md     — markdown view for browsing
"""

from __future__ import annotations

import json, os, time

import jax, jax.numpy as jnp
import numpy as np

from discodj import DiscoDJ
from discodj.analysis import (
    kuhn_tet_deformation_tensor_from_positions,
    strain_invariants_from_F,
)
from discodj.analysis.sim_store import SimSnapshot


SF_N_VALUES = (("-2.25", -2.25), ("-2.0", -2.0), ("-1.5", -1.5))
A_TARGETS  = [0.02, 0.05, 0.10, 0.20, 0.35, 0.55, 0.85]
N_PART     = 128
BOXSIZE    = 128.0
SEED       = 1234
STRIDES    = [1, 2, 4, 8]
LPT_ORDERS = [1, 2, 3, 5]        # drop 10 — numerically unstable at large σ


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


def snap_from_positions(x_any, a: float, L: float, N: int) -> SimSnapshot:
    arr = np.asarray(x_any).astype(np.float32)
    if arr.ndim == 4:
        arr = arr.reshape(N**3, 3)
    return SimSnapshot(x=arr, v=np.zeros_like(arr),
                        a=float(a), N=N, L=L,
                        meta=dict(N=N, L=L, scale_free=True))


def _eulerian_positions_from_snap(snap):
    """Replicate the private helper in phase_space_sheet to get unwrapped ψ.
    We need the position mesh as a JAX array to feed into the jitted F kernel.
    """
    from discodj.analysis.sim_store import displacement_field
    psi = displacement_field(snap, as_mesh=True)
    h = snap.L / snap.N
    ix, iy, iz = np.meshgrid(np.arange(snap.N), np.arange(snap.N),
                              np.arange(snap.N), indexing="ij")
    q = np.stack([ix * h, iy * h, iz * h], axis=-1).astype(np.float32)
    return jnp.asarray(q + psi.astype(np.float32))


def moments_of_invariants(F: jnp.ndarray) -> dict:
    """Per-tet invariants → ensemble means of the standard products.

    Returns a dict of float moments; all shapes (6, Nc, Nc, Nc) are
    flattened to one long array before averaging.
    """
    inv = strain_invariants_from_F(F)
    I1 = np.asarray(inv["I1"]).ravel()
    I2 = np.asarray(inv["I2"]).ravel()
    I3 = np.asarray(inv["I3"]).ravel()
    trG2 = np.asarray(inv["trG2"]).ravel()
    trS2 = np.asarray(inv["trS2"]).ravel()
    trS3 = np.asarray(inv["trS3"]).ravel()
    asq = np.asarray(inv["antisym_sq"]).ravel()
    # first-order (σ²)
    out = {}
    out["I1_mean"]         = float(I1.mean())
    out["I1sq_mean"]       = float((I1 * I1).mean())
    out["I2_mean"]         = float(I2.mean())
    out["I3_mean"]         = float(I3.mean())
    out["trG2_mean"]       = float(trG2.mean())
    out["trS2_mean"]       = float(trS2.mean())
    # second-order (σ⁴) — what the theory agent asked for
    out["I1_I2_mean"]      = float((I1 * I2).mean())
    out["I1sq_I2_mean"]    = float((I1 * I1 * I2).mean())
    out["I2sq_mean"]       = float((I2 * I2).mean())
    out["I1sq_trG2_mean"]  = float((I1 * I1 * trG2).mean())
    out["I1sq_trS2_mean"]  = float((I1 * I1 * trS2).mean())
    out["I1_I3_mean"]      = float((I1 * I3).mean())
    out["trS3_mean"]       = float(trS3.mean())
    out["antisym_sq_mean"] = float(asq.mean())
    # additional σ⁶ 6-field traceful moments (requested by theory agent
    # for the A_0 σ⁶ accounting).
    out["I1_I2sq_mean"]    = float((I1 * I2 * I2).mean())
    out["I1sq_I3_mean"]    = float((I1 * I1 * I3).mean())
    out["I2_I3_mean"]      = float((I2 * I3).mean())
    out["I2_cubed_mean"]   = float((I2 * I2 * I2).mean())
    # Also useful: ⟨I_3²⟩ and ⟨I_1 I_2 I_3⟩ — complete the 3-invariant
    # cubic algebra at the next order.
    out["I3sq_mean"]       = float((I3 * I3).mean())
    out["I1_I2_I3_mean"]   = float((I1 * I2 * I3).mean())
    # useful for Gaussian checks
    out["I1_cubed_mean"]   = float((I1 * I1 * I1).mean())
    return out


def process_sim(n_tag: str, n_s: float, orders, a_targets, strides):
    sigma8 = sigma8_for_unit_variance_at_d(n_s)
    gamma_true = -(n_s + 3.0)
    print(f"\n=== SF n={n_s}  γ={gamma_true:+.2f}  sigma8={sigma8:.4f} ===")
    dj = make_dj(n_s, sigma8, SEED, max(orders))

    rows = []
    for a in a_targets:
        for order in orders:
            t0 = time.time()
            x_flat = dj.evaluate_lpt_pos_at_a(float(a), n_order=order)
            jax.block_until_ready(x_flat)
            snap = snap_from_positions(x_flat, a, BOXSIZE, N_PART)
            x_mesh = _eulerian_positions_from_snap(snap)
            for s in strides:
                F = kuhn_tet_deformation_tensor_from_positions(
                    x_mesh, float(BOXSIZE), stride=s, overlapping=True)
                m = moments_of_invariants(F)
                d_c = BOXSIZE / (N_PART // s)
                R_lag = (d_c**3 * 3/(4*np.pi))**(1/3)
                s2_lin = sigma8**2 * (8.0 / R_lag)**(n_s + 3.0) * a**2
                row = dict(
                    sim=f"sf_n{n_tag}", n_s=n_s, gamma=gamma_true,
                    a=float(a), lpt_order=order, stride=s,
                    R_lag=R_lag, sigma2_lin=float(s2_lin),
                    sigma_lin=float(np.sqrt(max(s2_lin, 0))),
                    **m,
                )
                rows.append(row)
            dt = time.time() - t0
            print(f"  a={a:.3f}  order={order}  "
                  f"⟨I₁²⟩ last: {rows[-1]['I1sq_mean']:.3e}  "
                  f"σ²_lin last: {rows[-1]['sigma2_lin']:.3e}  ({dt:.1f}s)")
    return rows


def emit_markdown(rows, path):
    lines = ["# Invariant moments of the deformation gradient F = ∂x/∂q\n"]
    lines.append(
        "Per-tet ensemble moments from pure-nLPT scale-free sims. "
        "``F`` is the deformation gradient, ``G = F − 1``, ``Σ`` is the "
        "traceless symmetric part of ``G``. Averages taken over the full "
        "(6, N/s, N/s, N/s) ensemble of Kuhn tets at the given stride.\n"
    )
    lines.append(
        "## Leading-σ scaling expectations (schematic)\n"
        "- ``⟨I₁⟩ ≈ 0`` — mean-zero by construction.\n"
        "- ``⟨I₁²⟩ ∝ σ²_lin``, ``⟨tr(G²)⟩ ∝ σ²_lin`` — tree-level σ² pieces.\n"
        "- ``⟨I₁·I₂⟩, ⟨I₁²·I₂⟩, ⟨I₂²⟩, ⟨I₁²·tr(G²)⟩ ∝ σ⁴_lin`` — tensor-"
        "algebra rationals live here.\n"
        "- ``⟨antisym_sq⟩ = 0`` for potential flow (2LPT-pure); diagnostic.\n"
    )
    cols = ["sim", "gamma", "lpt_order", "a", "stride",
            "sigma2_lin", "R_lag",
            "I1_mean", "I1sq_mean", "I1_cubed_mean",
            "trG2_mean", "trS2_mean", "trS3_mean",
            "I1_I2_mean", "I1sq_I2_mean", "I2sq_mean",
            "I1sq_trG2_mean", "I1sq_trS2_mean", "I1_I3_mean",
            "I1_I2sq_mean", "I1sq_I3_mean", "I2_I3_mean",
            "I2_cubed_mean", "I3sq_mean", "I1_I2_I3_mean",
            "antisym_sq_mean"]
    lines.append("## Full table (subset of most relevant columns)\n")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for r in sorted(rows, key=lambda r: (r["gamma"], r["lpt_order"],
                                           r["a"], r["stride"])):
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, str): vals.append(v)
            elif isinstance(v, int): vals.append(str(v))
            else: vals.append(f"{v:+.4e}" if abs(v) >= 1e-3 else f"{v:+.2e}")
        lines.append("| " + " | ".join(vals) + " |")
    lines.append(
        "\n## Derived ratios at 2LPT (quick algebraic cross-checks)\n"
        "Tree-level scaling predicts the following ratios are σ-independent\n"
        "(up to small corrections):\n\n"
        "- ``⟨I₁²⟩ / σ²_lin``  — should be 1 (by definition of σ).\n"
        "- ``⟨I₁·I₂⟩ / ⟨I₁²⟩²`` — rational proportional to `(d−1)/d = 2/3`.\n"
        "- ``⟨I₁²·I₂⟩ / ⟨I₁²⟩³`` — cubic tensor rational.\n"
        "- ``⟨tr(Σ²)⟩ / ⟨I₁²⟩`` — shear-vs-trace ratio (depends on γ).\n"
        "- ``⟨antisym_sq⟩ / ⟨I₁²⟩`` — curl fraction (should be ~0 at 2LPT).\n"
    )
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    os.makedirs("docs", exist_ok=True)
    rows = []
    for n_tag, n_s in SF_N_VALUES:
        rows += process_sim(n_tag, n_s, LPT_ORDERS, A_TARGETS, STRIDES)
    with open("docs/pss_invariant_moments.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved docs/pss_invariant_moments.json ({len(rows)} rows)")
    emit_markdown(rows, "docs/pss_invariant_moments.md")
    print("Saved docs/pss_invariant_moments.md")


if __name__ == "__main__":
    main()
