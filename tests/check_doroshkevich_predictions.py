"""Cross-check exact continuum Doroshkevich / Wick-level predictions.

Three independent validations against our PSS pipeline at stride=1:

1. Per-tet Jacobian skewness: κ_3(J) / σ²(J)² → 8/7? 34/7?
   The theory-agent target is **34/7 ≈ 4.857** (isotropic-Gaussian tree-level).
   Note: we measure S_3^V on V_cube = Σ of 6 tets per Lagrangian cube, which
   averages and reduces skewness. The per-tet value is expected to be larger.

2. Void / sheet / filament / halo fractions — Doroshkevich classification.
   At each tet, diagonalise the symmetric strain ``G_sym = (G + G^T)/2``, count
   how many eigenvalues are negative (= axes currently collapsing in the
   sign convention G_ij = ∂ψ_i/∂q_j). For Gaussian G the fractions are
   {0-, 1-, 2-, 3-} = {8.08, 41.92, 41.92, 8.08}% at any σ; 2LPT adds
   σ²-suppressed corrections.

3. First-crossing σ²: the smallest σ²_lin at which each Lagrangian tet
   first has J<0 (i.e. flips orientation). Distribution should track the
   Doroshkevich integral over the largest-eigenvalue tail.

Input: scale-free SF sims (3 γ), 5 seeds, pure-LPT positions at
7 scale factors a ∈ {0.02, 0.05, 0.10, 0.20, 0.35, 0.55, 0.85}.

Outputs:
  docs/doroshkevich_check.json
  docs/doroshkevich_check.png
"""

from __future__ import annotations

import json, os, time
from functools import partial

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax, jax.numpy as jnp

from discodj import DiscoDJ
from discodj.analysis import kuhn_tet_deformation_tensor_from_positions


SF_N_VALUES = (("-2.25", -2.25), ("-2.0", -2.0), ("-1.5", -1.5))
A_TARGETS  = [0.02, 0.05, 0.10, 0.20, 0.35, 0.55, 0.85]
SEEDS      = [1234, 2345, 3456, 4567, 5678]
N_PART     = 128
BOXSIZE    = 128.0
LPT_ORDER  = 2


def sigma8_for_unit_variance_at_d(n: float) -> float:
    return 8.0 ** (-(n + 3.0) / 2.0)


def build_cosmo(n_s: float, sigma8: float) -> dict:
    return dict(Omega_c=1.0, Omega_b=0.0, Omega_k=0.0,
                h=1.0, n_s=float(n_s), sigma8=float(sigma8),
                w0=-1.0, wa=0.0)


def make_dj(n_s: float, sigma8: float, seed: int):
    return (DiscoDJ(dim=3, res=N_PART, boxsize=BOXSIZE,
                     cosmo=build_cosmo(n_s, sigma8))
            .with_timetables()
            .with_linear_ps(transfer_function="none")
            .with_ics(seed=seed, white_noise_space="fourier",
                      k_order_fourier="stable")
            .with_lpt(n_order=LPT_ORDER))


def lpt_positions_mesh(dj, a: float) -> jnp.ndarray:
    x_flat = dj.evaluate_lpt_pos_at_a(float(a), n_order=LPT_ORDER)
    L = float(dj._boxsize); N = int(dj._res)
    x_mesh = np.asarray(x_flat).reshape(N, N, N, 3).astype(np.float32)
    ix, iy, iz = np.meshgrid(np.arange(N), np.arange(N), np.arange(N),
                              indexing="ij")
    h = L / N
    q = np.stack([ix * h, iy * h, iz * h], axis=-1).astype(np.float32)
    psi = x_mesh - q
    psi = psi - L * np.round(psi / L)
    return jnp.asarray(q + psi)


@jax.jit
def tet_J_and_classify(F: jnp.ndarray):
    """Compute per-tet J and classify by # negative eigenvalues of G_sym."""
    J = jnp.linalg.det(F)                                # (6, N, N, N)
    eye = jnp.eye(3, dtype=F.dtype)
    G = F - eye
    G_sym = 0.5 * (G + jnp.swapaxes(G, -1, -2))
    eigvals = jnp.linalg.eigvalsh(G_sym)                 # (6, N, N, N, 3)
    n_neg = (eigvals < 0).sum(axis=-1)                   # {0, 1, 2, 3}
    return J, n_neg


def run_one(n_tag: str, n_s: float, seed: int):
    """Single (sim, seed) run: sweep a, accumulate per-tet stats."""
    sigma8 = sigma8_for_unit_variance_at_d(n_s)
    dj = make_dj(n_s, sigma8, seed)
    gamma = -(n_s + 3.0)

    # Track first crossing: a at which each Lagrangian tet first has J<0.
    a_first = np.full((6, N_PART, N_PART, N_PART), np.nan, dtype=np.float32)

    rows = []
    for a in A_TARGETS:
        x = lpt_positions_mesh(dj, a)
        F = kuhn_tet_deformation_tensor_from_positions(
            x, float(BOXSIZE), stride=1, overlapping=True)
        J, n_neg = tet_J_and_classify(F)
        J = np.asarray(J)
        n_neg = np.asarray(n_neg)

        # Moments of J (per-tet)
        m_J = float(J.mean())
        dJ = J - m_J
        c2 = float((dJ**2).mean())
        c3 = float((dJ**3).mean())
        c4 = float((dJ**4).mean()) - 3.0 * c2**2
        sigma2_J = c2
        S3_J = c3 / sigma2_J**2 if sigma2_J > 0 else float("nan")

        # Eigenvalue class fractions
        fracs = [float((n_neg == k).mean()) for k in range(4)]

        # σ²_lin at the stride-1 tet Lagrangian radius R_lag
        d_grid = BOXSIZE / N_PART
        R_lag = (d_grid**3 * 3/(4*np.pi))**(1/3)
        s2_lin = sigma8**2 * (8.0/R_lag)**(n_s + 3.0) * a**2

        # Collapse fraction (J<0)
        frac_J_neg = float((J < 0).mean())

        # Update a_first
        new_flips = (J < 0) & np.isnan(a_first)
        a_first[new_flips] = float(a)

        rows.append(dict(
            n_tag=n_tag, n_s=n_s, gamma=gamma, seed=seed, a=float(a),
            sigma2_lin=float(s2_lin), sigma_lin=float(np.sqrt(max(s2_lin, 0))),
            sigma2_J=sigma2_J, sigma_J=float(np.sqrt(max(sigma2_J, 0))),
            mean_J=m_J, kappa3_J=c3, kappa4_J=c4, S3_J=S3_J,
            frac_0neg=fracs[0], frac_1neg=fracs[1],
            frac_2neg=fracs[2], frac_3neg=fracs[3],
            frac_J_negative=frac_J_neg,
        ))
        print(f"    a={a:.2f}  σ_J={np.sqrt(max(sigma2_J,0)):.3f}  "
              f"⟨J⟩={m_J:.4f}  S_3^(J)={S3_J:+.3f}  "
              f"frac(J<0)={frac_J_neg:.4f}  "
              f"classes={fracs[0]:.3f}/{fracs[1]:.3f}/{fracs[2]:.3f}/{fracs[3]:.3f}")

    # Wrap up: first-crossing histogram (σ²_lin at which each tet first flipped).
    d_grid = BOXSIZE / N_PART
    R_lag = (d_grid**3 * 3/(4*np.pi))**(1/3)
    sigma8_val = sigma8
    # a_first → σ²_lin = sigma8² · (8/R)^(n+3) · a_first²
    s2_at_first = sigma8_val**2 * (8.0/R_lag)**(n_s + 3.0) * a_first**2
    flipped = ~np.isnan(a_first)
    s2_at_first_flat = s2_at_first[flipped]

    return rows, s2_at_first_flat


def main():
    os.makedirs("docs", exist_ok=True)
    all_rows = []
    first_crossing = {}      # {(n_tag, seed) -> array of σ²_at_first_flip}

    for seed in SEEDS:
        for n_tag, n_s in SF_N_VALUES:
            print(f"\n=== SF n={n_s}  γ={-(n_s+3.0):+.2f}  seed={seed} ===")
            t0 = time.time()
            rows, s2_first = run_one(n_tag, n_s, seed)
            all_rows.extend(rows)
            first_crossing[(n_tag, seed)] = s2_first
            print(f"    ({time.time()-t0:.0f}s)")

    with open("docs/doroshkevich_check.json", "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nSaved docs/doroshkevich_check.json ({len(all_rows)} rows)")

    # ------- Plots -------
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (1) S_3^(J) vs σ²_J — target 34/7 ≈ 4.857
    ax = axes[0, 0]
    colors = {-0.75: "tab:blue", -1.00: "tab:orange", -1.50: "tab:red"}
    for g, color in colors.items():
        sub = [r for r in all_rows if abs(r["gamma"] - g) < 0.01]
        # group by a: mean ± std over seeds
        a_vals = sorted(set(r["a"] for r in sub))
        x = []; y = []; yerr = []
        for a in a_vals:
            a_rows = [r for r in sub if abs(r["a"] - a) < 1e-4]
            s2s = np.array([r["sigma2_J"] for r in a_rows])
            S3s = np.array([r["S3_J"] for r in a_rows if np.isfinite(r["S3_J"])])
            if len(S3s) == 0: continue
            x.append(np.mean(s2s)); y.append(np.mean(S3s))
            yerr.append(np.std(S3s) / np.sqrt(len(S3s)))
        ax.errorbar(x, y, yerr=yerr, fmt="o-", color=color, ms=6,
                     label=fr"γ_tom={g:+.2f}  (γ_paper={-g:+.2f})")
    ax.axhline(34/7, ls="--", color="black",
                label=r"theory: $S_3^{(J)} = 34/7 \approx 4.857$")
    ax.axhline(8/7, ls=":", color="grey", alpha=0.5,
                label=r"V_cube reference: 8/7")
    ax.set_xlabel(r"$\sigma^2_{J}$ (per-tet Jacobian variance)")
    ax.set_ylabel(r"$S_3^{(J)} = \kappa_3(J)/\sigma^4_J$")
    ax.set_title("Per-tet Jacobian skewness vs σ² — target 34/7")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_xscale("log")

    # (2) Eigenvalue class fractions vs σ_lin
    ax = axes[0, 1]
    for k in range(4):
        labels_k = ["0 neg (void)", "1 neg (sheet)",
                    "2 neg (filament)", "3 neg (halo)"]
        colors_k = ["tab:cyan", "tab:green", "tab:olive", "tab:brown"]
        # Aggregate over all sims and seeds for each a.
        a_vals = sorted(set(r["a"] for r in all_rows))
        x = []; y = []; yerr = []
        for a in a_vals:
            a_rows = [r for r in all_rows if abs(r["a"] - a) < 1e-4]
            s2s = np.array([r["sigma2_lin"] for r in a_rows])
            fracs = np.array([r[f"frac_{k}neg"] for r in a_rows])
            x.append(np.mean(s2s)); y.append(np.mean(fracs))
            yerr.append(np.std(fracs) / np.sqrt(len(fracs)))
        ax.errorbar(x, y, yerr=yerr, fmt="o-", color=colors_k[k], ms=5,
                     label=labels_k[k])
    # Doroshkevich asymptotic values
    for y_target, lbl in [(0.0808, "8.08% (void/halo)"),
                           (0.4192, "41.92% (sheet/filament)")]:
        ax.axhline(y_target, ls="--", color="grey", alpha=0.5)
    ax.set_xlabel(r"$\sigma^2_{\rm lin}$")
    ax.set_ylabel("fraction of tets")
    ax.set_title("Eigenvalue classification — Doroshkevich fractions (8/42/42/8)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_xscale("log")

    # (3) Fraction (J<0) vs σ — triaxial collapse
    ax = axes[1, 0]
    for g, color in colors.items():
        sub = [r for r in all_rows if abs(r["gamma"] - g) < 0.01]
        a_vals = sorted(set(r["a"] for r in sub))
        x = []; y = []; yerr = []
        for a in a_vals:
            a_rows = [r for r in sub if abs(r["a"] - a) < 1e-4]
            s2s = np.array([r["sigma2_lin"] for r in a_rows])
            f_neg = np.array([r["frac_J_negative"] for r in a_rows])
            x.append(np.mean(s2s)); y.append(np.mean(f_neg))
            yerr.append(np.std(f_neg) / np.sqrt(len(f_neg)))
        ax.errorbar(x, y, yerr=yerr, fmt="o-", color=color, ms=6,
                     label=fr"γ_paper={-g:+.2f}")
    ax.set_xlabel(r"$\sigma^2_{\rm lin}$")
    ax.set_ylabel(r"fraction of tets with $J<0$ (shell-crossed)")
    ax.set_title("Fraction of inverted tets vs σ²")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_xscale("log")

    # (4) First-crossing σ² distribution (across seeds & γ)
    ax = axes[1, 1]
    for g_tom, color in colors.items():
        # pool across seeds
        n_tag_map = {-0.75: "-2.25", -1.00: "-2.0", -1.50: "-1.5"}
        pooled = []
        for seed in SEEDS:
            arr = first_crossing.get((n_tag_map[g_tom], seed))
            if arr is not None and arr.size > 0:
                pooled.append(arr)
        if not pooled: continue
        all_s2 = np.concatenate(pooled)
        if all_s2.size < 10: continue
        log_s2 = np.log(all_s2)
        ax.hist(log_s2, bins=40, density=True, histtype="step",
                  color=color, label=fr"γ_paper={-g_tom:+.2f}  (n={all_s2.size})")
    ax.set_xlabel(r"$\ln \sigma^2_{\rm lin}$ at first crossing")
    ax.set_ylabel("PDF")
    ax.set_title("First-crossing distribution")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle("Doroshkevich / Wick-level cross-checks — 2LPT PSS at N=128",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig("docs/doroshkevich_check.png", dpi=130)
    print("Saved docs/doroshkevich_check.png")


if __name__ == "__main__":
    main()
