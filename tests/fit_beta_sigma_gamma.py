"""Joint fit β_eff(σ², γ) = β₀ + β_σ σ² + β_γ γ  against PM J-PDFs.

Uses Planck-cosmology snapshots with σ² ≤ 0.3 from the N=256 suite (both
cutoff and nocut, all available a's). Saves the best-fit coefficients
and per-snapshot residuals to disk, and produces diagnostic plots.
"""

import json
import os
import time

import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize

from discodj import DiscoDJ
from discodj.analysis import (
    load_sim, deformation_gradient_jacobian,
    gamma_for_snapshot, reconstruct_pk,
)
from discodj.cascade import fem_cascade_v3, pk_table_from_discodj


SIGMA2_CAP = 0.3
N_CAS = 32
N_STEPS = 10
BOXSIZE = 250.0
SEEDS = 3
BETA_GRID = np.linspace(-4.0, 1.0, 21)    # scan grid
WEIGHTS = dict(sigma2=10.0, S3=1.0, k4=0.3)


def Jmom(J):
    a = np.asarray(J).ravel()
    m = a.mean(); d = a - m
    v = float(np.mean(d * d))
    s3 = float(np.mean(d**3) / max(v, 1e-30)**1.5)
    k4 = float(np.mean(d**4) / max(v, 1e-30)**2 - 3.0)
    return float(m), v, s3, k4


def build_cascade_pk(snap):
    """Return (pk_k, pk_Pk) matching the snapshot's IC power spectrum."""
    # reconstruct_pk is a Planck-default fallback — ok for the Planck suite
    # but for the extreme suite we need the cosmo dict from meta.
    from discodj.analysis.spectral_slope import reconstruct_pk
    k_np, Pk_np = reconstruct_pk(snap)
    return jnp.asarray(k_np), jnp.asarray(Pk_np)


def scan_beta(snap, sigma2, pk_k, pk_Pk, betas):
    """Return arrays σ²_J_cas, S₃_cas, κ₄_cas aligned with betas."""
    s2s = np.empty_like(betas); S3s = np.empty_like(betas); k4s = np.empty_like(betas)
    for i, beta in enumerate(betas):
        vs, s3s, k4s_ = [], [], []
        for seed in range(SEEDS):
            key = jax.random.PRNGKey(500 + seed)
            r = fem_cascade_v3(sigma2=float(sigma2), n_steps=N_STEPS, N=N_CAS,
                               boxsize=BOXSIZE, key=key,
                               g_T=0.0, alpha_nl=0.0, beta_nl=float(beta),
                               pk_k=pk_k, pk_Pk=pk_Pk,
                               method="midpoint", use_abs_J=True)
            jax.block_until_ready(r.J)
            _, v, s3, k4 = Jmom(r.J)
            vs.append(v); s3s.append(s3); k4s_.append(k4)
        s2s[i] = np.mean(vs); S3s[i] = np.mean(s3s); k4s[i] = np.mean(k4s_)
    return s2s, S3s, k4s


def collect_data():
    """For each Planck snap with σ²_trG ≤ SIGMA2_CAP, build scan table."""
    table = []
    for cm in ("cutoff", "nocutoff"):
        for a_pm in (0.03, 0.05, 0.07, 0.10, 0.30):
            try:
                snap = load_sim(256, cm, a=a_pm)
            except FileNotFoundError:
                continue
            G_pm, J_pm = deformation_gradient_jacobian(snap)
            s2 = float(jnp.var(jnp.einsum("...ii->...", G_pm)))
            if s2 > SIGMA2_CAP:
                print(f"  [skip] {cm} a={snap.a:.2f} σ²={s2:.3f} > cap")
                continue
            m_pm, v_pm, S3_pm, k4_pm = Jmom(J_pm)
            sig2_g, R, gamma = gamma_for_snapshot(snap, sigma2_trG=s2)
            pk_k, pk_Pk = build_cascade_pk(snap)
            t0 = time.time()
            s2s_cas, S3s_cas, k4s_cas = scan_beta(snap, s2, pk_k, pk_Pk, BETA_GRID)
            dt = time.time() - t0
            print(f"  {cm} a={snap.a:.2f}  σ²={s2:.3f}  γ={gamma:+.2f}  "
                  f"S₃_PM={S3_pm:+.3f}  scan {dt:.0f}s")
            table.append(dict(
                cm=cm, a=float(snap.a), sigma2=s2, gamma=gamma, R_eff=R,
                S3_PM=S3_pm, k4_PM=k4_pm, v_PM=v_pm,
                betas=BETA_GRID.tolist(),
                sigma2_J_cas=s2s_cas.tolist(),
                S3_cas=S3s_cas.tolist(),
                k4_cas=k4s_cas.tolist(),
            ))
    return table


def residual_loss(params, table):
    """Weighted squared error over all snapshots, interpolating
    cascade(β) onto β_eff(σ², γ) for each snap."""
    b0, b_s, b_g = params
    total = 0.0
    for row in table:
        b_eff = b0 + b_s * row["sigma2"] + b_g * row["gamma"]
        betas = np.asarray(row["betas"])
        s2c = np.interp(b_eff, betas, row["sigma2_J_cas"])
        S3c = np.interp(b_eff, betas, row["S3_cas"])
        k4c = np.interp(b_eff, betas, row["k4_cas"])
        total += WEIGHTS["sigma2"] * ((s2c - row["v_PM"]) / max(row["v_PM"], 1e-9))**2
        total += WEIGHTS["S3"] * (S3c - row["S3_PM"])**2
        total += WEIGHTS["k4"] * (k4c - row["k4_PM"])**2
    return total


def main():
    os.makedirs("docs", exist_ok=True)

    print("Collecting cascade scans...")
    table = collect_data()

    # Initial guess: β₀=-1.5 (our single-param fit), β_σ=0, β_γ=0.
    x0 = np.array([-1.5, 0.0, 0.0])
    res = minimize(residual_loss, x0, args=(table,), method="Nelder-Mead",
                   options=dict(xatol=1e-4, fatol=1e-6, maxiter=3000))
    b0, b_s, b_g = res.x
    print(f"\nBest fit: β_eff = {b0:+.3f} + ({b_s:+.3f})·σ² + ({b_g:+.3f})·γ")
    print(f"  loss = {res.fun:.4e}, nit = {res.nit}")

    # Report per-snapshot residuals.
    print(f"\n{'suite':>10} {'a':>5} | {'σ²':>6} {'γ':>6} {'β_eff':>7} | "
          f"{'σ²_J':>8} {'S₃':>7} {'κ₄':>7} | "
          f"{'Δσ²_J/σ²_J':>11} {'ΔS₃':>7} {'Δκ₄':>7}")
    for row in table:
        b_eff = b0 + b_s * row["sigma2"] + b_g * row["gamma"]
        betas = np.asarray(row["betas"])
        s2c = np.interp(b_eff, betas, row["sigma2_J_cas"])
        S3c = np.interp(b_eff, betas, row["S3_cas"])
        k4c = np.interp(b_eff, betas, row["k4_cas"])
        print(f"{row['cm']:>10} {row['a']:5.2f} | {row['sigma2']:6.3f} "
              f"{row['gamma']:+6.2f} {b_eff:+7.2f} | "
              f"{s2c:8.4f} {S3c:+7.3f} {k4c:+7.3f} | "
              f"{(s2c-row['v_PM'])/row['v_PM']*100:+10.1f}% "
              f"{S3c-row['S3_PM']:+7.3f} {k4c-row['k4_PM']:+7.3f}")

    # Save results to disk for plotting + validation
    out = dict(beta_0=float(b0), beta_sigma=float(b_s), beta_gamma=float(b_g),
               weights=WEIGHTS, table=table, sigma2_cap=SIGMA2_CAP,
               n_cascade=N_CAS, n_steps=N_STEPS, seeds=SEEDS)
    with open("docs/cascade_two_param_fit.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved docs/cascade_two_param_fit.json")


if __name__ == "__main__":
    main()
