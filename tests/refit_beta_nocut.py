"""Refit β_best vs ln σ for the nocut suite with branch selection.

The S₃(β) curve is non-monotonic for σ² ≳ 0.5 — it has a local minimum at
β ≈ -2 and reaches PM's S₃ on BOTH branches. The previous scan (β ∈
[-4, +1]) hit the LEFT branch at a=0.30 with β_best = -4, but that branch
has σ²_J off by +85%. The physically sensible branch is the RIGHT one,
where cascade σ²_J is within ~15% of PM and β stays near the linear-law
prediction.

Strategy: scan β ∈ [-4, +1] and pick β minimising the joint loss
  L(β) = w_σ · (σ²_J_cas/σ²_J_PM − 1)² + w_S3 · (S₃_cas − S₃_PM)²
but restrict to the right branch: β ≥ β* where β* = argmin_β S₃_cas(β).
If S₃_cas is monotonic (low-σ snaps), no restriction applies.

Output:
  docs/beta_nocut_refit.json        — per-snapshot β_best with branch-right flag
  docs/beta_vs_logsigma_refit.png   — β_best vs ln σ with linear fit
"""

import json, time

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax, jax.numpy as jnp

from discodj.analysis import load_sim, deformation_gradient_jacobian
from discodj.analysis.spectral_slope import reconstruct_pk
from discodj.cascade import fem_cascade_v3


N_CAS, N_STEPS, BOXSIZE = 32, 10, 250.0
SEEDS = 3
BETA_GRID = np.linspace(-4.0, 1.0, 21)
W_SIGMA, W_S3 = 10.0, 1.0
A_LIST = [0.03, 0.05, 0.07, 0.10, 0.30, 0.50, 0.70, 1.0]


def Jmom(J):
    a = np.asarray(J).ravel()
    m = a.mean(); d = a - m
    v = float(np.mean(d * d))
    s3 = float(np.mean(d**3) / max(v, 1e-30)**1.5)
    return float(m), v, s3


def scan(snap, sigma2, pk_k, pk_Pk, betas):
    vs, S3s = np.empty_like(betas), np.empty_like(betas)
    for i, b in enumerate(betas):
        vi, si = [], []
        for s in range(SEEDS):
            r = fem_cascade_v3(sigma2=sigma2, n_steps=N_STEPS, N=N_CAS,
                               boxsize=BOXSIZE, key=jax.random.PRNGKey(700 + s),
                               g_T=0.0, alpha_nl=0.0, beta_nl=float(b),
                               pk_k=pk_k, pk_Pk=pk_Pk,
                               method="midpoint", use_abs_J=True)
            jax.block_until_ready(r.J)
            _, v, s3 = Jmom(r.J)
            vi.append(v); si.append(s3)
        vs[i] = np.mean(vi); S3s[i] = np.mean(si)
    return vs, S3s


def find_beta_best(betas, vs_cas, S3s_cas, v_pm, S3_pm):
    """Joint-loss best β, restricted to the right branch if S₃(β) is non-monotone."""
    # Right branch starts at the index of the S₃ minimum (may be idx 0 for
    # monotonic curves).
    imin = int(np.argmin(S3s_cas))
    mask = np.arange(len(betas)) >= imin
    rel_v = (vs_cas - v_pm) / max(v_pm, 1e-12)
    err_S3 = S3s_cas - S3_pm
    loss = W_SIGMA * rel_v**2 + W_S3 * err_S3**2
    loss_masked = np.where(mask, loss, np.inf)
    ibest = int(np.argmin(loss_masked))
    branch_right = imin > 0  # True if we actually restricted to a branch
    return float(betas[ibest]), branch_right, loss_masked, imin


def main():
    rows = []
    for a_pm in A_LIST:
        try:
            snap = load_sim(256, "nocutoff", a=a_pm)
        except FileNotFoundError:
            print(f"  [skip] a={a_pm}: not found"); continue
        G_pm, J_pm = deformation_gradient_jacobian(snap)
        s2 = float(jnp.var(jnp.einsum("...ii->...", G_pm)))
        _, v_pm, S3_pm = Jmom(J_pm)
        k_np, Pk_np = reconstruct_pk(snap)
        pk_k, pk_Pk = jnp.asarray(k_np), jnp.asarray(Pk_np)
        t0 = time.time()
        vs_cas, S3s_cas = scan(snap, s2, pk_k, pk_Pk, BETA_GRID)
        b_best, br, loss, imin = find_beta_best(BETA_GRID, vs_cas, S3s_cas, v_pm, S3_pm)
        dt = time.time() - t0
        print(f"  a={a_pm:.2f}  σ²={s2:.3f}  S₃_PM={S3_pm:+.3f}  "
              f"β_best={b_best:+.3f}  branch_right={br} (S₃ min at β={BETA_GRID[imin]:+.2f})  "
              f"({dt:.0f}s)")
        rows.append(dict(
            a=float(snap.a), sigma2=s2, S3_PM=S3_pm, v_PM=v_pm,
            beta_best=b_best, branch_right=br,
            beta_S3_min=float(BETA_GRID[imin]),
            betas=BETA_GRID.tolist(),
            sigma2_J_cas=vs_cas.tolist(),
            S3_cas=S3s_cas.tolist(),
        ))

    # Linear fit β = β₀ + β_ln · ln σ over σ² ≤ 0.3 (low-σ regime where curve is monotonic
    # but also extrapolates cleanly into a=0.30).
    ln_sig = np.array([0.5 * np.log(r["sigma2"]) for r in rows])
    b_best = np.array([r["beta_best"] for r in rows])
    mask_fit = np.array([r["sigma2"] for r in rows]) <= 0.8  # include a=0.30 (σ²=0.68)
    A = np.column_stack([np.ones(mask_fit.sum()), ln_sig[mask_fit]])
    coefs, *_ = np.linalg.lstsq(A, b_best[mask_fit], rcond=None)
    b0, b_ln = float(coefs[0]), float(coefs[1])
    resid = b_best[mask_fit] - (b0 + b_ln * ln_sig[mask_fit])
    rms = float(np.sqrt(np.mean(resid**2)))
    print(f"\nLinear fit over σ² ≤ 0.8: β = {b0:+.3f} + ({b_ln:+.3f}) · ln σ   "
          f"(rms = {rms:.3f}, n={mask_fit.sum()})")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    fit_pts = np.asarray(mask_fit, bool)
    ax.plot(ln_sig[fit_pts], b_best[fit_pts], "o", ms=8,
            color="tab:blue", label="nocut (fit)")
    ax.plot(ln_sig[~fit_pts], b_best[~fit_pts], "s", ms=8,
            color="tab:grey", alpha=0.6, label="nocut (excluded)")
    xs = np.linspace(ln_sig.min() - 0.1, ln_sig.max() + 0.1, 100)
    ax.plot(xs, b0 + b_ln * xs, "--", color="tab:red",
            label=f"β = {b0:+.2f} + ({b_ln:+.2f}) · ln σ  (rms={rms:.2f})")
    for r in rows:
        ax.annotate(f"a={r['a']:.2f}",
                    (0.5 * np.log(r["sigma2"]), r["beta_best"]),
                    fontsize=7, alpha=0.7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(r"$\ln \sigma$ (σ = $\sqrt{\mathrm{Var}\,\mathrm{tr}\,\delta G}$)")
    ax.set_ylabel(r"$\beta_{\rm nl, best}$")
    ax.set_title("Nocut suite: β_best vs ln σ (right-branch-selected)")
    ax.grid(alpha=0.3); ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig("docs/beta_vs_logsigma_refit.png", dpi=130)
    print(f"Saved docs/beta_vs_logsigma_refit.png")

    with open("docs/beta_nocut_refit.json", "w") as f:
        json.dump(dict(intercept=b0, slope=b_ln, rms=rms,
                       fit_sigma2_max=0.8, rows=rows), f, indent=2)
    print("Saved docs/beta_nocut_refit.json")


if __name__ == "__main__":
    main()
