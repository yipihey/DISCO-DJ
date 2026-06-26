"""Refined β fit with fine scan grid + parabolic refinement.

Scan β on a fine grid (step 0.05) for the low-σ nocut snapshots, locate
the grid minimum of the joint loss, then refine with a local parabolic
fit to get sub-grid accuracy. Outputs the refined linear law
β = β₀ + β_ln · ln σ and tests the 3/5 hypothesis.

Low-σ snapshots (a=0.03, 0.05, 0.07, 0.10) have monotonic S₃(β), so no
branch-selection needed. More seeds (SEEDS=5) for tighter moment estimates.
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
SEEDS = 5
BETA_GRID = np.arange(-2.50, -0.99, 0.05)  # 31 points
W_SIGMA, W_S3 = 10.0, 1.0
A_LIST = [0.03, 0.05, 0.07, 0.10]


def Jmom(J):
    a = np.asarray(J).ravel()
    m = a.mean(); d = a - m
    v = float(np.mean(d * d))
    s3 = float(np.mean(d**3) / max(v, 1e-30)**1.5)
    return float(m), v, s3


def scan(sigma2, pk_k, pk_Pk, betas):
    vs, S3s = np.empty_like(betas), np.empty_like(betas)
    for i, b in enumerate(betas):
        vi, si = [], []
        for s in range(SEEDS):
            r = fem_cascade_v3(sigma2=sigma2, n_steps=N_STEPS, N=N_CAS,
                               boxsize=BOXSIZE, key=jax.random.PRNGKey(3100 + s),
                               g_T=0.0, alpha_nl=0.0, beta_nl=float(b),
                               pk_k=pk_k, pk_Pk=pk_Pk,
                               method="midpoint", use_abs_J=True)
            jax.block_until_ready(r.J)
            _, v, s3 = Jmom(r.J)
            vi.append(v); si.append(s3)
        vs[i] = np.mean(vi); S3s[i] = np.mean(si)
    return vs, S3s


def refine_parabola(betas, loss):
    """Quadratic fit to the 3 points around the grid minimum → sub-grid β*.
    Returns (β*, L*)."""
    imin = int(np.argmin(loss))
    if imin == 0 or imin == len(betas) - 1:
        return float(betas[imin]), float(loss[imin])  # boundary — no refinement
    b = betas[imin - 1:imin + 2]
    L = loss[imin - 1:imin + 2]
    # fit L = A(β - β*)² + L*
    # using 3-point parabola formula:
    denom = (L[0] - 2 * L[1] + L[2])
    if abs(denom) < 1e-12:
        return float(betas[imin]), float(L[1])
    offset = 0.5 * (L[0] - L[2]) / denom * (b[1] - b[0])
    beta_star = float(b[1] + offset)
    L_star = float(L[1] - 0.25 * (L[0] - L[2])**2 / denom)
    return beta_star, L_star


def main():
    rows = []
    for a_pm in A_LIST:
        snap = load_sim(256, "nocutoff", a=a_pm)
        G_pm, J_pm = deformation_gradient_jacobian(snap)
        s2 = float(jnp.var(jnp.einsum("...ii->...", G_pm)))
        _, v_pm, S3_pm = Jmom(J_pm)
        k_np, Pk_np = reconstruct_pk(snap)
        pk_k, pk_Pk = jnp.asarray(k_np), jnp.asarray(Pk_np)
        t0 = time.time()
        vs_cas, S3s_cas = scan(s2, pk_k, pk_Pk, BETA_GRID)
        # joint loss
        rel_v = (vs_cas - v_pm) / max(v_pm, 1e-12)
        err_S3 = S3s_cas - S3_pm
        loss = W_SIGMA * rel_v**2 + W_S3 * err_S3**2
        b_best_grid = float(BETA_GRID[int(np.argmin(loss))])
        b_best_par, _ = refine_parabola(BETA_GRID, loss)
        # S₃-only minimum (for comparison)
        loss_S3 = err_S3**2
        b_S3_grid = float(BETA_GRID[int(np.argmin(loss_S3))])
        b_S3_par, _ = refine_parabola(BETA_GRID, loss_S3)
        dt = time.time() - t0
        print(f"  a={a_pm:.2f}  σ²={s2:.4f}  S₃_PM={S3_pm:+.3f}  "
              f"β_best(joint)={b_best_par:+.3f}  β_best(S₃-only)={b_S3_par:+.3f}  ({dt:.0f}s)")
        rows.append(dict(a=float(snap.a), sigma2=s2, S3_PM=S3_pm, v_PM=v_pm,
                         beta_best_joint=b_best_par, beta_best_S3=b_S3_par,
                         beta_grid_joint=b_best_grid, beta_grid_S3=b_S3_grid,
                         betas=BETA_GRID.tolist(),
                         sigma2_J_cas=vs_cas.tolist(),
                         S3_cas=S3s_cas.tolist()))

    # Linear fit both variants
    ln_sig = np.array([0.5 * np.log(r["sigma2"]) for r in rows])
    for var, key in [("joint (σ²+S₃)", "beta_best_joint"),
                     ("S₃-only", "beta_best_S3")]:
        y = np.array([r[key] for r in rows])
        A = np.column_stack([np.ones_like(ln_sig), ln_sig])
        coefs, *_ = np.linalg.lstsq(A, y, rcond=None)
        b0, b_ln = float(coefs[0]), float(coefs[1])
        resid = y - (b0 + b_ln * ln_sig)
        rms = float(np.sqrt(np.mean(resid**2)))
        print(f"\n[{var}]  β = {b0:+.4f} + ({b_ln:+.4f}) · ln σ   rms = {rms:.4f}")
        print(f"    slope vs 3/5 = 0.6000 : {b_ln - 0.6:+.4f}   "
              f"(fractional {(b_ln/0.6 - 1)*100:+.2f}%)")
        print(f"    slope vs 1/2 = 0.5000 : {b_ln - 0.5:+.4f}   "
              f"(fractional {(b_ln/0.5 - 1)*100:+.2f}%)")

    # Save + plot — use joint variant as headline
    y_joint = np.array([r["beta_best_joint"] for r in rows])
    A = np.column_stack([np.ones_like(ln_sig), ln_sig])
    coefs, *_ = np.linalg.lstsq(A, y_joint, rcond=None)
    b0, b_ln = float(coefs[0]), float(coefs[1])
    resid = y_joint - (b0 + b_ln * ln_sig)
    rms = float(np.sqrt(np.mean(resid**2)))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ln_sig, y_joint, "o", ms=9, color="tab:blue", label="β_best (joint)")
    y_S3 = np.array([r["beta_best_S3"] for r in rows])
    ax.plot(ln_sig, y_S3, "s", ms=7, color="tab:orange", alpha=0.7, label="β_best (S₃-only)")
    xs = np.linspace(ln_sig.min() - 0.2, ln_sig.max() + 0.2, 100)
    ax.plot(xs, b0 + b_ln * xs, "--", color="tab:red",
            label=fr"fit: $\beta = {b0:+.3f} + ({b_ln:+.3f})\ln\sigma$")
    # Reference lines
    for c, lbl in [(0.6, "3/5"), (0.5, "1/2"), (3/7, "3/7")]:
        # anchor at same intercept of the joint fit for visual
        ax.plot(xs, b0 + c * xs, ":", alpha=0.5, lw=1,
                label=fr"slope={c:.3f} ({lbl})")
    for r in rows:
        ax.annotate(f"a={r['a']:.2f}",
                    (0.5 * np.log(r["sigma2"]), r["beta_best_joint"]),
                    fontsize=8, alpha=0.7, xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel(r"$\ln \sigma$")
    ax.set_ylabel(r"$\beta_{\rm nl, best}$")
    ax.set_title("Nocut low-σ: refined β vs ln σ (parabolic-refined, SEEDS=5)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig("docs/beta_vs_logsigma_refined.png", dpi=130)
    print(f"\nSaved docs/beta_vs_logsigma_refined.png")

    with open("docs/beta_nocut_refined.json", "w") as f:
        json.dump(dict(intercept=b0, slope=b_ln, rms=rms,
                       slope_vs_three_fifths=b_ln - 0.6,
                       rows=rows), f, indent=2)
    print("Saved docs/beta_nocut_refined.json")


if __name__ == "__main__":
    main()
