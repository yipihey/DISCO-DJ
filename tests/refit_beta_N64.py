"""Full fine β-scan at N_CAS=64 — pin down the true β(ln σ) law.

S₃ is converged in N (see check_cascade_N_convergence.py), but to
eliminate any residual N-dependence from the fit we repeat the refined
scan at N_CAS=64 with SEEDS=5 and parabolic refinement.
"""

import json, time

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax, jax.numpy as jnp

from discodj.analysis import load_sim, deformation_gradient_jacobian
from discodj.analysis.spectral_slope import reconstruct_pk
from discodj.cascade import fem_cascade_v3


N_CAS, N_STEPS, BOXSIZE = 64, 10, 250.0
SEEDS = 5
BETA_GRID = np.arange(-2.30, -1.09, 0.05)  # 25 points, tighter range
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
                               boxsize=BOXSIZE, key=jax.random.PRNGKey(5200 + s),
                               g_T=0.0, alpha_nl=0.0, beta_nl=float(b),
                               pk_k=pk_k, pk_Pk=pk_Pk,
                               method="midpoint", use_abs_J=True)
            jax.block_until_ready(r.J)
            _, v, s3 = Jmom(r.J)
            vi.append(v); si.append(s3)
        vs[i] = np.mean(vi); S3s[i] = np.mean(si)
    return vs, S3s


def refine_parabola(betas, loss):
    imin = int(np.argmin(loss))
    if imin == 0 or imin == len(betas) - 1:
        return float(betas[imin])
    b = betas[imin - 1:imin + 2]
    L = loss[imin - 1:imin + 2]
    denom = (L[0] - 2 * L[1] + L[2])
    if abs(denom) < 1e-12:
        return float(betas[imin])
    offset = 0.5 * (L[0] - L[2]) / denom * (b[1] - b[0])
    return float(b[1] + offset)


def main():
    print(f"N_CAS={N_CAS}, SEEDS={SEEDS}, β grid step {BETA_GRID[1]-BETA_GRID[0]:.3f}")
    rows = []
    t_all = time.time()
    for a_pm in A_LIST:
        snap = load_sim(256, "nocutoff", a=a_pm)
        G_pm, J_pm = deformation_gradient_jacobian(snap)
        s2 = float(jnp.var(jnp.einsum("...ii->...", G_pm)))
        _, v_pm, S3_pm = Jmom(J_pm)
        k_np, Pk_np = reconstruct_pk(snap)
        pk_k, pk_Pk = jnp.asarray(k_np), jnp.asarray(Pk_np)
        t0 = time.time()
        vs_cas, S3s_cas = scan(s2, pk_k, pk_Pk, BETA_GRID)
        loss_S3 = (S3s_cas - S3_pm)**2
        b_S3 = refine_parabola(BETA_GRID, loss_S3)
        print(f"  a={a_pm:.2f}  σ²={s2:.4f}  S₃_PM={S3_pm:+.3f}  "
              f"β_best(S₃)={b_S3:+.3f}   ({time.time()-t0:.0f}s)")
        rows.append(dict(a=float(snap.a), sigma2=s2, S3_PM=S3_pm, v_PM=v_pm,
                         beta_best_S3=b_S3,
                         betas=BETA_GRID.tolist(),
                         sigma2_J_cas=vs_cas.tolist(),
                         S3_cas=S3s_cas.tolist()))

    # Linear fit S₃-only
    ln_sig = np.array([0.5 * np.log(r["sigma2"]) for r in rows])
    y = np.array([r["beta_best_S3"] for r in rows])
    A = np.column_stack([np.ones_like(ln_sig), ln_sig])
    coefs, *_ = np.linalg.lstsq(A, y, rcond=None)
    b0, b_ln = float(coefs[0]), float(coefs[1])
    resid = y - (b0 + b_ln * ln_sig)
    rms = float(np.sqrt(np.mean(resid**2)))
    SSx = float(np.sum((ln_sig - ln_sig.mean())**2))
    se_slope = float(np.sqrt((resid**2).sum() / (len(ln_sig) - 2) / SSx))
    se_intercept = float(se_slope * np.sqrt((ln_sig**2).mean()))

    print(f"\n[N_CAS={N_CAS}, S₃-only]  β = {b0:+.4f}(±{se_intercept:.4f}) + "
          f"({b_ln:+.4f}±{se_slope:.4f}) · ln σ   rms = {rms:.4f}")
    for c, lbl in [(3/5, "3/5"), (5/9, "5/9"), (4/7, "4/7"), (1/2, "1/2"),
                   (3/7, "3/7")]:
        sigma_devs = (b_ln - c) / se_slope
        print(f"    slope vs {lbl:>5} = {c:.4f} : {b_ln - c:+.4f}  ({sigma_devs:+.2f}σ)")
    for c, lbl in [(-3/5, "-3/5"), (-4/7, "-4/7"), (-1/2, "-1/2"), (-2/3, "-2/3")]:
        sigma_devs = (b0 - c) / se_intercept
        print(f"    intercept vs {lbl:>5} = {c:+.4f}: {b0 - c:+.4f}  ({sigma_devs:+.2f}σ)")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(ln_sig, y, yerr=rms, fmt="o", ms=9, color="tab:blue",
                label=f"N_CAS={N_CAS} β_best (S₃-only)")
    xs = np.linspace(ln_sig.min() - 0.2, ln_sig.max() + 0.2, 100)
    ax.plot(xs, b0 + b_ln * xs, "--", color="tab:red",
            label=fr"fit: $\beta = {b0:+.3f} + ({b_ln:+.3f})\ln\sigma$")
    # Reference: fixed-intercept candidates
    for c, lbl in [(3/5, "3/5"), (5/9, "5/9"), (4/7, "4/7")]:
        ax.plot(xs, b0 + c * xs, ":", alpha=0.6, lw=1,
                label=f"slope={c:.4f} ({lbl})")
    for r in rows:
        ax.annotate(f"a={r['a']:.2f}",
                    (0.5 * np.log(r["sigma2"]), r["beta_best_S3"]),
                    fontsize=8, alpha=0.7, xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel(r"$\ln \sigma$")
    ax.set_ylabel(r"$\beta_{\rm nl, best}$ (S₃ match)")
    ax.set_title(fr"Nocut low-σ at N_CAS={N_CAS}: refined β vs ln σ")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig("docs/beta_vs_logsigma_N64.png", dpi=130)
    print(f"Saved docs/beta_vs_logsigma_N64.png")

    with open("docs/beta_nocut_N64.json", "w") as f:
        json.dump(dict(N_CAS=N_CAS, SEEDS=SEEDS,
                       intercept=b0, slope=b_ln, rms=rms,
                       se_slope=se_slope, se_intercept=se_intercept,
                       rows=rows), f, indent=2)
    print(f"Saved docs/beta_nocut_N64.json")
    print(f"Total wall time: {time.time()-t_all:.0f}s")


if __name__ == "__main__":
    main()
