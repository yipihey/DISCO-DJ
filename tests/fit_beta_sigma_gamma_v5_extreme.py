"""Extend the v5 β(σ, γ) joint fit with extreme-cosmology snapshots
to break the σ-γ degeneracy.

Planck cosmo alone has σ ↔ γ strongly correlated along the growth
trajectory, so the 2-parameter fit is poorly conditioned. The
extreme-cosmo suite (n_s = 1.25, σ_8 = 1.0, Ω_c = 0.45) gives
different γ(σ) — points slide off the Planck γ-σ line, which breaks
the degeneracy.

Runs a cascade β-scan on each extreme-cosmo snapshot, combines with
the existing Planck N=256 + N=512 v5 β_best values, and fits the
2-parameter joint law.
"""

import json, time

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax, jax.numpy as jnp

from discodj import DiscoDJ
from discodj.analysis import load_sim, deformation_gradient_jacobian
from discodj.analysis.sim_store import SimSnapshot
from discodj.analysis.spectral_slope import (
    R_from_sigma2, spectral_gamma, smoothed_sigma2)
from discodj.cascade import fem_cascade_v5


N_CAS, N_STEPS, BOXSIZE = 64, 10, 250.0
SEEDS = 5
EXTREME_COSMO = dict(
    Omega_c=0.45, Omega_b=0.05, Omega_k=0.0, h=0.67,
    n_s=1.25, sigma8=1.0, w0=-1.0, wa=0.0,
)


def Jmom(J):
    a = np.asarray(J).ravel()
    m = a.mean(); d = a - m
    v = float(np.mean(d * d))
    s3 = float(np.mean(d**3) / max(v, 1e-30)**1.5)
    return float(m), v, s3


def load_extreme(cm, a_pm):
    path = f"sims/N256_{cm}_extreme.npz"
    data = np.load(path, allow_pickle=True)
    meta = json.loads(str(data["meta"]))
    xs, vs, as_ = data["x"], data["v"], data["a"]
    idx = int(np.argmin(np.abs(as_ - a_pm)))
    return SimSnapshot(x=xs[idx], v=vs[idx], a=float(as_[idx]),
                       N=meta["N"], L=meta["L"], meta=meta)


def extreme_pk(cutoff_mode):
    """Build the extreme-cosmo P(k), with Gaussian cutoff applied if requested."""
    dj = (DiscoDJ(dim=3, res=32, boxsize=BOXSIZE, cosmo=EXTREME_COSMO)
          .with_timetables().with_linear_ps())
    k = np.asarray(dj._pk_table["k"])
    Pk = np.asarray(dj._pk_table["Pk"])
    if cutoff_mode == "cutoff":
        k_cut = 0.5 * np.pi * 64 / BOXSIZE
        Pk = Pk * np.exp(-(k / k_cut)**2)
    return k, Pk


def scan(sigma2, pk_k, pk_Pk, betas):
    vs, S3s = np.empty_like(betas), np.empty_like(betas)
    for i, b in enumerate(betas):
        vi, si = [], []
        for s in range(SEEDS):
            r = fem_cascade_v5(sigma2=sigma2, n_steps=N_STEPS, N=N_CAS,
                               boxsize=BOXSIZE, key=jax.random.PRNGKey(9500 + s),
                               g_T=0.0, alpha_nl=0.0, beta_nl=float(b), gamma_nl=0.0,
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
    b = betas[imin - 1:imin + 2]; L = loss[imin - 1:imin + 2]
    denom = (L[0] - 2 * L[1] + L[2])
    if abs(denom) < 1e-12:
        return float(betas[imin])
    offset = 0.5 * (L[0] - L[2]) / denom * (b[1] - b[0])
    return float(b[1] + offset)


def gamma_for_extreme(sigma2, k_np, Pk_np):
    """γ at the Lagrangian R where σ²_lin(R) matches PM σ² — using the
    extreme P(k)."""
    R = R_from_sigma2(k_np, Pk_np, sigma2)
    return float(spectral_gamma(k_np, Pk_np, R, dln=0.1))


def scan_extreme_snaps():
    """Run v5 β-scan on each extreme-cosmo snapshot with σ² ≤ 0.3."""
    out = []
    for cm in ("cutoff", "nocutoff"):
        k_np, Pk_np = extreme_pk(cm)
        pk_k, pk_Pk = jnp.asarray(k_np), jnp.asarray(Pk_np)
        for a_pm in (0.1, 0.3):
            snap = load_extreme(cm, a_pm)
            G_pm, J_pm = deformation_gradient_jacobian(snap)
            s2 = float(jnp.var(jnp.einsum("...ii->...", G_pm)))
            if s2 > 0.3:
                print(f"  [skip] extreme {cm} a={a_pm} σ²={s2:.3f} > 0.3")
                continue
            _, v_pm, S3_pm = Jmom(J_pm)
            gamma = gamma_for_extreme(s2, k_np, Pk_np)
            # Widen β grid — extreme γ may push β_best outside the Planck range.
            betas = np.arange(-0.40, 0.11, 0.01)
            t0 = time.time()
            vs_cas, S3s_cas = scan(s2, pk_k, pk_Pk, betas)
            b_best = refine_parabola(betas, (S3s_cas - S3_pm)**2)
            print(f"  extreme {cm:>8} a={a_pm:.2f}  σ²={s2:.4f}  γ={gamma:+.3f}  "
                  f"S₃_PM={S3_pm:+.3f}  β_best={b_best:+.4f}  ({time.time()-t0:.0f}s)")
            out.append(dict(N_PM=256, suite=f"extreme_{cm}", a=float(snap.a),
                            sigma2=s2, gamma=gamma, beta_best=b_best))
    return out


def fit_2d(rows):
    ln_sig = np.array([0.5 * np.log(r["sigma2"]) for r in rows])
    gamma = np.array([r["gamma"] for r in rows])
    y = np.array([r["beta_best"] for r in rows])
    A = np.column_stack([np.ones_like(ln_sig), ln_sig, gamma])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ c
    rms = float(np.sqrt(np.mean(resid**2)))
    cov = np.linalg.inv(A.T @ A) * rms**2
    se = np.sqrt(np.diag(cov))
    return dict(b0=float(c[0]), b_sigma=float(c[1]), b_gamma=float(c[2]),
                se_b0=float(se[0]), se_sigma=float(se[1]), se_gamma=float(se[2]),
                rms=rms, resid=resid.tolist())


def fit_1d(rows):
    ln_sig = np.array([0.5 * np.log(r["sigma2"]) for r in rows])
    y = np.array([r["beta_best"] for r in rows])
    A = np.column_stack([np.ones_like(ln_sig), ln_sig])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ c
    rms = float(np.sqrt(np.mean(resid**2)))
    return float(c[0]), float(c[1]), rms


def main():
    # Planck rows from prior fit file
    d = json.load(open("docs/beta_sigma_gamma_v5.json"))
    planck_rows = d["data"]
    print(f"Loaded {len(planck_rows)} Planck rows.\n")

    # Fresh: extreme-cosmo scans
    print("=== Scanning v5 β_best on extreme-cosmo snapshots ===")
    extreme_rows = scan_extreme_snaps()

    all_rows = planck_rows + extreme_rows
    for r in all_rows:
        r.setdefault("suite", f"N{r['N_PM']}_planck")

    print(f"\nCombined: {len(all_rows)} points\n")

    # 1D fit on combined
    b0, bs, rms1 = fit_1d(all_rows)
    print(f"[1-param, all]  β = {b0:+.4f} + ({bs:+.4f})·ln σ   rms = {rms1:.4f}")

    # 2D fit on combined
    f = fit_2d(all_rows)
    print(f"[2-param, all]  β = {f['b0']:+.4f}(±{f['se_b0']:.4f}) + "
          f"({f['b_sigma']:+.4f}±{f['se_sigma']:.4f})·ln σ + "
          f"({f['b_gamma']:+.4f}±{f['se_gamma']:.4f})·γ")
    print(f"    rms = {f['rms']:.4f}   β_γ vs 0: "
          f"{f['b_gamma']/max(f['se_gamma'],1e-9):+.2f}σ")
    print("\nPer-snap residuals (2-param, combined):")
    for r, dr in zip(all_rows, f["resid"]):
        print(f"  {r['suite']:>18} a={r['a']:.2f}  σ²={r['sigma2']:.4f}  "
              f"γ={r['gamma']:+.3f}  Δβ = {dr:+.4f}")

    # Plot: colour by γ
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"N256_planck": "tab:blue", "N512_planck": "tab:red",
              "extreme_nocutoff": "tab:green", "extreme_cutoff": "tab:purple"}
    markers = {"N256_planck": "o", "N512_planck": "s",
               "extreme_nocutoff": "^", "extreme_cutoff": "v"}
    for lbl in colors:
        pts = [r for r in all_rows if r["suite"] == lbl]
        if not pts: continue
        ln_s = [0.5 * np.log(r["sigma2"]) for r in pts]
        y = [r["beta_best"] for r in pts]
        ax.plot(ln_s, y, markers[lbl], ms=10, color=colors[lbl], label=lbl)

    xs = np.linspace(-2.5, -0.3, 100)
    ax.plot(xs, b0 + bs * xs, "-.", color="grey", alpha=0.7,
            label=f"1-param: β = {b0:+.3f} + ({bs:+.3f})·ln σ")
    # slices of 2-param at low-γ and high-γ
    g_lo = min(r["gamma"] for r in all_rows)
    g_hi = max(r["gamma"] for r in all_rows)
    ax.plot(xs, f["b0"] + f["b_sigma"] * xs + f["b_gamma"] * g_lo, "--",
            color="black", alpha=0.7,
            label=fr"2-param, γ={g_lo:+.2f}")
    ax.plot(xs, f["b0"] + f["b_sigma"] * xs + f["b_gamma"] * g_hi, ":",
            color="black", alpha=0.7,
            label=fr"2-param, γ={g_hi:+.2f}")
    ax.set_xlabel(r"$\ln \sigma$")
    ax.set_ylabel(r"$\beta_{\rm nl, best}$")
    ax.set_title(r"v5 β_best on Planck (N=256, 512) + extreme cosmo: γ-dependence test")
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig("docs/beta_sigma_gamma_v5_extreme.png", dpi=130)
    print("\nSaved docs/beta_sigma_gamma_v5_extreme.png")

    with open("docs/beta_sigma_gamma_v5_extreme.json", "w") as fo:
        json.dump(dict(fit_1param=dict(b0=b0, b_sigma=bs, rms=rms1),
                       fit_2param=f, rows=all_rows), fo, indent=2)
    print("Saved docs/beta_sigma_gamma_v5_extreme.json")


if __name__ == "__main__":
    main()
