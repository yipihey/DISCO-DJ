"""Multi-scale (σ, γ, S₃^V) extraction from existing PM snapshots.

At each PM snapshot the Jacobian field ``J`` is defined on the full
mesh. Tophat-smooth ``J`` at a range of Lagrangian radii ``R`` and
record:
  σ²(R)   = variance of the smoothed J
  S₃^V(R) = κ₃(R) / σ²(R)²            [volume-weighted skewness]
  γ(R)    = d ln σ²_lin(R) / d ln R   [from the linear P(k)]

Each snapshot contributes N_R data points instead of one. The γ-
trajectory within a snapshot (R-sweep at fixed growth time) is roughly
orthogonal to the γ-trajectory across snapshots (growth at fixed R),
so the union fills the (σ, γ) plane and breaks the degeneracy that
the single-scale fit suffered from.

Pure post-processing of existing `.npz` sim files — no new PM runs.
"""

from __future__ import annotations

import json, time

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax, jax.numpy as jnp

from discodj.analysis import load_sim, deformation_gradient_jacobian
from discodj.analysis.spectral_slope import (
    spectral_gamma, smoothed_sigma2, reconstruct_pk,
)


BOXSIZE_DEFAULT = 250.0
R_LIST = np.array([4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0])  # Mpc/h


def tophat_Wk(kR: np.ndarray) -> np.ndarray:
    """Top-hat window in k-space: W(kR) = 3(sin x − x cos x)/x³."""
    small = kR < 1e-3
    x = np.where(small, 1e-3, kR)
    W = 3.0 * (np.sin(x) - x * np.cos(x)) / x**3
    return np.where(small, 1.0, W)


def smooth_J_tophat(J: np.ndarray, boxsize: float, R: float) -> np.ndarray:
    """Apply a tophat smoothing of radius R to the 3D J field via FFT."""
    N = J.shape[0]
    J_f = np.fft.rfftn(J.astype(np.float64))
    # k-magnitudes on the rfft grid
    kx = 2 * np.pi * np.fft.fftfreq(N, d=boxsize / N)
    ky = kx
    kz = 2 * np.pi * np.fft.rfftfreq(N, d=boxsize / N)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    kmag = np.sqrt(KX**2 + KY**2 + KZ**2)
    W = tophat_Wk(kmag * R)
    J_s = np.fft.irfftn(J_f * W, s=(N, N, N))
    return J_s.astype(np.float32)


def moments(x: np.ndarray) -> tuple[float, float, float]:
    """Return (mean, variance, κ₃) of a 1-D array."""
    x = np.asarray(x).ravel().astype(np.float64)
    m = float(x.mean())
    d = x - m
    v = float((d**2).mean())
    k3 = float((d**3).mean())
    return m, v, k3


def multi_scale_moments_one_snap(snap, pk_k, pk_Pk,
                                  R_list=R_LIST):
    """For one snapshot, measure (σ²(R), S₃^V(R), γ(R)) at each R."""
    # J field
    _, J = deformation_gradient_jacobian(snap)
    J_np = np.asarray(J)
    out = []
    for R in R_list:
        J_R = smooth_J_tophat(J_np, snap.L, float(R))
        _, v, k3 = moments(J_R)
        if v <= 0:
            continue
        S3_V = k3 / v**2                        # volume-weighted S3
        S3_std = k3 / v**1.5                    # standard S3 for cascade match
        gamma = spectral_gamma(pk_k, pk_Pk, float(R), dln=0.1)
        out.append(dict(R=float(R), sigma2=float(v),
                        sigma2_lin=float(smoothed_sigma2(pk_k, pk_Pk, float(R))),
                        S3_V=float(S3_V), S3_std=float(S3_std),
                        gamma=float(gamma), kappa3=float(k3)))
    return out


def main():
    results = {}
    for N_PM in (256, 512):
        for a_pm in (0.03, 0.05, 0.07, 0.10):
            try:
                snap = load_sim(N_PM, "nocutoff", a=a_pm)
            except FileNotFoundError:
                continue
            if abs(snap.a - a_pm) > 0.005:
                # load_sim snaps to closest; skip when the requested a isn't stored.
                continue
            k_np, Pk_np = reconstruct_pk(snap)
            t0 = time.time()
            rows = multi_scale_moments_one_snap(snap, k_np, Pk_np)
            for r in rows:
                r.update(N_PM=N_PM, a=float(snap.a))
            results[(N_PM, float(snap.a))] = rows
            print(f"N={N_PM} a={snap.a:.2f}: {len(rows)} R-points "
                  f"({time.time()-t0:.0f}s)")
            for r in rows:
                print(f"   R={r['R']:5.1f}  σ²_J={r['sigma2']:.4f}  "
                      f"σ²_lin={r['sigma2_lin']:.4f}  γ={r['gamma']:+.2f}  "
                      f"S₃^V={r['S3_V']:+.2f}   S₃_std={r['S3_std']:+.2f}")

    flat = [r for rows in results.values() for r in rows]
    with open("docs/multiscale_J_moments.json", "w") as f:
        json.dump(flat, f, indent=2)
    print(f"\nSaved docs/multiscale_J_moments.json with {len(flat)} rows.")

    # Diagnostic plot: (σ, γ) coverage.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, xk, xlabel in [(axes[0], "gamma", r"$\gamma$"),
                            (axes[1], "S3_V", r"$S_3^V$")]:
        for (N_PM, a), rows in results.items():
            color = "tab:blue" if N_PM == 256 else "tab:red"
            marker = "o"
            ln_s = [0.5 * np.log(r["sigma2"]) for r in rows]
            y = [r[xk] for r in rows]
            ax.plot(ln_s, y, "-", color=color, alpha=0.3)
            ax.plot(ln_s, y, marker, color=color,
                    label=f"N={N_PM} a={a:.2f}" if xk == "gamma" else None,
                    ms=6, alpha=0.8)
        ax.set_xlabel(r"$\ln\,\sigma(R)$ (PM J)")
        ax.set_ylabel(xlabel)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2, loc="best")
    axes[0].set_title("Multi-scale (σ,γ) coverage")
    axes[1].set_title("S₃^V vs ln σ(R)")
    fig.tight_layout()
    fig.savefig("docs/multiscale_coverage.png", dpi=130)
    print("Saved docs/multiscale_coverage.png")


if __name__ == "__main__":
    main()
