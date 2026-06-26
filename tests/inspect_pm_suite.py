"""Quick sanity + summary of the stored PM sim suite.

Loads every sim, reports (N, cutoff, a, σ²_ψ, σ²_δ_smoothed, ⟨J⟩, var(J)).
Good for spot-checking that the suite is self-consistent before any
cascade analysis.

Usage::

    python tests/inspect_pm_suite.py
"""

from __future__ import annotations

import os
import numpy as np

from discodj.analysis.sim_store import (
    load_sim, list_sims, displacement_field, deformation_gradient_jacobian,
)


def main():
    root = "sims"
    tags = list_sims(root)
    if not tags:
        print(f"No sims found in {root}/. Run tests/gen_pm_suite.py first.")
        return

    print(f"Found {len(tags)} sim(s): {tags}\n")
    hdr = (f"{'sim':>14} | {'a':>5} | {'<J>':>7} | {'σ²_J':>7} | "
           f"{'S₃(J)':>7} | {'Jmin':>8} | {'J<0 %':>7} | {'J 5%':>7} | {'J 95%':>7}")
    print(hdr); print("-" * len(hdr))

    for tag in tags:
        N = int(tag.split("_")[0].lstrip("N"))
        cutoff_mode = tag.split("_", 1)[1]
        snaps = load_sim(N, cutoff_mode, root=root)
        for snap in snaps:
            _, J = deformation_gradient_jacobian(snap)
            J_flat = J.ravel()
            mean_J = float(np.mean(J_flat))
            var_J = float(np.var(J_flat))
            # Signed-J skewness (finite at all σ²; not sensitive to 1/J tails).
            S3_J = float(np.mean((J_flat - mean_J) ** 3) / max(var_J, 1e-30) ** 1.5)
            Jmin = float(np.min(J_flat))
            frac_neg = 100.0 * float(np.mean(J_flat < 0))
            J_p5 = float(np.percentile(J_flat, 5))
            J_p95 = float(np.percentile(J_flat, 95))
            print(f"{tag:>14} | {snap.a:5.2f} | {mean_J:7.4f} | {var_J:7.4f} | "
                  f"{S3_J:+7.3f} | {Jmin:+8.2f} | {frac_neg:6.3f}% | "
                  f"{J_p5:+7.3f} | {J_p95:+7.3f}")


if __name__ == "__main__":
    main()
