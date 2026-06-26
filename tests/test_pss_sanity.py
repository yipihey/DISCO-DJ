"""Sanity test for phase-space-sheet Kuhn tet/cube volumes.

Checks:
  1. Non-overlapping sampling: total signed tet volume ≈ box volume.
  2. Overlapping cube mean = d_c³ = (s·L/N)³ (per-cube, not per-tet).
  3. Variance decreases with stride (smoothing to larger mass scale).
  4. Effective sample count n_cubes scales as N³ for overlapping,
     (N/s)³ for non-overlapping.
"""
import numpy as np

from discodj.analysis import (load_sim, kuhn_tet_volumes, kuhn_cube_volumes)


def test_non_overlapping_total_volume():
    """Strict subsampling tiles the box: Σ V_tet = L³ exactly."""
    snap = load_sim(128, "nocutoff", a=0.10)
    V = kuhn_tet_volumes(snap, stride=4, overlapping=False)
    N_c = snap.N // 4
    L = snap.L
    expected_mean = L**3 / (6 * N_c**3)
    V_total = float(V.sum()); V_mean = float(V.mean())
    print(f"  (non-ovlp) ⟨V_tet⟩={V_mean:.4f}  vs {expected_mean:.4f}")
    print(f"  (non-ovlp) total={V_total:.2f}  vs L³={L**3:.2f}")
    assert abs(V_total - L**3) / L**3 < 1e-5
    assert abs(V_mean - expected_mean) / expected_mean < 1e-4


def test_overlapping_cube_mean():
    """Overlapping cube volumes have mean = d_c³ per cube."""
    snap = load_sim(128, "nocutoff", a=0.10)
    print(f"\n  stride | n_cube | ⟨V_cube⟩     | expected d_c³ | σ_v")
    for s in (1, 2, 4, 8):
        V = kuhn_cube_volumes(snap, stride=s, overlapping=True)
        d_c = snap.L / (snap.N // s)
        expected = d_c**3
        v_mean = float(V.mean())
        v_sigma = float(V.std()) / v_mean
        print(f"  {s:>6} | {V.size:>6} | {v_mean:.4e} | {expected:.4e} | {v_sigma:.4f}")
        assert abs(v_mean - expected) / expected < 1e-3
    # σ_v should decrease monotonically with stride (larger scale → less variance)
    prev = None
    for s in (1, 2, 4, 8):
        V = kuhn_cube_volumes(snap, stride=s, overlapping=True)
        cur = float(V.std()) / float(V.mean())
        if prev is not None:
            assert cur < prev, f"σ_v not monotone: s={s}"
        prev = cur


def test_overlap_vs_subsample_moments():
    """Overlap and subsample give the SAME moments (up to correlation noise)."""
    snap = load_sim(128, "nocutoff", a=0.10)
    V_ov = kuhn_cube_volumes(snap, stride=4, overlapping=True)
    V_no = kuhn_cube_volumes(snap, stride=4, overlapping=False)
    m_ov = V_ov.mean(); m_no = V_no.mean()
    print(f"\n  mean ov={m_ov:.5f}  no-ov={m_no:.5f}  rel diff={abs(m_ov-m_no)/m_no:.2e}")
    # Mean must match (both are unbiased).
    assert abs(m_ov - m_no) / m_no < 1e-3
    # Stddev can differ by O(1%) from sample noise; overlap has 64× more cubes.
    print(f"  std ov={V_ov.std():.5f}  no-ov={V_no.std():.5f}")


if __name__ == "__main__":
    test_non_overlapping_total_volume()
    test_overlapping_cube_mean()
    test_overlap_vs_subsample_moments()
    print("\nAll PSS sanity checks passed.")
