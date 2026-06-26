#!/usr/bin/env python
"""Render APOD-style dark-matter density wedges from a DiscoDJ lightcone.

The colour transfer function follows the requested APOD reference:
low projected density is warm white, the filamentary web falls through grey
to black, and only the densest knots turn gold.

By default this script reuses ``$APOD_LIGHTCONE_H5`` when set, otherwise it
creates/reuses a clean single-box lightcone in ``/tmp``. Run from the repo root:

    .venv/bin/python docs/render_lightcone_apod_density.py

Useful overrides:

    APOD_LIGHTCONE_H5=/path/to/lc.h5 APOD_GENERATE=0 ...
    APOD_RES=192 APOD_NRESAMPLE=2 APOD_BOX=4000 ...
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from discodj import DiscoDJ
import discodj.core.io  # registers hdf5plugin filters for compressed catalogues


HERE = Path(__file__).resolve().parent
FIGDIR = HERE / "figures"
FIGDIR.mkdir(exist_ok=True)

RES = int(os.environ.get("APOD_RES", "256"))
BOXSIZE = float(os.environ.get("APOD_BOX", "4000.0"))  # Mpc/h
A_FAR = float(os.environ.get("APOD_A_FAR", "0.55"))
A_NEAR = float(os.environ.get("APOD_A_NEAR", "1.0"))
N_SHELLS = int(os.environ.get("APOD_N_SHELLS", "32"))
N_RESAMPLE = int(os.environ.get("APOD_NRESAMPLE", "2"))
SEED = int(os.environ.get("APOD_SEED", "42"))

WEDGE_CENTER_DEG = float(os.environ.get("APOD_WEDGE_RA", "38.0"))
WEDGE_HALF_DEG = float(os.environ.get("APOD_WEDGE_HALF_DEG", "24.0"))
SLICE_HALF_DEC_DEG = float(os.environ.get("APOD_SLICE_HALF_DEC_DEG", "3.0"))
PATCH_HALF_RA_DEG = float(os.environ.get("APOD_PATCH_HALF_RA_DEG", "9.0"))
PATCH_HALF_DEC_DEG = float(os.environ.get("APOD_PATCH_HALF_DEC_DEG", "5.0"))
SMOOTH_PIX = float(os.environ.get("APOD_SMOOTH_PIX", "3.5"))
JITTER_CELLS = float(os.environ.get("APOD_JITTER_CELLS", "0.35"))

plt.rcParams.update({
    "figure.facecolor": "#050505",
    "savefig.facecolor": "#050505",
    "axes.facecolor": "#050505",
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "text.color": "#d8d8d8",
    "axes.labelcolor": "#c9c9c9",
    "xtick.color": "#9b9b9b",
    "ytick.color": "#9b9b9b",
    "axes.edgecolor": "#2b2b2b",
})


def apod_density_cmap() -> LinearSegmentedColormap:
    """White -> black -> gold transfer function for projected density."""
    colors = [
        (0.00, "#f4f1e8"),
        (0.20, "#b6b1a7"),
        (0.46, "#343434"),
        (0.70, "#010101"),
        (0.83, "#2b1803"),
        (0.93, "#b37425"),
        (1.00, "#fff0a8"),
    ]
    cmap = LinearSegmentedColormap.from_list("apod_dm_density", colors, N=512)
    cmap.set_bad("#050505")
    return cmap


CMAP = apod_density_cmap()
_CMAP_LUT = (CMAP(np.linspace(0.0, 1.0, 4096))[:, :3] * 255.0).astype(np.uint8)


def lightcone_path() -> Path:
    env_path = os.environ.get("APOD_LIGHTCONE_H5")
    if env_path:
        return Path(env_path).expanduser().resolve()
    name = f"discodj_apod_lc_R{RES}_L{BOXSIZE:.0f}_a{A_FAR:.2f}_nr{N_RESAMPLE}.h5"
    return Path(tempfile.gettempdir()) / name


def maybe_generate_lightcone(path: Path) -> None:
    if path.exists() and os.environ.get("APOD_REUSE", "1") != "0":
        print(f"Reusing lightcone: {path}", flush=True)
        return
    if os.environ.get("APOD_GENERATE", "1") == "0":
        raise FileNotFoundError(f"{path} does not exist and APOD_GENERATE=0")

    print(f"Generating {RES}^3 2LPT lightcone at {path}", flush=True)
    print(f"  L={BOXSIZE:g} Mpc/h, a={A_FAR:g}..{A_NEAR:g}, n_resample={N_RESAMPLE}", flush=True)
    dj = (DiscoDJ(dim=3, res=RES, boxsize=BOXSIZE, cosmo="Planck18EEBAOSN")
          .with_timetables()
          .with_linear_ps()
          .with_ics(seed=SEED)
          .with_lpt(n_order=2))
    observer = np.array([BOXSIZE / 2.0] * 3)
    summary = dj.evaluate_lpt_lightcone_to_hdf5(
        str(path),
        a_far=A_FAR,
        a_near=A_NEAR,
        n_shells=N_SHELLS,
        observer=observer,
        n_part_chunks=int(os.environ.get("APOD_N_PART_CHUNKS", "4")),
        n_newton_iters=1,
        v_mode="radial",
        n_resample=N_RESAMPLE,
        keep_particle_idx=False,
        compression="zstd",
        verbose=True,
    )
    print(f"  wrote {summary['n_particles']:,} crossings; replicas={summary['n_replicas']}", flush=True)


def _angle_delta_deg(phi_deg: np.ndarray, center_deg: float) -> np.ndarray:
    return ((phi_deg - center_deg + 180.0) % 360.0) - 180.0


def _display_image(hist: np.ndarray, empty_floor: float = 0.35) -> tuple[np.ndarray, float, float]:
    pos = hist[hist > 0]
    if pos.size == 0:
        raise ValueError("empty density histogram")
    floor = max(np.percentile(pos, 0.2) * empty_floor, np.finfo(float).tiny)
    img = np.log10(np.maximum(hist, floor))
    log_pos = np.log10(np.maximum(pos, floor))
    vmin_pct = float(os.environ.get("APOD_LOG_VMIN_PCT", "10.0"))
    vmax_pct = float(os.environ.get("APOD_LOG_VMAX_PCT", "99.85"))
    vmin = np.percentile(log_pos, vmin_pct)
    vmax = np.percentile(log_pos, vmax_pct)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.min(log_pos))
        vmax = float(np.max(log_pos))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return img, float(vmin), float(vmax)


def _log_rgb_image(hist: np.ndarray, mask: np.ndarray | None = None,
                   smooth_pix: float | None = None) -> np.ndarray:
    """Return an opaque RGB image from a smoothed log-density map."""
    sm = gaussian_filter(hist, sigma=SMOOTH_PIX if smooth_pix is None else smooth_pix,
                         mode="nearest")
    if mask is not None:
        valid = mask & np.isfinite(sm)
    else:
        valid = np.isfinite(sm)
    pos = sm[valid & (sm > 0)]
    if pos.size == 0:
        raise ValueError("empty density histogram")

    floor = max(np.percentile(pos, 0.2) * 0.35, np.finfo(float).tiny)
    log_img = np.log10(np.maximum(sm, floor))
    log_pos = np.log10(np.maximum(pos, floor))
    vmin_pct = float(os.environ.get("APOD_LOG_VMIN_PCT", "10.0"))
    vmax_pct = float(os.environ.get("APOD_LOG_VMAX_PCT", "99.85"))
    vmin = np.percentile(log_pos, vmin_pct)
    vmax = np.percentile(log_pos, vmax_pct)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(log_pos.min()), float(log_pos.max())
    if vmax <= vmin:
        vmax = vmin + 1.0

    t = np.clip((log_img - vmin) / (vmax - vmin), 0.0, 1.0)
    idx = np.minimum((_CMAP_LUT.shape[0] - 1), np.floor(t * (_CMAP_LUT.shape[0] - 1)).astype(np.int32))
    rgb = _CMAP_LUT[idx]
    if mask is not None:
        rgb = np.where(mask[..., None], rgb, np.array([1, 1, 1], dtype=np.uint8))
    return rgb.astype(np.uint8)


def _add_stamp(rgb: np.ndarray, text: str) -> np.ndarray:
    im = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(im)
    x = max(16, int(im.width * 0.018))
    y = im.height - max(18, int(im.height * 0.027)) - 14
    # A dim but readable label; pure RGB, no alpha channel.
    draw.text((x, y), text, fill=(160, 160, 160))
    return np.asarray(im)


def _save_rgb(rgb: np.ndarray, out: Path) -> None:
    Image.fromarray(rgb, mode="RGB").save(out, optimize=True)
    print(f"wrote {out}", flush=True)


def stream_histograms(path: Path) -> dict[str, np.ndarray | float]:
    """Stream the catalogue once and accumulate the requested visual products."""
    with h5py.File(path, "r") as f:
        h = f["Header"].attrs
        observer = np.asarray(h["Observer"], dtype=np.float64)
        boxsize = float(h["BoxSize"])
        num_resample = int(h.get("NumResample", N_RESAMPLE))
        num_per_replica = int(h.get("NumPart_PerReplica", RES ** 3 * max(num_resample, 1) ** 3))
        base_res = int(round((num_per_replica / max(num_resample, 1) ** 3) ** (1.0 / 3.0)))
        cell_size = boxsize / max(base_res, 1)
        jitter_halfwidth = JITTER_CELLS * cell_size
        rng = np.random.default_rng(int(os.environ.get("APOD_JITTER_SEED", "90210")))
        g = f["PartType1"]
        n_rows = g["Coordinates"].shape[0]
        if n_rows == 0:
            raise ValueError(f"{path} has no catalogue rows")

        # For the clean gallery setup chi(a_far) is just under L/2; staying
        # slightly inside L/2 avoids a blank high-radius strip in cone plots.
        r_max = boxsize * 0.4975
        r_min = max(120.0, boxsize * 0.035)
        y_lim = r_max * np.sin(np.radians(WEDGE_HALF_DEG)) * 1.06

        wedge_bins = (1150, 680)
        cone_bins = (1050, 620)
        patch_bins = (720, 420)
        triptych_bins = (460, 280)

        H_wedge = np.zeros((wedge_bins[1], wedge_bins[0]), dtype=np.float64)
        H_cone = np.zeros((cone_bins[1], cone_bins[0]), dtype=np.float64)
        H_patch = np.zeros((patch_bins[1], patch_bins[0]), dtype=np.float64)
        H_slabs = [np.zeros((triptych_bins[1], triptych_bins[0]), dtype=np.float64)
                   for _ in range(3)]

        x_range = [r_min, r_max]
        y_range = [-y_lim, y_lim]
        theta_range = [-WEDGE_HALF_DEG, WEDGE_HALF_DEG]
        dec_range = [-PATCH_HALF_DEC_DEG, PATCH_HALF_DEC_DEG]
        slab_edges = np.array([0.18, 0.32, 0.46, 0.62])

        batch = int(os.environ.get("APOD_BATCH_ROWS", str(1 << 21)))
        for s in range(0, n_rows, batch):
            e = min(s + batch, n_rows)
            x = np.asarray(g["Coordinates"][s:e], dtype=np.float64)
            a = np.asarray(g["ScaleFactor"][s:e], dtype=np.float64)
            rel = x - observer[None, :]
            if jitter_halfwidth > 0.0:
                rel += rng.uniform(-jitter_halfwidth, jitter_halfwidth, size=rel.shape)
            chi = np.linalg.norm(rel, axis=1)
            phi = np.degrees(np.arctan2(rel[:, 1], rel[:, 0])) % 360.0
            dec = np.degrees(np.arcsin(np.clip(rel[:, 2] / np.maximum(chi, 1e-30), -1, 1)))
            theta = _angle_delta_deg(phi, WEDGE_CENTER_DEG)
            z = 1.0 / a - 1.0

            in_ang = np.abs(theta) <= WEDGE_HALF_DEG
            in_r = (chi >= r_min) & (chi <= r_max)
            in_slice = in_ang & in_r & (np.abs(dec) <= SLICE_HALF_DEC_DEG)
            if np.any(in_slice):
                th = np.radians(theta[in_slice])
                xx = chi[in_slice] * np.cos(th)
                yy = chi[in_slice] * np.sin(th)
                # Angular slices thicken with radius; 1/chi keeps radial contrast
                # dominated by structure rather than wedge geometry.
                w = 1.0 / np.clip(chi[in_slice], 1.0, None)
                H, _, _ = np.histogram2d(yy, xx, bins=[wedge_bins[1], wedge_bins[0]],
                                         range=[y_range, x_range], weights=w)
                H_wedge += H
                Hc, _, _ = np.histogram2d(chi[in_slice], theta[in_slice],
                                          bins=[cone_bins[1], cone_bins[0]],
                                          range=[[r_min, r_max], theta_range],
                                          weights=w)
                H_cone += Hc

            in_patch = (np.abs(theta) <= PATCH_HALF_RA_DEG) & (np.abs(dec) <= PATCH_HALF_DEC_DEG)
            in_patch &= (z >= slab_edges[0]) & (z <= slab_edges[-1])
            if np.any(in_patch):
                H, _, _ = np.histogram2d(dec[in_patch], theta[in_patch],
                                         bins=[patch_bins[1], patch_bins[0]],
                                         range=[dec_range, [-PATCH_HALF_RA_DEG, PATCH_HALF_RA_DEG]])
                H_patch += H
                for i in range(3):
                    m = in_patch & (z >= slab_edges[i]) & (z < slab_edges[i + 1])
                    if np.any(m):
                        Hs, _, _ = np.histogram2d(
                            dec[m], theta[m],
                            bins=[triptych_bins[1], triptych_bins[0]],
                            range=[dec_range, [-PATCH_HALF_RA_DEG, PATCH_HALF_RA_DEG]],
                        )
                        H_slabs[i] += Hs

    return {
        "wedge": H_wedge,
        "cone": H_cone,
        "patch": H_patch,
        "slabs": H_slabs,
        "r_min": r_min,
        "r_max": r_max,
        "y_lim": y_lim,
        "slab_edges": slab_edges,
        "n_rows": n_rows,
    }


def save_wedge(hist: np.ndarray, r_min: float, r_max: float, y_lim: float, out: Path) -> None:
    ny, nx = hist.shape
    x = np.linspace(r_min, r_max, nx)
    y = np.linspace(-y_lim, y_lim, ny)
    xx, yy = np.meshgrid(x, y)
    theta = np.degrees(np.arctan2(yy, np.maximum(xx, 1e-30)))
    rr = np.sqrt(xx * xx + yy * yy)
    mask = (np.abs(theta) <= WEDGE_HALF_DEG) & (rr >= r_min) & (rr <= r_max)
    rgb = _log_rgb_image(hist, mask=mask)
    rgb = _add_stamp(rgb, f"{2 * WEDGE_HALF_DEG:.0f} deg x {2 * SLICE_HALF_DEC_DEG:.1f} deg lightcone wedge")
    _save_rgb(rgb, out)


def save_cone(hist: np.ndarray, r_min: float, r_max: float, out: Path) -> None:
    rgb = _log_rgb_image(hist)
    rgb = _add_stamp(rgb, "survey cone plot: projected dark-matter density")
    _save_rgb(rgb, out)


def save_patch(hist: np.ndarray, out: Path) -> None:
    rgb = _log_rgb_image(hist)
    rgb = _add_stamp(rgb, f"{2 * PATCH_HALF_RA_DEG:.0f} deg x {2 * PATCH_HALF_DEC_DEG:.0f} deg sky projection")
    _save_rgb(rgb, out)


def save_triptych(slabs: list[np.ndarray], slab_edges: np.ndarray, out: Path) -> None:
    panels = []
    gap = 8
    for i, H in enumerate(slabs):
        rgb = _log_rgb_image(H)
        rgb = _add_stamp(rgb, f"z = {slab_edges[i]:.2f}-{slab_edges[i + 1]:.2f}")
        panels.append(rgb)
    h = max(p.shape[0] for p in panels)
    w = sum(p.shape[1] for p in panels) + gap * (len(panels) - 1)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    x0 = 0
    for p in panels:
        canvas[:p.shape[0], x0:x0 + p.shape[1]] = p
        x0 += p.shape[1] + gap
    _save_rgb(canvas, out)


def main() -> None:
    path = lightcone_path()
    maybe_generate_lightcone(path)
    print(f"Streaming density histograms from {path}", flush=True)
    h = stream_histograms(path)

    save_wedge(h["wedge"], h["r_min"], h["r_max"], h["y_lim"],
               FIGDIR / "lightcone_dm_apod_wedge_slice.png")
    save_cone(h["cone"], h["r_min"], h["r_max"],
              FIGDIR / "lightcone_dm_apod_survey_cone.png")
    save_patch(h["patch"], FIGDIR / "lightcone_dm_apod_sky_projection.png")
    save_triptych(h["slabs"], h["slab_edges"],
                  FIGDIR / "lightcone_dm_apod_tomographic_slices.png")
    print(f"Rendered from {h['n_rows']:,} catalogue rows", flush=True)


if __name__ == "__main__":
    main()
