"""Measure what fraction of a night's binned frames has data at each sky position, after derotation.

usage: python analysis/procyon4_2026-11/sky_coverage.py REDUCED_PY PREFIX [--half 240] [--step 2]
       [--radii 2.5 3.0 3.5 4.0] [--n-pa 20] [--pxscale 0.0179]

A sky position at separation r and position angle θ falls, in frame i, on the detector position
turned clockwise by that frame's parallactic angle, since KLIP-TPE derotates a frame
counterclockwise by its angle (``rotate_ccw(frame, angle)``). The frame has data there if that
pixel of the centered (2 half)² cut is finite. This counts, for both chops' binned frames
together, the fraction of frames with data at each position, which is what klip-tpe bf99474's
``--min-coverage`` compares with: the fraction that must hold data where a source or a noise
aperture goes. Positions are sampled at the nearest pixel.

Parameters
----------
REDUCED_PY : str
    The night's ``reduced_py`` folder.
PREFIX : str
    File prefix of the side, e.g. ``Procyon_20211220_left``.
--half : int, default 240
    Half-width of the cut, pixels.
--step : int, default 2
    Use every step-th binned frame (1 uses them all).
--radii : float list, default 2.5 3.0 3.5 4.0
    Separations, arcsec.
--n-pa : int, default 20
    Number of evenly spaced position angles counted per separation, from 0 (the IDL curve's 20).
--pxscale : float, default 0.0179
    Plate scale, arcsec per pixel.

Returns
-------
stdout : text
    Per separation: the smallest, median and largest coverage over 360 position angles at 1°,
    the fraction of the circle with coverage at least 0.5 and above 0, and how many of the n_pa
    position angles reach 0.5 and have any data.
"""
import argparse
import os
import sys

import numpy as np
from astropy.io import fits


def frame_masks(cube_path: str, half: int, step: int) -> np.ndarray:
    """Read every step-th frame's finite-pixel mask within a centered cut.

    Parameters
    ----------
    cube_path : str
        Binned cube, shape (n, ny, nx), with the star at the array center.
    half : int
        Half-width of the cut, pixels.
    step : int
        Frame stride.

    Returns
    -------
    masks : ndarray of bool, shape (m, 2 half, 2 half)
        True where the frame is finite, for the m = ceil(n / step) frames read.
    """
    with fits.open(cube_path, memmap=True) as hd:
        d = hd[0].data
        cy, cx = d.shape[1] // 2, d.shape[2] // 2
        return np.array([np.isfinite(np.asarray(d[k, cy - half:cy + half, cx - half:cx + half]))
                         for k in range(0, d.shape[0], step)])


def coverage(masks: np.ndarray, angles: np.ndarray, r_px: float, theta_deg: np.ndarray) -> np.ndarray:
    """Fraction of frames with data at sky positions on one circle.

    Parameters
    ----------
    masks : ndarray of bool, shape (m, 2 half, 2 half)
        Finite-pixel masks of the m frames.
    angles : ndarray, shape (m,)
        Parallactic angle of each frame, degrees (KLIP-TPE turns frame i counterclockwise by it).
    r_px : float
        Separation, pixels.
    theta_deg : ndarray, shape (p,)
        Position angles, degrees, measured from +y toward -x (north up, east left).

    Returns
    -------
    frac : ndarray, shape (p,)
        Fraction of the m frames finite at each position.
    """
    m, n, _ = masks.shape
    c = (n - 1) / 2.0
    t = np.radians(theta_deg)
    sx, sy = -r_px * np.sin(t), r_px * np.cos(t)
    hits = np.zeros(theta_deg.size)
    for k in range(m):
        a = np.radians(angles[k])
        dx = np.cos(a) * sx + np.sin(a) * sy          # turned clockwise by the frame's angle
        dy = -np.sin(a) * sx + np.cos(a) * sy
        ix, iy = np.rint(c + dx).astype(int), np.rint(c + dy).astype(int)
        inside = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
        ok = np.zeros(theta_deg.size, bool)
        ok[inside] = masks[k][iy[inside], ix[inside]]
        hits += ok
    return hits / m


def main(argv=None) -> int:
    """Print the coverage per separation for both chops together.

    Parameters
    ----------
    argv : list of str or None, default None
        Command-line arguments; None reads ``sys.argv``.

    Returns
    -------
    status : int
        0.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("reduced_py")
    ap.add_argument("prefix")
    ap.add_argument("--half", type=int, default=240)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--radii", type=float, nargs="+", default=[2.5, 3.0, 3.5, 4.0])
    ap.add_argument("--n-pa", type=int, default=20)
    ap.add_argument("--pxscale", type=float, default=0.0179)
    a = ap.parse_args(argv)
    masks, angs = [], []
    for cn in (1, 2):
        m = frame_masks(os.path.join(a.reduced_py, f"{a.prefix}_cube{cn}_align_cen_bin.fits"), a.half, a.step)
        ang = np.asarray(fits.getdata(os.path.join(a.reduced_py, f"{a.prefix}_angles{cn}_bin.fits")), float).ravel()
        masks.append(m)
        angs.append(ang[::a.step][:m.shape[0]])
    M, A = np.concatenate(masks), np.concatenate(angs)
    print(f"{M.shape[0]} frames (every {a.step}), cut {2 * a.half} px, angles {A.min():.1f} to {A.max():.1f} deg")
    fine = np.arange(360.0)
    coarse = np.arange(a.n_pa) * 360.0 / a.n_pa
    for r in a.radii:
        rp = r / a.pxscale
        if rp > a.half - 1:
            print(f"{r:4.2f}\": outside the cut")
            continue
        f, g = coverage(M, A, rp, fine), coverage(M, A, rp, coarse)
        print(f"{r:4.2f}\" ({rp:5.1f} px): coverage min {f.min():.2f}, median {np.median(f):.2f}, max {f.max():.2f}; "
              f"circle with >= 0.5: {np.mean(f >= 0.5):.0%}, with any data: {np.mean(f > 0):.0%}; "
              f"of {a.n_pa} PAs: {int(np.sum(g >= 0.5))} reach 0.5, {int(np.sum(g > 0))} have data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
