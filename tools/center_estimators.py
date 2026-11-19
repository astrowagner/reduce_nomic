#!/usr/bin/env python3
"""Measure how consistently the frames of binned cubes are registered, with two center estimators.

For each FITS cube given (n frames of 2 radius x 2 radius, the star near (radius - 0.5,
radius - 0.5), such as ``<night>_<side>_cube<n>_align_cen_bin.fits``), reads the central
256 x 256 pixels of every frame and prints:

* the IDL center stage's estimate of every frame (``geometry.idl_center``): median, robust
  scatter (1.4826 x the median absolute deviation) and the share of frames more than 0.5 pixel
  from the median, in x and y, in the cube's pixels;
* the same for a Gaussian fit to the core (``geometry.gauss_center``);
* the robust scatter of their difference, which shows how far the IDL estimate jumps on the
  same images;
* the Pearson correlation of every frame's central 64 x 64 pixels with the cube's median frame
  (median and 5th percentile), a registration measure independent of both estimators;
* the FWHM of the median frame, in pixels;
* with ``--recenter``, the same correlation and FWHM after shifting every frame (``idl.fshift``)
  so that its Gaussian center lands on (radius - 0.5, radius - 0.5): what centering on the core
  would give, frame by frame.

Usage: center_estimators.py [--recenter] CUBE [CUBE ...]
Run with the reduce_nomic package on PYTHONPATH.
"""
import os
import sys
from typing import List, Optional

import numpy as np
from astropy.io import fits

from reduce_nomic import geometry, idl

H = 128         # half size of the central region read, pixels


def robust(v: np.ndarray) -> str:
    """Summarize centers: median, robust scatter and share beyond 0.5 pixel, in x and y.

    Parameters
    ----------
    v : ndarray, shape (n, 2)
        Centers (x, y) in pixels; rows with NaN are left out.

    Returns
    -------
    text : str
        The summary.
    """
    v = v[np.all(np.isfinite(v), axis=1)]
    med = np.median(v, axis=0)
    mad = 1.4826 * np.median(np.abs(v - med), axis=0)
    far = np.mean(np.abs(v - med) > 0.5, axis=0)
    return (f"median ({med[0]:.2f}, {med[1]:.2f}), scatter ({mad[0]:.2f}, {mad[1]:.2f}), "
            f"beyond 0.5 px ({far[0]:.0%}, {far[1]:.0%}) of {len(v)}")


def fwhm(img: np.ndarray) -> float:
    """FWHM of a circular Gaussian plus constant fitted to an image's core.

    Parameters
    ----------
    img : ndarray, shape (2 H, 2 H)
        Image with the star near its center.

    Returns
    -------
    w : float
        FWHM in pixels; NaN when the fit fails.
    """
    from scipy.optimize import curve_fit
    n = img.shape[0]
    xc, yc = geometry.gauss_center(img)
    iy, ix = int(round(yc)), int(round(xc))
    sub = img[iy - 20:iy + 21, ix - 20:ix + 21]
    gy, gx = np.mgrid[iy - 20:iy + 21, ix - 20:ix + 21]
    ok = np.isfinite(sub)

    def model(xy, amp, x0, y0, sig, off):
        """Circular Gaussian plus constant at the points xy = (x, y)."""
        return amp * np.exp(-((xy[0] - x0) ** 2 + (xy[1] - y0) ** 2) / (2.0 * sig ** 2)) + off

    try:
        p, _ = curve_fit(model, (gx[ok], gy[ok]), sub[ok], p0=(float(np.nanmax(sub)), xc, yc, 7.0, 0.0), maxfev=20000)
    except (RuntimeError, ValueError):
        return float("nan")
    return 2.3548 * abs(p[3]) if n else float("nan")


def consistency(cut: np.ndarray) -> str:
    """Correlate every frame with the median frame, and measure the median frame's FWHM.

    Parameters
    ----------
    cut : ndarray, shape (n, 2 H, 2 H)
        Central pixels of n frames.

    Returns
    -------
    text : str
        Median and 5th percentile of the correlations of the central 64 x 64 pixels with the
        median frame's, and the median frame's FWHM in pixels.
    """
    med = np.nanmedian(cut, axis=0)
    core = med[H - 32:H + 32, H - 32:H + 32]
    rs = []
    for f in cut:
        a = f[H - 32:H + 32, H - 32:H + 32]
        ok = np.isfinite(a) & np.isfinite(core)
        rs.append(np.corrcoef(a[ok], core[ok])[0, 1] if ok.sum() > 10 else np.nan)
    rs = np.asarray(rs)
    return (f"correlation with the median frame (central 64x64): median {np.nanmedian(rs):.4f}, "
            f"5th percentile {np.nanpercentile(rs, 5):.4f}; median frame FWHM {fwhm(med):.2f} px")


def report(path: str, recenter: bool = False) -> List[str]:
    """Measure one cube; see the module docstring.

    Parameters
    ----------
    path : str
        FITS cube.
    recenter : bool, default False
        Also measure the frames after shifting each onto its Gaussian center.

    Returns
    -------
    lines : list of str
        Report lines.
    """
    with fits.open(path, memmap=True) as h:
        _, ny, nx = h[0].shape
        y0, x0 = ny // 2 - H, nx // 2 - H
        cut = np.asarray(h[0].section[:, y0:y0 + 2 * H, x0:x0 + 2 * H], dtype=np.float64)
    idl_xy = np.full((cut.shape[0], 2), np.nan)
    gau_xy = np.full((cut.shape[0], 2), np.nan)
    for i, f in enumerate(cut):
        for out, fn in ((idl_xy, geometry.idl_center), (gau_xy, geometry.gauss_center)):
            try:
                out[i] = fn(f)
            except RuntimeError:
                pass
    d = idl_xy - gau_xy
    d = d[np.all(np.isfinite(d), axis=1)]
    dmad = 1.4826 * np.median(np.abs(d - np.median(d, axis=0)), axis=0)
    lines = [f"== {os.path.basename(path)}: {cut.shape[0]} frames",
             f"  IDL estimator:      {robust(idl_xy + (x0, y0))}",
             f"  Gaussian estimator: {robust(gau_xy + (x0, y0))}",
             f"  IDL minus Gaussian: scatter ({dmad[0]:.2f}, {dmad[1]:.2f}) px",
             f"  as written: {consistency(cut)}"]
    if recenter:
        moved = np.stack([idl.fshift(f, H - 0.5 - g[0], H - 0.5 - g[1]) if np.all(np.isfinite(g)) else f
                          for f, g in zip(cut, gau_xy)])
        lines.append(f"  centered on each frame's Gaussian center: {consistency(moved)}")
    return lines


def main(argv: Optional[List[str]] = None) -> None:
    """Report every cube given; see the module docstring.

    Parameters
    ----------
    argv : list of str, optional
        ``--recenter`` and cube paths; default ``sys.argv[1:]``.

    Returns
    -------
    None
        Prints the report.
    """
    args = list(argv if argv is not None else sys.argv[1:])
    recenter = "--recenter" in args
    for path in [a for a in args if a != "--recenter"]:
        print("\n".join(report(path, recenter)), flush=True)


if __name__ == "__main__":
    main()
