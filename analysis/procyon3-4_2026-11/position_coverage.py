"""Measure how many of each partition's frames hold data at one sky position and on its circle.

usage: python analysis/procyon3-4_2026-11/position_coverage.py --part NAME CUBE ANGLES
       [--part NAME CUBE ANGLES ...] --rho RHO --pa PA [--half HALF] [--step STEP]
       [--radii R [R ...]] [--fwhm FWHM] [--pxscale PXSCALE] [--min-coverage F] [--out OUT]

A sky position falls, in frame i, on the detector position turned clockwise by that frame's angle,
since KLIP-TPE derotates a frame counterclockwise by its angle; the frame has data there if that
pixel of the centered (2 half)^2 cut is finite (``sky_coverage.frame_masks`` and
``sky_coverage.coverage`` of analysis/procyon4_2026-11, reused here). KLIP-TPE's ``--min-coverage``
compares this fraction, over every frame of the run, with its threshold before it injects a source
at a position or lets a noise aperture sample it. For the position (RHO, PA) this gives, for each
partition and for all frames together,

* the fraction of frames with data at the position itself;
* the fraction whose whole disk of radius one FWHM around it holds data (a source fully on the
  detector);
* the range of position angles on the circle at RHO where the coverage of all frames together
  reaches ``--min-coverage``, and that fraction of the circle,

and for each of ``--radii`` the fraction of the circle that reaches ``--min-coverage``.

Parameters
----------
--part : three str, repeatable
    A partition's name, its cube (n, ny, nx) with the star at the array center and NaN where there
    is no data, and its angles (n values, degrees; KLIP-TPE turns frame i counterclockwise by them).
--rho : float
    Separation of the position, arcsec.
--pa : float
    Position angle of the position, degrees east of north.
--half : int, default 340
    Half-width of the cut read from each frame, pixels; it must reach past RHO plus one FWHM.
--step : int, default 1
    Use every step-th frame.
--radii : float list, default 2.5 3.0 3.5 4.0 4.5 5.0 5.5
    Separations whose circles are summarized, arcsec.
--fwhm : float, default 15.653
    PSF FWHM, pixels.
--pxscale : float, default 0.0179
    Plate scale, arcsec per pixel.
--min-coverage : float, default 0.5
    Coverage threshold, as KLIP-TPE's ``--min-coverage``.
--out : str, optional
    Text file for the printed summary; not overwritten if it exists.

Returns
-------
stdout : text
    The summary described above.
OUT : file
    The same text [--out].
"""
import argparse
import importlib.util
import os
import sys
from typing import List, Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits

_HERE = os.path.dirname(os.path.abspath(__file__))
_SPEC = importlib.util.spec_from_file_location(
    "sky_coverage", os.path.join(_HERE, "..", "procyon4_2026-11", "sky_coverage.py"))
sky_coverage = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sky_coverage)


def disk_coverage(masks: np.ndarray, angles: np.ndarray, r_px: float, theta_deg: float,
                  radius_px: float) -> float:
    """Fraction of frames whose whole disk around a sky position holds data.

    Parameters
    ----------
    masks : ndarray of bool, shape (m, 2 half, 2 half)
        Finite-pixel masks of the m frames.
    angles : ndarray, shape (m,)
        Each frame's angle, degrees.
    r_px : float
        Separation of the disk's center, pixels.
    theta_deg : float
        Position angle of the disk's center, degrees east of north.
    radius_px : float
        Disk radius, pixels.

    Returns
    -------
    float
        Fraction of the m frames finite at every sky point of the disk sampled on a 2 px grid.
    """
    t = np.radians(theta_deg)
    cx, cy = -r_px * np.sin(t), r_px * np.cos(t)          # sky offset in pixels, north up, east left
    g = np.arange(-radius_px, radius_px + 1e-9, 2.0)
    gx, gy = np.meshgrid(g, g)
    inside = gx ** 2 + gy ** 2 <= radius_px ** 2
    sx, sy = cx + gx[inside], cy + gy[inside]
    m, n, _ = masks.shape
    c = (n - 1) / 2.0
    ok = np.ones(m, bool)
    for k in range(m):
        a = np.radians(angles[k])
        dx = np.cos(a) * sx + np.sin(a) * sy               # turned clockwise by the frame's angle
        dy = -np.sin(a) * sx + np.cos(a) * sy
        ix, iy = np.rint(c + dx).astype(int), np.rint(c + dy).astype(int)
        good = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
        if not good.all():
            ok[k] = False
            continue
        ok[k] = bool(masks[k][iy, ix].all())
    return float(ok.mean())


def covered_arc(frac: np.ndarray, theta: np.ndarray, threshold: float, pa: float) -> Tuple[float, float, bool]:
    """The run of position angles around ``pa`` whose coverage reaches a threshold.

    Parameters
    ----------
    frac : ndarray, shape (p,)
        Coverage at each position angle of ``theta``.
    theta : ndarray, shape (p,)
        Position angles, degrees, evenly spaced over the circle.
    threshold : float
        Coverage threshold.
    pa : float
        Position angle the run must hold, degrees.

    Returns
    -------
    start, end : float
        First and last position angle of the run, degrees (going east of north); NaN if ``pa``
        itself falls short.
    whole : bool
        True if the whole circle reaches the threshold.
    """
    ok = frac >= threshold
    if ok.all():
        return 0.0, 360.0, True
    p = theta.size
    i0 = int(np.argmin(np.abs((theta - pa + 180.0) % 360.0 - 180.0)))
    if not ok[i0]:
        return float("nan"), float("nan"), False
    lo = i0
    while ok[(lo - 1) % p]:
        lo -= 1
    hi = i0
    while ok[(hi + 1) % p]:
        hi += 1
    return float(theta[lo % p]), float(theta[hi % p]), False


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Print the coverage of the position and of the circles (see the module docstring).

    Parameters
    ----------
    argv : sequence of str or None, default None
        Arguments; None reads ``sys.argv``.

    Returns
    -------
    int
        0 on success.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--part", nargs=3, action="append", required=True, metavar=("NAME", "CUBE", "ANGLES"))
    ap.add_argument("--rho", type=float, required=True)
    ap.add_argument("--pa", type=float, required=True)
    ap.add_argument("--half", type=int, default=340)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--radii", type=float, nargs="+", default=[2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5])
    ap.add_argument("--fwhm", type=float, default=15.653)
    ap.add_argument("--pxscale", type=float, default=0.0179)
    ap.add_argument("--min-coverage", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.out and os.path.exists(a.out):
        raise SystemExit(f"refusing to overwrite {a.out}")
    rp = a.rho / a.pxscale
    if rp + a.fwhm > a.half - 1:
        raise SystemExit(f"--half {a.half} does not reach {a.rho}\" plus one FWHM ({rp + a.fwhm:.0f} px)")
    theta = np.arange(0.0, 360.0, 0.5)
    out: List[str] = [f"# coverage of ({a.rho:.3f}\", PA {a.pa:.1f} deg) = {rp:.1f} px; cut {2 * a.half} px; "
                      f"every {a.step} frame(s); min coverage {a.min_coverage}"]
    all_m, all_a = [], []
    for name, cube, angles in a.part:
        m = sky_coverage.frame_masks(cube, a.half, a.step)
        ang = np.asarray(fits.getdata(angles), float).ravel()[::a.step][:m.shape[0]]
        if ang.size != m.shape[0]:
            raise SystemExit(f"{name}: {m.shape[0]} frames but {ang.size} angles")
        all_m.append(m)
        all_a.append(ang)
        f0 = float(sky_coverage.coverage(m, ang, rp, np.array([a.pa]))[0])
        fd = disk_coverage(m, ang, rp, a.pa, a.fwhm)
        out.append(f"{name:12s} {m.shape[0]:4d} frames, angles {ang.min():7.2f} to {ang.max():7.2f} deg: "
                   f"data at the position in {f0:.3f}, whole 1-FWHM disk in {fd:.3f}")
    M, A = np.concatenate(all_m), np.concatenate(all_a)
    f0 = float(sky_coverage.coverage(M, A, rp, np.array([a.pa]))[0])
    fd = disk_coverage(M, A, rp, a.pa, a.fwhm)
    circ = sky_coverage.coverage(M, A, rp, theta)
    s, e, whole = covered_arc(circ, theta, a.min_coverage, a.pa)
    out.append(f"{'all':12s} {M.shape[0]:4d} frames: data at the position in {f0:.3f}, whole 1-FWHM disk in {fd:.3f}")
    arc = ("the whole circle" if whole else
           ("none: the position itself falls short" if not np.isfinite(s) else f"PA {s:.1f} to {e:.1f} deg"))
    out.append(f"circle at {a.rho:.3f}\": {np.mean(circ >= a.min_coverage):.0%} reaches {a.min_coverage}; "
               f"the covered run holding PA {a.pa:.1f}: {arc}")
    for r in a.radii:
        rr = r / a.pxscale
        if rr > a.half - 1:
            out.append(f"{r:4.2f}\": outside the cut")
            continue
        f = sky_coverage.coverage(M, A, rr, theta)
        out.append(f"{r:4.2f}\" ({rr:5.1f} px): {np.mean(f >= a.min_coverage):4.0%} of the circle reaches "
                   f"{a.min_coverage}, {np.mean(f > 0):4.0%} has any data; median {np.median(f):.2f}")
    text = "\n".join(out) + "\n"
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
