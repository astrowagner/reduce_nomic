"""Measure how far from the star each binned frame of a night is finite, and how the rest is filled.

usage: python analysis/procyon4_2026-11/coverage_check.py REDUCED_PY PREFIX [--half 200] [--step 1]
       [--chops 1 2]

KLIP-TPE's high-pass at the pinned version (``hp_nan_aware=False``) spreads one NaN pixel over the
whole frame, so a cut that reaches past the detector's edge loses the frame. For each of
``<PREFIX>_cube1_align_cen_bin.fits`` and ``<PREFIX>_cube2_align_cen_bin.fits`` in ``REDUCED_PY``
this prints the distribution of ``finite_half`` (tools/klip_tpe_inputs.py) over the frames, the
number of frames finite within a set of half-widths, the fraction of NaN pixels in rings around the
star, and the detector's finite extent in x and y in the first and last frame read.

Parameters
----------
REDUCED_PY : str
    The night's ``reduced_py`` folder.
PREFIX : str
    File prefix of the side, e.g. ``Procyon_20211220_left``.
--half : int, default 200
    Half-width, pixels, of the centered cut that is read.
--step : int, default 1
    Read every step-th frame (1 reads them all).
--chops : int list, default 1 2
    Chops to read, 1 (cube 1, chop A) and/or 2 (cube 2, chop B).

Returns
-------
stdout : text
    Per cube: the finite_half percentiles (pixels), frame counts finite within 150-180 px, NaN
    fraction per 20 px ring, and the finite x and y range relative to the star (pixels).
"""
import argparse
import os
import sys

import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from klip_tpe_inputs import finite_half  # noqa: E402


def read_cut(path: str, half: int, step: int) -> np.ndarray:
    """Read every step-th frame of a cube, cut to (2 half)² pixels around the array center.

    Parameters
    ----------
    path : str
        FITS cube, shape (n, ny, nx), with the star at the array center.
    half : int
        Half-width of the cut, pixels.
    step : int
        Frame stride.

    Returns
    -------
    cut : ndarray, float32, shape (m, 2 half, 2 half)
        The cut frames, where m = ceil(n / step).
    """
    with fits.open(path, memmap=True) as hd:
        d = hd[0].data
        cy, cx = d.shape[1] // 2, d.shape[2] // 2
        return np.array([np.asarray(d[k, cy - half:cy + half, cx - half:cx + half], dtype=np.float32)
                         for k in range(0, d.shape[0], step)])


def main(argv=None) -> int:
    """Print the coverage of both chops' binned cubes.

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
    ap.add_argument("--half", type=int, default=200)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--chops", type=int, nargs="+", default=[1, 2], choices=[1, 2])
    a = ap.parse_args(argv)
    for cn in a.chops:
        path = os.path.join(a.reduced_py, f"{a.prefix}_cube{cn}_align_cen_bin.fits")
        cut = read_cut(path, a.half, a.step)
        fh = finite_half(cut)
        q = np.percentile(fh, [0, 5, 50, 100])
        print(f"cube{cn}: {cut.shape[0]} frames read; finite_half px min {q[0]:.0f}, p05 {q[1]:.0f}, "
              f"median {q[2]:.0f}, max {q[3]:.0f}")
        print("  frames finite within half " + ", ".join(f"{h}: {np.count_nonzero(fh >= h)}"
                                                        for h in (150, 155, 160, 163, 165, 170, 175, 180)))
        bad = ~np.isfinite(cut)
        n2 = 2 * a.half
        yy, xx = np.mgrid[0:n2, 0:n2] - (n2 - 1) / 2.0
        rr = np.hypot(xx, yy)
        frac = bad.mean(axis=0)
        print("  NaN fraction per ring: " + ", ".join(
            f"{r0}-{r0 + 20} px {frac[(rr >= r0) & (rr < r0 + 20)].mean():.2f}" for r0 in range(120, a.half, 20)))
        for k in (0, cut.shape[0] - 1):
            fin = np.isfinite(cut[k])
            cols = xx[0][fin.any(axis=0)]
            rows = yy[:, 0][fin.any(axis=1)]
            print(f"  frame {k * a.step}: finite x {cols.min():+.1f} to {cols.max():+.1f} px, "
                  f"y {rows.min():+.1f} to {rows.max():+.1f} px from the star (cut edge {a.half - 0.5:.1f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
