#!/usr/bin/env python3
"""Measure the pre-clean noise of chop-subtracted frames, without and with subpixel shifts.

Reads a run's chop-subtracted work file, ``<side>_round0_chopsub.f32`` (n frames of 512 x 512
float32 in counts per second: the corrected chop-combined frames before the split, NaN where
masked), and for frames FIRST to FIRST + COUNT - 1 writes ``<OUT>_frames.csv`` with, per frame:

* ``frame``: the row in the work file (the chop-combined frame's index);
* ``chop``: A when the positive star image lies left of x = 256 (a LEFT-RIGHT chop), else B;
* ``x``, ``y``: the positive star image's centroid (``idl.cntrd``, FWHM 7 pixels; -1 when it
  fails), in detector pixels;
* ``px``, ``py``, ``nx``, ``ny``: the maximum and minimum of the 5-pixel boxcar-smoothed frame
  (NaN as 0), the positive and negative star images;
* ``peak``: the frame's maximum within 10 pixels of (px, py), in counts per second;
* ``noise``: the pre-clean noise measured on the unshifted frame, in counts per second: the
  standard deviation (ddof 1) of the finite pixels outside the 100 columns centered on the
  star (px - 50 to px + 49) and outside a 30-pixel disk around the negative image, the region
  the split stage and ``select.measure_and_select`` leave.

It also writes ``<OUT>_shift.csv``: the same noise of the first chop-A frame after
``idl.fshift`` by each subpixel shift (dx, dy) on a 0.1-pixel grid, divided by the unshifted
frame's noise. The alignment and centering stages shift every frame by such amounts before the
pre-clean measures its noise.

With ``--raw DIR``, it also writes ``<OUT>_raw.csv`` with the header values DATE-OBS,
TIME-OBS, TIME-END, LBT_UTC, EXPTIME, NCOADDS, ACQTIME, CHOPRATE, CHOP_POS and NOD_POS (empty
when missing) of the raw frames numbered RAW_FIRST to RAW_FIRST + RAW_COUNT - 1 (files
``n_<date>_<number>.fits``), for the frame cadence.

Usage: noise_sawtooth.py WORKFILE FIRST COUNT OUT [--raw DIR --raw-first K --raw-count M]
Run with the reduce_nomic package on PYTHONPATH.
"""
import argparse
import csv
import os
import sys
from typing import List, Optional, Tuple

import numpy as np

from reduce_nomic import idl

NX = NY = 512          # detector frame size, pixels
HALF = 50              # half width of the blanked strip of columns, as in the pre-clean
BLOCKRAD = 30.0        # radius of the blanked disk around the negative image, as in the split


def star_images(img: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Find the positive and negative star images of a chop-subtracted frame.

    Parameters
    ----------
    img : ndarray, shape (512, 512)
        Chop-subtracted frame in counts per second, NaN where masked.

    Returns
    -------
    pos : tuple of int
        (x, y) of the maximum of the 5-pixel boxcar-smoothed frame (NaN as 0).
    neg : tuple of int
        (x, y) of its minimum.
    """
    s = idl.smooth(np.nan_to_num(img, nan=0.0), 5)
    iy, ix = np.unravel_index(int(np.argmax(s)), s.shape)
    jy, jx = np.unravel_index(int(np.argmin(s)), s.shape)
    return (int(ix), int(iy)), (int(jx), int(jy))


def noise_outside_star(img: np.ndarray, xs: int, neg: Tuple[float, float], dist_neg: Optional[np.ndarray] = None) -> float:
    """Measure a frame's pre-clean noise: the scatter away from the star's columns.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        Frame in counts per second, NaN where masked, where ny and nx are its size.
    xs : int
        Column of the star; columns xs - 50 to xs + 49 are left out.
    neg : tuple of float
        (x, y) of the negative chop image; pixels within 30 pixels are left out.
    dist_neg : ndarray, shape (ny, nx), optional
        Distance of every pixel from ``neg`` in pixels; computed when None.

    Returns
    -------
    noise : float
        Standard deviation (ddof 1) of the remaining finite pixels, in counts per second.
    """
    w = img.copy()
    w[:, max(0, xs - HALF):max(0, xs + HALF)] = np.nan
    if dist_neg is None:
        yy, xx = np.mgrid[0:w.shape[0], 0:w.shape[1]]
        dist_neg = np.hypot(xx - neg[0], yy - neg[1])
    w[dist_neg < BLOCKRAD] = np.nan
    return float(np.nanstd(w, ddof=1))


def raw_headers(folder: str, first: int, count: int) -> List[List[str]]:
    """Read the timing and chop header values of consecutive raw frames.

    Parameters
    ----------
    folder : str
        The night's raw folder, holding ``n_<date>_<number>.fits``.
    first : int
        First frame number.
    count : int
        Number of frames.

    Returns
    -------
    rows : list of list of str
        Per frame found: number, DATE-OBS, TIME-OBS, TIME-END, LBT_UTC, EXPTIME, NCOADDS,
        ACQTIME, CHOPRATE, CHOP_POS, NOD_POS (empty when missing).
    """
    from astropy.io import fits
    names = {}
    for name in os.listdir(folder):
        stem = name[:-5] if name.endswith(".fits") else ""
        tail = stem.rsplit("_", 1)[-1]
        if stem.startswith("n_") and tail.isdigit():
            names[int(tail)] = name
    keys = ("DATE-OBS", "TIME-OBS", "TIME-END", "LBT_UTC", "EXPTIME", "NCOADDS", "ACQTIME", "CHOPRATE",
            "CHOP_POS", "NOD_POS")
    rows = []
    for num in range(first, first + count):
        if num not in names:
            continue
        h = fits.getheader(os.path.join(folder, names[num]))
        rows.append([str(num)] + [str(h.get(k, "")) for k in keys])
    return rows


def main(argv: Optional[List[str]] = None) -> None:
    """Run the measurement; see the module docstring.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; default ``sys.argv[1:]``.

    Returns
    -------
    None
        Writes ``<OUT>_frames.csv``, ``<OUT>_shift.csv`` and, with ``--raw``, ``<OUT>_raw.csv``.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("workfile", help="chop-subtracted work file, n x 512 x 512 float32")
    ap.add_argument("first", type=int, help="first frame (row) to measure")
    ap.add_argument("count", type=int, help="number of frames to measure")
    ap.add_argument("out", help="output prefix")
    ap.add_argument("--raw", help="raw folder, for the header values")
    ap.add_argument("--raw-first", type=int, default=0, help="first raw frame number")
    ap.add_argument("--raw-count", type=int, default=0, help="number of raw frames")
    a = ap.parse_args(argv)
    fsize = NX * NY * 4
    size = os.path.getsize(a.workfile)
    if size % fsize:
        raise SystemExit(f"{a.workfile}: {size} bytes is not a whole number of 512 x 512 float32 frames")
    n = size // fsize
    mm = np.memmap(a.workfile, dtype=np.float32, mode="r", shape=(n, NY, NX))
    yy, xx = np.mgrid[0:NY, 0:NX]
    rows = []
    first_a = None
    for i in range(max(0, a.first), min(n, a.first + a.count)):
        img = np.asarray(mm[i], dtype=np.float64)
        (px, py), neg = star_images(img)
        cx, cy = idl.cntrd(np.nan_to_num(img, nan=0.0), px, py, 7.0)
        near = np.hypot(xx - px, yy - py) <= 10
        peak = float(np.nanmax(img[near])) if np.isfinite(img[near]).any() else float("nan")
        dneg = np.hypot(xx - neg[0], yy - neg[1])
        noise = noise_outside_star(img, px, neg, dneg)
        chop = "A" if px < NX // 2 else "B"
        rows.append([i, chop, f"{cx:.3f}", f"{cy:.3f}", px, py, neg[0], neg[1], f"{peak:.6g}", f"{noise:.6g}"])
        if first_a is None and chop == "A":
            first_a = (img, px, neg, dneg, noise)
    with open(f"{a.out}_frames.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "chop", "x", "y", "px", "py", "nx", "ny", "peak", "noise"])
        w.writerows(rows)
    if first_a is not None:
        img0, px0, neg0, dneg0, base = first_a
        with open(f"{a.out}_shift.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dx", "dy", "noise_ratio"])
            for dx in np.round(np.arange(0.0, 1.0001, 0.1), 1):
                for dy in np.round(np.arange(0.0, 1.0001, 0.1), 1):
                    sh = idl.fshift(img0, float(dx), float(dy))
                    w.writerow([dx, dy, f"{noise_outside_star(sh, px0, neg0, dneg0) / base:.5f}"])
    if a.raw and a.raw_count > 0:
        with open(f"{a.out}_raw.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["number", "date_obs", "time_obs", "time_end", "lbt_utc", "exptime", "ncoadds", "acqtime",
                        "choprate", "chop_pos", "nod_pos"])
            w.writerows(raw_headers(a.raw, a.raw_first, a.raw_count))
    print(f"{len(rows)} frames measured of {n}; outputs {a.out}_*.csv")


if __name__ == "__main__":
    main(sys.argv[1:])
