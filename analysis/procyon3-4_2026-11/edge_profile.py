#!/usr/bin/env python3
"""Measure how pixel statistics change toward the detector's edge in frames that hold NaN past it.

For a cube whose frames are NaN beyond the detector's edge (KLIP-TPE inputs cut wider than the
detector), this takes each finite pixel's distance to the nearest NaN pixel of its own frame and
reports, in 1 px bins of that distance, the median and robust sigma of the pixel values, raw and
after KLIP-TPE's high-pass filter (the NaN-aware boxcar unsharp mask of ``klip_tpe.klip.highpass``,
copied here as `highpass`). Only pixels between ``rmin`` and ``rmax`` from the star and at least
``border`` px from the array's border are used, so the star's core and the filter's zeroed array
border stay out. Each frame's values are divided by the robust sigma of its own interior pixels
(edge distance at least ``dint``) before the frames are pooled, so every bin compares the edge with
the same frames' interior: 1 means no change toward the edge.

Usage
-----
python edge_profile.py CUBE [CUBE ...] --out PREFIX [--labels L ...] [--frames STEP]
    [--widths 0 25] [--rmin 139.66] [--rmax 307.26] [--border 30] [--dmax 40] [--dint 40]

CUBE is a FITS cube of shape (n, ny, nx) with the star at ((nx - 1) / 2, (ny - 1) / 2), as
KLIP-TPE keeps it. Writes PREFIX.csv (one row per cube, filter width and distance bin) and
PREFIX.png, and refuses to overwrite either.
"""

import argparse
import csv
import os

import numpy as np
from astropy.io import fits
from scipy import ndimage


def robust_sigma(a):
    """Return 1.4826 times the median absolute deviation of the finite values.

    Parameters
    ----------
    a : array_like
        Values; non-finite ones are ignored.

    Returns
    -------
    float
        The robust sigma, NaN with no finite value.
    """
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(a - np.median(a))))


def highpass(img, width):
    """Subtract a NaN-aware boxcar-smoothed copy of a frame, as ``klip_tpe.klip.highpass`` does.

    The boxcar width is made odd, the pixels within ``width // 2`` of the array's border come out
    0, and NaN pixels stay NaN (klip-tpe bf99474, ``_highpass_frames`` with ``nan_aware=True``).

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        The frame.
    width : int
        Boxcar width in pixels; 1 or less returns the frame unfiltered.

    Returns
    -------
    ndarray, shape (ny, nx)
        The filtered frame, float64.
    """
    img = np.asarray(img, np.float64)
    w = int(width)
    if w <= 1:
        return img.copy()
    if w % 2 == 0:
        w += 1
    fin = np.isfinite(img)
    num = ndimage.uniform_filter(np.where(fin, img, 0.0), (w, w), mode="constant")
    den = ndimage.uniform_filter(fin.astype(float), (w, w), mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        sm = np.where(den > 0, num / den, np.nan)
    out = img - sm
    h = w // 2
    out[:h, :] = 0.0
    out[-h:, :] = 0.0
    out[:, :h] = 0.0
    out[:, -h:] = 0.0
    out[~fin] = np.nan
    return out


def edge_distance(frame):
    """Return each finite pixel's distance to the nearest NaN pixel of the frame.

    Parameters
    ----------
    frame : ndarray, shape (ny, nx)
        The frame.

    Returns
    -------
    ndarray, shape (ny, nx) or None
        Euclidean distance in pixels (0 at NaN pixels), or None when the frame has no NaN pixel.
    """
    fin = np.isfinite(frame)
    if fin.all():
        return None
    return ndimage.distance_transform_edt(fin)


def profile(cube, widths, step=1, rmin=139.66, rmax=307.26, border=30, dmax=40, dint=40.0):
    """Pool every used frame's normalized values by edge distance, raw and high-passed.

    Parameters
    ----------
    cube : ndarray, shape (n, ny, nx)
        The frames, NaN beyond the detector's edge; n frames of ny by nx pixels.
    widths : sequence of int
        High-pass boxcar widths, in pixels; 0 is the raw frame.
    step : int, default 1
        Use every ``step``-th frame.
    rmin, rmax : float, default 139.66, 307.26
        Radii from the star, in pixels, between which pixels are used.
    border : int, default 30
        Fewest pixels from the array's border for a pixel to be used.
    dmax : int, default 40
        Distance bins are 1 px wide from 0 to ``dmax``; farther pixels pool in a last bin.
    dint : float, default 40.0
        Edge distance, in pixels, from which a pixel is interior, for the normalization.

    Returns
    -------
    dict of int to list of ndarray
        For each width, ``dmax + 1`` arrays of normalized values: bin i holds the pixels with
        ``i <= distance < i + 1`` for i < dmax, and bin ``dmax`` those at ``dmax`` or more.
    int
        Number of frames used (those with a NaN pixel).
    """
    n, ny, nx = cube.shape
    yy, xx = np.mgrid[:ny, :nx]
    r = np.hypot(xx - (nx - 1) / 2.0, yy - (ny - 1) / 2.0)
    far = (yy >= border) & (yy < ny - border) & (xx >= border) & (xx < nx - border)
    region = (r >= rmin) & (r <= rmax) & far
    out = {w: [[] for _ in range(dmax + 1)] for w in widths}
    used = 0
    for i in range(0, n, max(int(step), 1)):
        frame = np.asarray(cube[i], np.float64)
        d = edge_distance(frame)
        if d is None:
            continue
        used += 1
        sel = region & np.isfinite(frame)
        b = np.minimum(np.floor(d[sel]).astype(int), dmax)
        for w in widths:
            v = highpass(frame, w)[sel]
            s = robust_sigma(v[d[sel] >= dint])
            if not np.isfinite(s) or s <= 0:
                continue
            v = (v - np.median(v[d[sel] >= dint])) / s
            for k in range(dmax + 1):
                vk = v[b == k]
                if vk.size:
                    out[w][k].append(vk.astype(np.float32))
    return {w: [np.concatenate(x) if x else np.zeros(0, np.float32) for x in bins] for w, bins in out.items()}, used


def summarize(pooled):
    """Return the count, median and robust sigma of each distance bin.

    Parameters
    ----------
    pooled : list of ndarray
        One array of normalized values per distance bin, from `profile`.

    Returns
    -------
    list of tuple
        (bin index, count, median, robust sigma) per bin; NaN statistics for an empty bin.
    """
    rows = []
    for k, v in enumerate(pooled):
        if v.size:
            rows.append((k, int(v.size), float(np.median(v)), robust_sigma(v)))
        else:
            rows.append((k, 0, float("nan"), float("nan")))
    return rows


def main(argv=None):
    """Parse the command line, measure each cube, and write the table and the figure.

    Parameters
    ----------
    argv : list of str or None, default None
        Command-line arguments; None reads ``sys.argv[1:]``.

    Returns
    -------
    None
        Writes ``PREFIX.csv`` and ``PREFIX.png`` and prints the table's first bins.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cubes", nargs="+", help="FITS cubes (n, ny, nx), NaN beyond the detector's edge")
    ap.add_argument("--out", required=True, help="output prefix for .csv and .png")
    ap.add_argument("--labels", nargs="+", default=None, help="a label per cube (default the file names)")
    ap.add_argument("--frames", type=int, default=1, help="use every N-th frame (default 1)")
    ap.add_argument("--widths", type=int, nargs="+", default=[0, 25],
                    help="high-pass boxcar widths in pixels, 0 for raw (default 0 25)")
    ap.add_argument("--rmin", type=float, default=139.66, help="inner radius in pixels (default 139.66)")
    ap.add_argument("--rmax", type=float, default=307.26, help="outer radius in pixels (default 307.26)")
    ap.add_argument("--border", type=int, default=30, help="fewest pixels from the array border (default 30)")
    ap.add_argument("--dmax", type=int, default=40, help="last distance bin in pixels (default 40)")
    ap.add_argument("--dint", type=float, default=40.0, help="interior edge distance in pixels (default 40)")
    a = ap.parse_args(argv)
    labels = a.labels or [os.path.basename(c) for c in a.cubes]
    if len(labels) != len(a.cubes):
        raise SystemExit("--labels needs one label per cube")
    for ext in (".csv", ".png"):
        if os.path.exists(a.out + ext):
            raise SystemExit(f"{a.out + ext} exists; not overwriting")
    results = []
    for path, lab in zip(a.cubes, labels):
        with fits.open(path, memmap=True) as h:
            pooled, used = profile(h[0].data, a.widths, a.frames, a.rmin, a.rmax, a.border, a.dmax, a.dint)
        for w in a.widths:
            results.append((lab, w, used, summarize(pooled[w])))
        print(f"{lab}: {used} frames used")
    with open(a.out + ".csv", "x", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["cube", "width_px", "frames", "distance_bin_px", "n_pixels", "median_over_sigma",
                     "robust_sigma_ratio"])
        for lab, w, used, rows in results:
            for k, cnt, med, sig in rows:
                wr.writerow([lab, w, used, k, cnt, f"{med:.4f}", f"{sig:.4f}"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(a.widths), figsize=(5.2 * len(a.widths), 7.6), squeeze=False, sharex=True)
    for j, w in enumerate(a.widths):
        for lab, ww, used, rows in results:
            if ww != w:
                continue
            k = np.array([r[0] for r in rows[:-1]]) + 0.5
            axes[0, j].plot(k, [r[3] for r in rows[:-1]], marker="o", ms=3, label=lab)
            axes[1, j].plot(k, [r[2] for r in rows[:-1]], marker="o", ms=3, label=lab)
        name = "Raw Frames" if w <= 1 else f"High-Passed Frames (Boxcar {w} px)"
        axes[0, j].set_title(name)
        axes[0, j].axhline(1.0, color="0.5", lw=0.8)
        axes[1, j].axhline(0.0, color="0.5", lw=0.8)
        axes[0, j].set_ylabel("Robust Sigma / Interior Robust Sigma")
        axes[1, j].set_ylabel("Median / Interior Robust Sigma")
        axes[1, j].set_xlabel("Distance to the Frame's NaN Edge (px)")
        for ax in axes[:, j]:
            ax.tick_params(direction="in", top=True, right=True)
            for s in ax.spines.values():
                s.set_visible(True)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out + ".png", dpi=110)
    for lab, w, used, rows in results:
        print(lab, f"width {w}:", " ".join(f"{k}:{sig:.2f}/{med:+.2f}" for k, cnt, med, sig in rows[:8]),
              f"... interior {rows[-1][3]:.2f}/{rows[-1][2]:+.2f}")


if __name__ == "__main__":
    main()
