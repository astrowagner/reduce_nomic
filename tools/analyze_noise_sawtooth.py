#!/usr/bin/env python3
"""Summarize the outputs of noise_sawtooth.py: chop sequence, periods and shift modulation.

Reads ``<PREFIX>_frames.csv``, ``<PREFIX>_shift.csv`` and, when present, ``<PREFIX>_raw.csv``
(written by ``noise_sawtooth.py``) and prints:

* the chop sequence: where the chop fails to alternate (a frame in the same chop as the frame
  before it, a "repeat") and the spacing of the repeats, in frames;
* for each chop, the strongest periods (Lomb-Scargle, astropy) of the noise, the peak and the
  star's x and y centroids, in frames, after removing a running median over 101 of that chop's
  frames;
* for each chop, the median noise and peak against the number of frames since the last repeat;
* a stretch of the noise of each chop, for a look at the pattern;
* the noise ratio after subpixel shifts at (0.5, 0), (0, 0.5) and (0.5, 0.5) pixels;
* the raw frame cadence from TIME-OBS, the chop half-period 1/CHOPRATE, and their beat period,
  in frames.

Usage: analyze_noise_sawtooth.py PREFIX [--stretch FIRST COUNT]
"""
import argparse
import csv
import os
import sys
from typing import Dict, List, Optional

import numpy as np


def read_csv(path: str) -> Dict[str, np.ndarray]:
    """Read a CSV file into columns.

    Parameters
    ----------
    path : str
        CSV file with a header row.

    Returns
    -------
    cols : dict of str to ndarray
        Each column as floats when every entry parses as a number, else as strings.
    """
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    cols = {}
    for j, name in enumerate(rows[0]):
        vals = [r[j] for r in rows[1:]]
        try:
            cols[name] = np.array([float(v) if v != "" else np.nan for v in vals])
        except ValueError:
            cols[name] = np.array(vals)
    return cols


def detrended(v: np.ndarray, width: int = 101) -> np.ndarray:
    """Subtract a running median from a series.

    Parameters
    ----------
    v : ndarray, shape (m,)
        Series, where m is its length.
    width : int, default 101
        Running-median window, in samples.

    Returns
    -------
    d : ndarray, shape (m,)
        ``v`` minus its running median (edges by the nearest full window).
    """
    from scipy.ndimage import median_filter
    return v - median_filter(v, size=min(width, max(1, v.size)), mode="nearest")


def top_periods(t: np.ndarray, v: np.ndarray, pmin: float = 4.0, pmax: float = 1000.0, k: int = 4) -> List[str]:
    """Find the strongest periods of an unevenly sampled series.

    Parameters
    ----------
    t : ndarray, shape (m,)
        Sample positions (frame numbers), where m is their number.
    v : ndarray, shape (m,)
        Values.
    pmin, pmax : float, default 4.0 and 1000.0
        Shortest and longest period searched, in frames.
    k : int, default 4
        Number of separate peaks reported.

    Returns
    -------
    peaks : list of str
        ``"<period> (<power>)"`` for the k highest local maxima of the Lomb-Scargle power.
    """
    from astropy.timeseries import LombScargle
    ok = np.isfinite(t) & np.isfinite(v)
    freq = np.linspace(1.0 / pmax, 1.0 / pmin, 20000)
    pw = LombScargle(t[ok], v[ok]).power(freq)
    loc = np.nonzero((pw[1:-1] > pw[:-2]) & (pw[1:-1] >= pw[2:]))[0] + 1
    loc = loc[np.argsort(pw[loc])[::-1]][:k]
    return [f"{1.0 / freq[i]:.1f} ({pw[i]:.3f})" for i in loc]


def seconds(hms: str) -> float:
    """Convert hh:mm:ss.sss to seconds.

    Parameters
    ----------
    hms : str
        Time of day.

    Returns
    -------
    s : float
        Seconds since midnight; NaN when it does not parse.
    """
    try:
        h, m, s = hms.strip().split(":")
        return 3600.0 * float(h) + 60.0 * float(m) + float(s)
    except ValueError:
        return float("nan")


def main(argv: Optional[List[str]] = None) -> None:
    """Print the summary; see the module docstring.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; default ``sys.argv[1:]``.

    Returns
    -------
    None
        Prints to standard output.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("prefix", help="output prefix given to noise_sawtooth.py")
    ap.add_argument("--stretch", nargs=2, type=int, default=(0, 120), metavar=("FIRST", "COUNT"),
                    help="chop frames of each chop to print (index within the chop)")
    a = ap.parse_args(argv)
    fr = read_csv(f"{a.prefix}_frames.csv")
    num, chop = fr["frame"], fr["chop"]
    rep = np.nonzero(chop[1:] == chop[:-1])[0] + 1
    print(f"frames {int(num[0])}-{int(num[-1])}: {np.sum(chop == 'A')} A, {np.sum(chop == 'B')} B, "
          f"{rep.size} repeats")
    if rep.size > 1:
        sp = np.diff(num[rep])
        print(f"  repeat spacing (frames): median {np.median(sp):.0f}, min {sp.min():.0f}, max {sp.max():.0f}; "
              f"first: {' '.join(str(int(s)) for s in sp[:25])}")
        print(f"  repeats in chop: {' '.join(chop[rep][:25])}")
    repset = set(rep.tolist())
    since = np.full(num.size, -1)
    last = -1
    for i in range(num.size):
        if i in repset:
            last = i
        since[i] = i - last if last >= 0 else -1
    for c in ("A", "B"):
        m = chop == c
        t = num[m]
        print(f"chop {c}: {m.sum()} frames; noise median {np.median(fr['noise'][m]):.4g}, "
              f"rms of detrended {np.std(detrended(fr['noise'][m])):.3g}; peak median {np.median(fr['peak'][m]):.4g}")
        for key in ("noise", "peak", "x", "y"):
            v = fr[key][m].astype(float)
            if key in ("x", "y"):
                v = np.where(v < 0, np.nan, v)
            print(f"  {key:5s} periods, frames (power): {', '.join(top_periods(t, detrended(np.nan_to_num(v, nan=np.nanmedian(v)))))}")
        if rep.size:
            s = since[m]
            edges = [0, 5, 10, 20, 30, 40, 60, 80, 120, 200, 400, 10**9]
            parts = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                q = (s >= lo) & (s < hi)
                if q.sum() >= 3:
                    parts.append(f"{lo}-{hi if hi < 10**9 else ''}: {np.median(fr['noise'][m][q]):.4g}/"
                                 f"{np.median(fr['peak'][m][q]):.4g} ({q.sum()})")
            print(f"  median noise/peak by frames since last repeat: {'; '.join(parts)}")
        lo, cnt = a.stretch
        seg = fr["noise"][m][lo:lo + cnt]
        print(f"  noise of {c} frames {lo}-{lo + seg.size - 1} (x1000 / median): "
              + " ".join(f"{1000 * v / np.median(fr['noise'][m]):.0f}" for v in seg))
    sh_path = f"{a.prefix}_shift.csv"
    if os.path.exists(sh_path):
        sh = read_csv(sh_path)
        for dx, dy in ((0.5, 0.0), (0.0, 0.5), (0.5, 0.5), (0.25, 0.25)):
            q = np.isclose(sh["dx"], dx) & np.isclose(sh["dy"], dy)
            if q.any():
                print(f"noise after fshift by ({dx}, {dy}) px: {sh['noise_ratio'][q][0]:.4f} x unshifted")
        print(f"noise ratio range over the 0.1-px grid: {sh['noise_ratio'].min():.4f}-{sh['noise_ratio'].max():.4f}")
    raw_path = f"{a.prefix}_raw.csv"
    if os.path.exists(raw_path):
        rw = read_csv(raw_path)
        t = np.array([seconds(str(s)) for s in rw["time_obs"]])
        dt = np.diff(t)
        n = rw["number"]
        cad = (t[-1] - t[0]) / (n[-1] - n[0])
        half = 1.0 / float(np.nanmedian(rw["choprate"])) if "choprate" in rw else float("nan")
        big = np.nonzero(dt > 1.5 * np.median(dt))[0]
        print(f"raw frames {int(n[0])}-{int(n[-1])}: TIME-OBS step median {np.median(dt):.4f} s "
              f"(min {dt.min():.4f}, max {dt.max():.4f}), mean cadence {cad:.6f} s; "
              f"EXPTIME {np.nanmedian(rw['exptime']):.4f} s, ACQTIME {np.nanmedian(rw['acqtime']):.6f} s; "
              f"chop half-period 1/CHOPRATE {half:.6f} s")
        if np.isfinite(half) and cad != half:
            print(f"  beat of cadence and chop half-period: {half / abs(cad - half):.0f} frames "
                  f"({half * half / abs(cad - half):.1f} s)")
        print(f"  steps over 1.5x the median: {big.size}"
              + (f" at frames {' '.join(str(int(n[i])) for i in big[:20])}" if big.size else ""))


if __name__ == "__main__":
    main(sys.argv[1:])
