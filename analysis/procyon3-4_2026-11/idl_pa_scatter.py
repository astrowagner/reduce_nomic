#!/usr/bin/env python3
"""Tabulate the scatter over position angle of Kevin's IDL contrast curves at a common S/N.

usage: python analysis/procyon3-4_2026-11/idl_pa_scatter.py OUT_CSV SNR_REF LABEL:SNR:PATH [LABEL:SNR:PATH ...]
       [--clip SIGMA]

For each curve and separation: the mean contrast over position angle, its sample standard
deviation (ddof 1) and the number of position angles, taken from the values that the IDL tables of
``../procyon4_2026-11/idl_curves_compare.py`` average, so a band of mean ± standard deviation is
centered on the plotted IDL curve. A save file (``*_contrast_output.sav`` of
``contrastcurve_nomic.pro``) gives ``curves``, one row per position angle; only finite, positive
contrasts count, as in the tables (NaN is a test not recovered within the trial cap, 0 a failed
test). A PDF gives its gray curves (``../procyon4_2026-11/idl_curve_pdf.py``). A PDF drawn before
its last test finished holds unfinished position angles: with test k of N under way (the title's
"Test k / N", N the number of position angles times the n_sep separations), the first
(k - 1) // n_sep position angles are complete, since the tests run one position angle at a time,
and only those count. The PDF's black line must then equal their mean, which is checked. With
``--clip``, the values of each separation are sigma-clipped over position angle (``pa_clip.py``)
before the mean and standard deviation are taken; the mean of all of them, the IDL curve's value,
is kept beside. A curve that targeted S/N ``SNR`` is scaled to ``SNR_REF`` by ``SNR_REF / SNR``,
as in the tables.

Parameters
----------
OUT_CSV : str
    Table to write, one row per separation that any curve measured.
SNR_REF : float
    The common S/N, e.g. 5.
LABEL:SNR:PATH : str
    A curve: a short label (the column name in the IDL table), the S/N it targeted and its save
    file or PDF.
--clip : float, optional
    Clipping threshold over position angle, in standard deviations; none by default.

Returns
-------
stdout : text
    Per curve, its source and the position angles used (for a PDF, the test counter and the
    largest difference between the black line and the mean), each value clipped [--clip], and the
    table.
OUT_CSV : file
    Columns ``separation_arcsec``; per curve ``<LABEL>_mean_all`` (the mean of every value counted,
    the IDL curve's value), ``<LABEL>_mean`` and ``<LABEL>_sd`` (mean and sample standard deviation
    of the values kept), ``<LABEL>_n`` (values kept), ``<LABEL>_n_clipped`` (values clipped) and
    ``<LABEL>_n_pa`` (position angles in the file), contrasts at SNR_REF, empty where the curve has
    no point and ``<LABEL>_sd`` empty where one value is kept; and ``clip_sigma`` (the threshold,
    empty without ``--clip``).
"""
import argparse
import csv
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.io import readsav

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "procyon4_2026-11"))
from idl_curve_pdf import read_curves  # noqa: E402
from pa_clip import sigma_clip_mask  # noqa: E402

#: largest relative difference allowed between a PDF's black line and the mean of its complete
#: position angles (the vertices are read to about 0.1 %)
BLACK_TOL = 0.005

#: per separation: mean of all values, mean and sample standard deviation of those kept, values
#: kept, values clipped
Stats = Tuple[float, float, float, int, int]


def _stats(v: np.ndarray, clip: Optional[float]) -> Tuple[Stats, np.ndarray]:
    """Return the statistics of a set of contrasts over position angle, after an optional clip.

    Parameters
    ----------
    v : ndarray, shape (m,)
        Finite, positive contrasts at one separation, where m is their number.
    clip : float or None
        Clipping threshold in standard deviations (``pa_clip.sigma_clip_mask``); None keeps all.

    Returns
    -------
    stats : tuple of (float, float, float, int, int)
        Mean of all m values, mean and sample standard deviation (ddof 1; NaN for one value) of the
        values kept, and the numbers kept and clipped.
    keep : ndarray of bool, shape (m,)
        True for the values kept.
    """
    keep = sigma_clip_mask(v, clip) if clip else np.ones(v.size, bool)
    k = v[keep]
    sd = float(k.std(ddof=1)) if k.size > 1 else float("nan")
    return (float(v.mean()), float(k.mean()), sd, int(k.size), int(v.size - k.size)), keep


def scatter_sav(path: str, clip: Optional[float] = None) -> Tuple[Dict[float, Stats], int, str, List[str]]:
    """Return a save file's statistics over position angle per separation.

    Parameters
    ----------
    path : str
        A ``*_contrast_output.sav`` holding ``curves`` (position angle by separation), ``pas`` and
        ``separations`` (arcsec).
    clip : float or None, default None
        Clipping threshold in standard deviations; None keeps every value.

    Returns
    -------
    stats : dict of float to tuple
        Separation (arcsec, 2 decimals) to the statistics of the finite, positive contrasts there
        (see `_stats`); separations with none are left out.
    n_pa : int
        Number of position angles in the file.
    note : str
        One line on the source, for the printed report.
    clipped : list of str
        One line per separation with a value clipped: the values and their position angles.
    """
    s = readsav(path, python_dict=True)
    c, sep, pa = np.asarray(s["curves"], float), np.asarray(s["separations"], float), np.asarray(s["pas"])
    if c.shape != (pa.size, sep.size):
        c = c.T
    if c.shape != (pa.size, sep.size):
        raise SystemExit(f"{path}: curves {c.shape} does not match {pa.size} position angles x {sep.size} separations")
    out, lines = {}, []
    for j, r in enumerate(sep):
        ok = np.isfinite(c[:, j]) & (c[:, j] > 0)
        if ok.any():
            out[round(float(r), 2)], keep = _stats(c[ok, j], clip)
            if not keep.all():
                gone = [f"{v:.3e} (position angle {float(p):g})" for v, p, kk in zip(c[ok, j], pa[ok], keep) if not kk]
                lines.append(f"{float(r):.2f}\": clipped {', '.join(gone)}")
    return out, int(pa.size), f"save file, {pa.size} position angles", lines


def scatter_pdf(path: str, clip: Optional[float] = None) -> Tuple[Dict[float, Stats], int, str, List[str]]:
    """Return a contrast-curve PDF's statistics over its complete position angles per separation.

    Parameters
    ----------
    path : str
        A PDF plotted by ``contrastcurve_nomic.pro``: one gray curve per position angle, in
        position-angle order, and their mean in black.
    clip : float or None, default None
        Clipping threshold in standard deviations; None keeps every value.

    Returns
    -------
    stats : dict of float to tuple
        Separation (arcsec, 2 decimals) to the statistics of the gray curves of the complete
        position angles (see `_stats`).
    n_pa : int
        Number of gray curves (position angles) in the PDF.
    note : str
        One line on the source: the test counter, the position angles used and the largest
        relative difference between the black line and their mean.
    clipped : list of str
        One line per separation with a value clipped: the values and their gray curves (0-based,
        in drawing order).

    Raises
    ------
    SystemExit
        If the gray curves do not share one set of separations, the counter does not match them,
        no position angle is complete or the black line differs from the mean of the complete
        position angles by more than `BLACK_TOL`.
    """
    d = read_curves(path)
    gray = d["pa"]
    n_pa = len(gray)
    if not n_pa:
        raise SystemExit(f"{path}: no gray curves")
    seps = [round(float(r), 2) for r, _ in gray[0]]
    if any([round(float(r), 2) for r, _ in g] != seps for g in gray):
        raise SystemExit(f"{path}: the gray curves do not share one set of separations")
    vals = np.array([[float(v) for _, v in g] for g in gray])
    n_sep = len(seps)
    m = re.search(r"Test\s+(\d+)\s*/\s*(\d+)", d.get("title") or "")
    if m:
        k, n_tests = int(m.group(1)), int(m.group(2))
        if n_tests != n_pa * n_sep:
            raise SystemExit(f"{path}: counter {k}/{n_tests} does not match {n_pa} position angles "
                             f"x {n_sep} separations")
        n_done = min(n_pa, (k - 1) // n_sep)
        counter = f"test {k} of {n_tests}"
    else:
        n_done, counter = n_pa, "no test counter"
    if n_done < 1:
        raise SystemExit(f"{path}: no complete position angle ({counter})")
    black = {round(float(r), 2): float(v) for r, v in d["mean"]}
    out, worst, lines = {}, 0.0, []
    for j, r in enumerate(seps):
        v = vals[:n_done, j]
        idx = np.where(np.isfinite(v) & (v > 0))[0]
        if not idx.size:
            continue
        out[r], keep = _stats(v[idx], clip)
        if r in black:
            worst = max(worst, abs(black[r] / out[r][0] - 1.0))
        if not keep.all():
            gone = [f"{v[i]:.3e} (gray curve {int(i)})" for i, kk in zip(idx, keep) if not kk]
            lines.append(f"{r:.2f}\": clipped {', '.join(gone)}")
    if worst > BLACK_TOL:
        raise SystemExit(f"{path}: the black line differs from the mean of the first {n_done} position angles "
                         f"by {worst:.2%} (> {BLACK_TOL:.1%}); the complete position angles are not known")
    return out, n_pa, (f"PDF, {counter}: the first {n_done} of {n_pa} position angles complete; "
                       f"black line within {worst:.2%} of their mean"), lines


def main(argv=None) -> int:
    """Build, write and print the table of means, standard deviations and counts.

    Parameters
    ----------
    argv : list of str or None, default None
        Command-line arguments (OUT_CSV SNR_REF LABEL:SNR:PATH ... [--clip SIGMA]); None reads
        ``sys.argv[1:]``.

    Returns
    -------
    status : int
        0.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out_csv")
    ap.add_argument("snr_ref", type=float)
    ap.add_argument("curves", nargs="+", metavar="LABEL:SNR:PATH")
    ap.add_argument("--clip", type=float, default=None, metavar="SIGMA")
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    if a.clip is not None and not a.clip > 0:
        raise SystemExit("--clip must be positive")
    curves: List[Tuple[str, Dict[float, Stats], int]] = []
    for spec in a.curves:
        label, snr, path = spec.split(":", 2)
        k = a.snr_ref / float(snr)
        reader = scatter_sav if path.lower().endswith(".sav") else scatter_pdf
        stats, n_pa, note, clipped = reader(path, a.clip)
        print(f"{label}: {os.path.basename(path)} ({note}); S/N {float(snr):g} scaled by {k:g}")
        for line in clipped:
            print(f"  {line}")
        curves.append((label, {s: (m0 * k, m1 * k, sd * k, n, nc) for s, (m0, m1, sd, n, nc) in stats.items()}, n_pa))
    if a.clip:
        print(f"clipped over position angle at {a.clip:g} sigma (median-centered, iterated)")
    seps = sorted({s for _, c, _ in curves for s in c})
    fields = ("mean_all", "mean", "sd", "n", "n_clipped", "n_pa")
    with open(a.out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["separation_arcsec"] + [f"{lab}_{x}" for lab, _, _ in curves for x in fields] + ["clip_sigma"])
        print("sep(\")  " + "  ".join(f"{lab + ' mean ± sd [n]':>30s}" for lab, _, _ in curves))
        for s in seps:
            row, cells = [f"{s:.2f}"], []
            for _, c, n_pa in curves:
                if s in c:
                    m0, m1, sd, n, nc = c[s]
                    row += [f"{m0:.4e}", f"{m1:.4e}", "" if not np.isfinite(sd) else f"{sd:.4e}", str(n), str(nc),
                            str(n_pa)]
                    cells.append(f"{m1:.3e} ± {sd:.2e} [{n}/{n_pa}]")
                else:
                    row += [""] * len(fields)
                    cells.append("")
            w.writerow(row + ["" if a.clip is None else f"{a.clip:g}"])
            print(f"{s:5.2f}  " + "  ".join(f"{x:>30s}" for x in cells))
    print(f"wrote {a.out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
