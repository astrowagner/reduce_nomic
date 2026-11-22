"""Compare a KLIP-TPE run's contrast curves with Kevin's IDL curves at S/N 5.

usage: python analysis/procyon4_2026-11/compare_run_idl.py RUN_DIR IDL_CSV [--sep S [S ...]]
       [--tol TOL] [--out OUT_CSV]

The run's curves come from its contrast-curve files: each annulus' ``annulusNN/contrast_curve.txt``
(the injection-calibrated 5-sigma curve, the KLIP-FM cross-check and the injection-recovery grid)
and, once the run has finished, the run's own ``contrast_curve.txt`` (the final-image grid's curve
and the KLIP-FM cross-check). Each curve is interpolated linearly in separation and contrast at the
IDL table's separations and at any given with ``--sep``. A separation outside a curve's span by at
most TOL takes the curve's end value and is flagged ``end``; farther out the curve has no value
there. An IDL curve without a point at a separation is interpolated linearly between its nearest
points on either side and flagged ``~``; without a point on one side it has no value there. The
ratio is the IDL contrast over the run's, so above 1 the run goes deeper.

Parameters
----------
RUN_DIR : str
    The KLIP-TPE run directory.
IDL_CSV : str
    The IDL curves at S/N 5, as ``idl_curves_compare.py`` writes them: ``separation_arcsec``, one
    column per curve, ``deepest`` and ``deepest_curve``.
--sep : float, optional
    Further separations to compare at, in arcsec.
--tol : float, default 0.02
    Distance beyond a curve's first or last point within which its end value stands, in arcsec.
--out : str, optional
    CSV to write, one row per curve and separation: the run's contrast and flag, each IDL curve's
    contrast, flag and ratio, and the deepest IDL curve's contrast, label, flag and ratio.

Returns
-------
stdout : text
    A table per curve: separation, the run's contrast, each IDL curve's contrast and ratio, and the
    deepest IDL curve's ratio.
OUT_CSV : file
    The same rows [--out].
"""
import argparse
import csv
import glob
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Section kinds, in the order the tables are printed, with their titles.
KINDS = {"inj": "Injection-Calibrated Curve", "grid": "Injection-Recovery Grid",
         "fm": "KLIP-FM Cross-Check", "final": "Final-Image Grid Curve", "main": "Main Curve"}


def read_sections(path: str) -> List[Tuple[str, np.ndarray]]:
    """Read the numeric sections of a KLIP-TPE ``contrast_curve.txt``.

    A section is a run of numeric rows; its title is every comment line since the previous run,
    joined with `` | ``.

    Parameters
    ----------
    path : str
        The contrast-curve file.

    Returns
    -------
    sections : list of (str, ndarray, shape (m, 2))
        Each section's title and its rows, separation (arcsec) and contrast; m is the number of
        rows.
    """
    sections, titles, rows = [], [], []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                if rows:
                    sections.append((" | ".join(titles), np.asarray(rows, float)))
                    titles, rows = [], []
                titles.append(s.lstrip("#").strip())
                continue
            parts = s.split()
            rows.append([float(parts[0]), float(parts[1])])
    if rows:
        sections.append((" | ".join(titles), np.asarray(rows, float)))
    return sections


def classify(title: str) -> str:
    """Name a section's kind from its title.

    Parameters
    ----------
    title : str
        The section's title, as :func:`read_sections` gives it.

    Returns
    -------
    kind : str
        ``"samples"`` (per-sample points), ``"fm"``, ``"final"``, ``"grid"``, ``"inj"`` or
        ``"main"``, the keys of :data:`KINDS` plus ``"samples"``.
    """
    t = title.lower()
    if "per-sample" in t:
        return "samples"
    if "klip-fm" in t:
        return "fm"
    if "final-image" in t:
        return "final"
    if "injection-recovery" in t:
        return "grid"
    if "injection-calibrated" in t:
        return "inj"
    return "main"


def run_curves(run_dir: str) -> List[Dict[str, object]]:
    """Collect a run's contrast curves from its per-annulus and final contrast-curve files.

    Parameters
    ----------
    run_dir : str
        The KLIP-TPE run directory.

    Returns
    -------
    curves : list of dict
        One per curve, per-annulus files first, with ``name`` (e.g. ``"Annulus 3: KLIP-FM
        Cross-Check"``), ``kind`` (a key of :data:`KINDS`), ``file`` (its path), ``r`` (separations,
        arcsec, sorted) and ``c`` (contrast at S/N 5) as ndarrays. Per-sample points are left out.
    """
    files = sorted(glob.glob(os.path.join(run_dir, "annulus[0-9][0-9]", "contrast_curve.txt")))
    top = os.path.join(run_dir, "contrast_curve.txt")
    if os.path.exists(top):
        files.append(top)
    out = []
    for path in files:
        parent = os.path.basename(os.path.dirname(path))
        where = f"Annulus {int(parent[7:])}" if parent.startswith("annulus") else "Run"
        for title, rows in read_sections(path):
            kind = classify(title)
            if kind == "samples" or rows.size == 0:
                continue
            o = np.argsort(rows[:, 0], kind="stable")
            out.append({"name": f"{where}: {KINDS[kind]}", "kind": kind, "file": path,
                        "r": rows[o, 0], "c": rows[o, 1]})
    return out


def read_idl(path: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Read the IDL curves at S/N 5 from the table ``idl_curves_compare.py`` writes.

    Parameters
    ----------
    path : str
        The table.

    Returns
    -------
    sep : ndarray, shape (m,)
        The table's separations, in arcsec; m is the number of rows.
    curves : dict of str to ndarray, shape (m,)
        Each curve's contrast at S/N 5 by its column label, NaN where it has no point. The
        ``deepest`` and ``deepest_curve`` columns are left out.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    labels = [k for k in rows[0] if k not in ("separation_arcsec", "deepest", "deepest_curve")]
    sep = np.array([float(r["separation_arcsec"]) for r in rows])
    curves = {k: np.array([float(r[k]) if r[k] not in ("", None) else np.nan for r in rows]) for k in labels}
    return sep, curves


def curve_at(r: np.ndarray, c: np.ndarray, s: float, tol: float) -> Tuple[float, str]:
    """Interpolate a run's curve at one separation.

    Parameters
    ----------
    r : ndarray, shape (m,)
        The curve's separations, sorted, in arcsec; m is the number of points.
    c : ndarray, shape (m,)
        The curve's contrast at S/N 5.
    s : float
        The separation, in arcsec.
    tol : float
        Distance beyond the first or last point within which the end value stands, in arcsec.

    Returns
    -------
    value : float
        The contrast at ``s``, NaN outside the span by more than ``tol``.
    flag : str
        ``"end"`` where the end value stood in, else ``""``.
    """
    if s < r[0] - tol or s > r[-1] + tol:
        return math.nan, ""
    return float(np.interp(s, r, c)), ("" if r[0] <= s <= r[-1] else "end")


def idl_at(sep: np.ndarray, vals: np.ndarray, s: float) -> Tuple[float, str]:
    """Read an IDL curve at one separation, interpolating between its points where it has none.

    Parameters
    ----------
    sep : ndarray, shape (m,)
        The table's separations, sorted, in arcsec; m is the number of rows.
    vals : ndarray, shape (m,)
        The curve's contrast at S/N 5, NaN where it has no point.
    s : float
        The separation, in arcsec.

    Returns
    -------
    value : float
        The contrast at ``s``: the curve's point there, else the linear interpolation between its
        nearest points on either side, else NaN.
    flag : str
        ``"~"`` where interpolated, else ``""``.
    """
    ok = np.isfinite(vals)
    hit = ok & (np.abs(sep - s) < 1e-6)
    if hit.any():
        return float(vals[hit][0]), ""
    lo, hi = ok & (sep < s), ok & (sep > s)
    if not (lo.any() and hi.any()):
        return math.nan, ""
    i, j = np.flatnonzero(lo)[-1], np.flatnonzero(hi)[0]
    return float(vals[i] + (vals[j] - vals[i]) * (s - sep[i]) / (sep[j] - sep[i])), "~"


def compare(curves: Sequence[Dict[str, object]], sep: np.ndarray, idl: Dict[str, np.ndarray],
            seps: Sequence[float], tol: float) -> List[Dict[str, object]]:
    """Compare every run curve with every IDL curve at the given separations.

    Parameters
    ----------
    curves : sequence of dict
        The run's curves, as :func:`run_curves` gives them.
    sep : ndarray, shape (m,)
        The IDL table's separations, in arcsec; m is the number of rows.
    idl : dict of str to ndarray, shape (m,)
        The IDL curves, as :func:`read_idl` gives them.
    seps : sequence of float
        The separations to compare at, in arcsec.
    tol : float
        Distance beyond a run curve's span within which its end value stands, in arcsec.

    Returns
    -------
    rows : list of dict
        One per curve and separation where the run's curve has a value, with ``curve``, ``kind``,
        ``sep``, ``run`` and ``run_flag``; per IDL label ``<label>``, ``<label>_flag`` and
        ``<label>_ratio`` (IDL over run); and ``deepest``, ``deepest_curve``, ``deepest_flag`` and
        ``deepest_ratio`` for the deepest IDL curve with a value there.
    """
    rows = []
    for cv in curves:
        for s in seps:
            v, fl = curve_at(cv["r"], cv["c"], float(s), tol)
            if not np.isfinite(v):
                continue
            row = {"curve": cv["name"], "kind": cv["kind"], "sep": float(s), "run": v, "run_flag": fl}
            best = (math.nan, "", "")
            for lab, vals in idl.items():
                iv, ifl = idl_at(sep, vals, float(s))
                row[lab], row[f"{lab}_flag"] = iv, ifl
                row[f"{lab}_ratio"] = iv / v if np.isfinite(iv) and v > 0 else math.nan
                if np.isfinite(iv) and not (iv >= best[0]):
                    best = (iv, lab, ifl)
            row["deepest"], row["deepest_curve"], row["deepest_flag"] = best
            row["deepest_ratio"] = best[0] / v if np.isfinite(best[0]) and v > 0 else math.nan
            rows.append(row)
    return rows


def _fmt(v: float, kind: str = "c") -> str:
    """Format a contrast (``kind="c"``, 3 significant digits) or a ratio (``"r"``, 2 decimals).

    Parameters
    ----------
    v : float
        The value; NaN prints as ``-``.
    kind : str, default "c"
        ``"c"`` for a contrast, ``"r"`` for a ratio.

    Returns
    -------
    text : str
        The formatted value.
    """
    if not np.isfinite(v):
        return "-"
    return f"{v:.2e}" if kind == "c" else f"{v:.2f}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Print the comparison tables and write the CSV.

    Parameters
    ----------
    argv : sequence of str or None, default None
        The command-line arguments (see the module's usage); None reads ``sys.argv``.

    Returns
    -------
    status : int
        0 on success.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("idl_csv")
    ap.add_argument("--sep", type=float, nargs="+", default=[])
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    sep, idl = read_idl(a.idl_csv)
    seps = sorted(set([round(float(s), 4) for s in sep] + [round(float(s), 4) for s in a.sep]))
    curves = run_curves(a.run_dir)
    rows = compare(curves, sep, idl, seps, a.tol)
    labels = list(idl)
    print(f"Run: {os.path.abspath(a.run_dir)}")
    print(f"IDL: {os.path.abspath(a.idl_csv)} (curves: {', '.join(labels)})")
    print("Ratio = IDL contrast / run contrast (above 1: the run goes deeper); "
          "'end' = the curve's end value within TOL; '~' = IDL interpolated")
    for cv in curves:
        mine = [r for r in rows if r["curve"] == cv["name"]]
        if not mine:
            continue
        print(f"\n{cv['name']} ({os.path.relpath(cv['file'], a.run_dir)}, "
              f"{cv['r'][0]:.3f}-{cv['r'][-1]:.3f}\", {cv['r'].size} points)")
        head = f"{'Sep':>6} {'Run':>13}" + "".join(f" {lab:>16} {'ratio':>6}" for lab in labels) + \
            f" {'Deepest IDL':>24} {'ratio':>6}"
        print(head)
        for r in mine:
            line = f"{r['sep']:6.2f} {_fmt(r['run']):>9}{r['run_flag']:>4}"
            for lab in labels:
                line += f" {_fmt(r[lab]) + r[f'{lab}_flag']:>16} {_fmt(r[f'{lab}_ratio'], 'r'):>6}"
            deep = f"{_fmt(r['deepest'])}{r['deepest_flag']} {r['deepest_curve']}"
            line += f" {deep:>24} {_fmt(r['deepest_ratio'], 'r'):>6}"
            print(line)
    if a.out:
        keys = ["curve", "kind", "sep", "run", "run_flag"] + \
            [k for lab in labels for k in (lab, f"{lab}_flag", f"{lab}_ratio")] + \
            ["deepest", "deepest_curve", "deepest_flag", "deepest_ratio"]
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in keys})
        print(f"\nwrote {a.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
