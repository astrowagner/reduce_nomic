#!/usr/bin/env python3
"""Tabulate every injection-recovery measurement behind the Procyon contrast curves, one row each.

usage: python analysis/procyon3-4_2026-11/pa_measurements_table.py OUT_CSV --archive ARCHIVE
       --run DATA_SET TAG RUN_DIR [--run ...] --idl DATA_SET LABEL PATH [--idl ...]
       --figure CSV --figure-clip CSV [--clip SIGMA]

One row per position: for a KLIP-TPE run, each position of its final-image grid
(``final_grid/curve.json``: 12 position angles per separation), with its status, its contrast at
S/N 5 and every reduction's injected contrast and measured aperture S/N; for an IDL version, each
position angle and separation of its ``*_contrast_output.sav`` (``curves``), or each gray curve of a
``contrastcurve_nomic.pro`` PDF (``../procyon4_2026-11/idl_curve_pdf.py``), with its converged
contrast. Neither pipeline saves the aperture sums themselves (the source aperture and the noise
ring's apertures): KLIP-TPE keeps each reduction's S/N, IDL only the converged contrast.

Each row is flagged by its part in the figures of ``procyon_curves_figure_1sigma.sh``: whether it
enters the plotted point of the unclipped figure and of the clipped one, and whether the clip drops
it from its own set (a run's settled contrasts, or an IDL version's measured ones, at one
separation; ``pa_clip.py``). The figures' tables give the version plotted at each IDL separation.
Every plotted point is then recomputed from the flagged rows (a run's median, an IDL mean) and must
match the figure table's contrast and count, which is checked.

Parameters
----------
OUT_CSV : str
    Table to write.
--archive : str
    The NOMIC archive; ``source_file`` is given relative to it.
--run DATA_SET TAG RUN_DIR : str, repeatable
    A finished KLIP-TPE run: its data set's name as in the figures (e.g. "Procyon_3 + 4"), its tag
    and its directory, holding ``final_grid/curve.json``.
--idl DATA_SET LABEL PATH : str, repeatable
    An IDL version computed at S/N 5: its data set's name, its label (the IDL table's column) and
    its save file or PDF.
--figure : str
    The unclipped figure's table (``procyon_klip_tpe_vs_idl_snr5_1sigma.csv``).
--figure-clip : str
    The clipped figure's table (``procyon_klip_tpe_vs_idl_snr5_1sigma_clip3.csv``).
--clip : float, default 3
    The clipping threshold the clipped figure was made with, in standard deviations.

Returns
-------
OUT_CSV : file
    Columns:

    - ``pipeline``: KLIP-TPE or IDL.
    - ``data_set``: Procyon_3, Procyon_4 or Procyon_3 + 4.
    - ``version``: the run's tag or the IDL version's label.
    - ``source_file``: the file read, relative to ARCHIVE.
    - ``separation_arcsec``: the curve's separation.
    - ``annulus``: the KLIP-TPE annulus the separation is read in; empty for IDL.
    - ``position_index``: the run's position index, the save file's position-angle index, or the
      PDF's gray curve in drawing order (all from 0).
    - ``angle_in_file_deg``: the angle as stored: KLIP-TPE's PA east of north, IDL's angle from +x
      toward +y; empty for a PDF.
    - ``pa_deg``: position angle east of north (north up, east left): KLIP-TPE's angle, or the IDL
      angle minus 90 (modulo 360); empty for a PDF.
    - ``status``: converged, accepted or blocked (KLIP-TPE); measured, failed (0) or not recovered
      (NaN) (save file); measured or unfinished (PDF, the gray curves past the complete position
      angles).
    - ``contrast_snr5``: the contrast at S/N 5; empty where none (for a PDF's unfinished curve, the
      value drawn).
    - ``n_reductions``, ``last_snr`` and ``reductions``: KLIP-TPE's number of reductions, the S/N of
      the last and every reduction as ``contrast:S/N`` separated by ``;``; empty for IDL.
    - ``note``: KLIP-TPE's reason for a blocked position, or the PDF's lack of angles.
    - ``clipped``: 1 if the clip drops the value from its set, 0 if kept, empty if in no set.
    - ``in_figure`` and ``in_figure_clip``: 1 if the value enters the plotted point of the
      unclipped or the clipped figure, else 0.
stdout : text
    Rows written per pipeline and data set, and the check of every plotted point.
"""
import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.io import readsav

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "procyon4_2026-11"))
from idl_curve_pdf import read_curves  # noqa: E402
from pa_clip import sigma_clip_mask  # noqa: E402

#: statuses of the final-grid positions whose contrasts make a run's curve (klip_tpe.contrast_grid)
SETTLED = ("converged", "accepted")
#: columns of the table, in order
FIELDS = ["pipeline", "data_set", "version", "source_file", "separation_arcsec", "annulus", "position_index",
          "angle_in_file_deg", "pa_deg", "status", "contrast_snr5", "n_reductions", "last_snr", "reductions", "note",
          "clipped", "in_figure", "in_figure_clip"]
#: largest relative difference allowed between a point recomputed from the rows and the figure's
#: table: 2e-4 where the figure's value is computed from the same values (5 significant digits in
#: its table), 1e-3 for the unclipped IDL points, which are the IDL tables' values (4 significant
#: digits; the Dec 2022 PDF's black line, within 0.04 % of its mean)
POINT_TOL, IDL_TABLE_TOL = 2e-4, 1e-3


def read_sav(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read an IDL contrast save file.

    Parameters
    ----------
    path : str
        A ``*_contrast_output.sav`` of ``contrastcurve_nomic.pro``.

    Returns
    -------
    curves : ndarray, shape (n_pa, n_sep)
        Converged contrast of each position angle and separation, where n_pa and n_sep are the
        numbers of position angles and separations (NaN: not recovered; 0: failed).
    pas : ndarray, shape (n_pa,)
        The position angles, in degrees, IDL's angle from +x toward +y.
    seps : ndarray, shape (n_sep,)
        The separations, in arcsec.
    """
    s = readsav(path, python_dict=True)
    c, pa, sep = np.asarray(s["curves"], float), np.asarray(s["pas"], float), np.asarray(s["separations"], float)
    if c.shape != (pa.size, sep.size):
        c = c.T
    if c.shape != (pa.size, sep.size):
        raise SystemExit(f"{path}: curves {c.shape} does not match {pa.size} position angles x {sep.size} separations")
    return c, pa, sep


def _f(x: Optional[float], fmt: str = ".4e") -> str:
    """Format a number for the table, empty for None or NaN."""
    return "" if x is None or not math.isfinite(x) else format(x, fmt)


def run_rows(data_set: str, tag: str, run_dir: str, archive: str) -> List[dict]:
    """Return one row per position of a run's final-image grid.

    Parameters
    ----------
    data_set : str
        The data set's name, as in the figures.
    tag : str
        The run's tag.
    run_dir : str
        The run directory, holding ``final_grid/curve.json``.
    archive : str
        The archive the source file is given relative to.

    Returns
    -------
    rows : list of dict
        The rows (see the module's Returns), without the flags, plus ``_value`` (the contrast, or
        NaN) and ``_set`` (the key of the clip's set, or None) for internal use.
    """
    path = os.path.join(run_dir, "final_grid", "curve.json")
    with open(path) as f:
        js = json.load(f)
    r_as, ann = js["r_as"], js.get("annulus") or [None] * len(js["r_as"])
    rows = []
    for p in sorted(js["positions"], key=lambda q: (q["sep_index"], q["index"])):
        i = p["sep_index"]
        trials = p.get("trials") or []
        c = p.get("contrast")
        st = p["status"]
        rows.append({"pipeline": "KLIP-TPE", "data_set": data_set, "version": tag,
                     "source_file": os.path.relpath(path, archive), "separation_arcsec": f"{r_as[i]:.4f}",
                     "annulus": "" if ann[i] is None else str(ann[i]), "position_index": str(p["index"]),
                     "angle_in_file_deg": _f(p["theta"], ".4f"), "pa_deg": _f(p["theta"] % 360.0, ".4f"),
                     "status": st, "contrast_snr5": _f(c), "n_reductions": str(len(trials)),
                     "last_snr": _f(trials[-1][1], ".4f") if trials else "",
                     "reductions": ";".join(f"{t[0]:.4e}:{t[1]:.4f}" for t in trials), "note": p.get("why") or "",
                     "_value": float(c) if c is not None else float("nan"),
                     "_set": (data_set, "KLIP-TPE", tag, round(r_as[i], 4)) if st in SETTLED and c is not None
                     else None})
    return rows


def idl_rows(data_set: str, label: str, path: str, archive: str) -> List[dict]:
    """Return one row per position angle and separation of an IDL version.

    Parameters
    ----------
    data_set : str
        The data set's name, as in the figures.
    label : str
        The version's label (the IDL table's column).
    path : str
        Its save file or contrast-curve PDF. A PDF drawn before its last test (the title's
        "Test k / N") counts only its first (k - 1) // n_sep gray curves, as ``idl_pa_scatter.py``.
    archive : str
        The archive the source file is given relative to.

    Returns
    -------
    rows : list of dict
        The rows (see the module's Returns), without the flags, plus ``_value`` and ``_set`` for
        internal use.
    """
    rel = os.path.relpath(path, archive)
    rows = []
    base = {"pipeline": "IDL", "data_set": data_set, "version": label, "source_file": rel, "annulus": "",
            "n_reductions": "", "last_snr": "", "reductions": ""}
    if path.lower().endswith(".sav"):
        c, pas, seps = read_sav(path)
        for j, s in enumerate(seps):
            for i, pa in enumerate(pas):
                v = float(c[i, j])
                ok = math.isfinite(v) and v > 0
                st = "measured" if ok else ("not recovered (NaN)" if not math.isfinite(v) else "failed (0)")
                rows.append(dict(base, separation_arcsec=f"{s:.2f}", position_index=str(i),
                                 angle_in_file_deg=_f(pa, ".4f"), pa_deg=_f((pa - 90.0) % 360.0, ".4f"), status=st,
                                 contrast_snr5=_f(v) if ok else "", note="",
                                 _value=v if ok else float("nan"), _set=(data_set, "IDL", label, round(float(s), 2))
                                 if ok else None))
        return rows
    d = read_curves(path)
    gray = d["pa"]
    n_sep = len(gray[0])
    m = re.search(r"Test\s+(\d+)\s*/\s*(\d+)", d.get("title") or "")
    n_done = min(len(gray), (int(m.group(1)) - 1) // n_sep) if m else len(gray)
    for j in range(n_sep):
        for i, g in enumerate(gray):
            s, v = float(g[j][0]), float(g[j][1])
            done = i < n_done
            ok = done and math.isfinite(v) and v > 0
            rows.append(dict(base, separation_arcsec=f"{s:.2f}", position_index=str(i), angle_in_file_deg="",
                             pa_deg="", status="measured" if done else "unfinished", contrast_snr5=_f(v),
                             note="gray curve in drawing order; the PDF has no angles",
                             _value=v if ok else float("nan"),
                             _set=(data_set, "IDL", label, round(s, 2)) if ok else None))
    return rows


def read_figure(path: str) -> Dict[Tuple[str, str, float], dict]:
    """Read a figure's table of plotted points.

    Parameters
    ----------
    path : str
        A table written by ``plot_procyon_curves.py --bands`` (``curve``, ``separation_arcsec``,
        ``contrast``, ``n``, ``n_clipped``, ``source``).

    Returns
    -------
    points : dict
        ``(pipeline, data_set, separation)`` (separation rounded to 2 decimals for IDL, 4 for
        KLIP-TPE) to the point's row.
    """
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            pipeline, data_set = r["curve"].split(" ", 1)
            nd = 4 if pipeline == "KLIP-TPE" else 2
            out[(pipeline, data_set, round(float(r["separation_arcsec"]), nd))] = r
    return out


def flag_and_check(rows: List[dict], figure: Dict, figure_clip: Dict, clip: float) -> List[str]:
    """Set the clip and figure flags of every row and check each plotted point against them.

    Parameters
    ----------
    rows : list of dict
        All rows, from `run_rows` and `idl_rows`; changed in place.
    figure : dict
        The unclipped figure's points (`read_figure`).
    figure_clip : dict
        The clipped figure's points.
    clip : float
        The clipping threshold, in standard deviations.

    Returns
    -------
    report : list of str
        One line per figure: the points checked.

    Raises
    ------
    SystemExit
        If a plotted point has no rows, or its value or count recomputed from the flagged rows
        differs from the figure's table.
    """
    sets = defaultdict(list)
    for r in rows:
        if r["_set"] is not None:
            sets[r["_set"]].append(r)
    for members in sets.values():
        keep = sigma_clip_mask(np.array([m["_value"] for m in members]), clip)
        for m, k in zip(members, keep):
            m["clipped"] = "0" if k else "1"
    for r in rows:
        r.setdefault("clipped", "")
    report = []
    for name, fig, key in (("unclipped", figure, "in_figure"), ("clipped", figure_clip, "in_figure_clip")):
        for r in rows:
            r[key] = "0"
        for (pipeline, data_set, sep), pt in fig.items():
            if pipeline == "KLIP-TPE":
                version = [v for (d, p, v, s) in sets if d == data_set and p == pipeline and s == sep]
            else:
                version = [pt["source"]]
            members = [m for v in version for m in sets.get((data_set, pipeline, v, sep), [])]
            if not members:
                raise SystemExit(f"{name} figure: no rows for {pipeline} {data_set} at {sep}\"")
            used = [m for m in members if key == "in_figure" or m["clipped"] == "0"]
            vals = np.array([m["_value"] for m in used])
            val = float(np.median(vals)) if pipeline == "KLIP-TPE" else float(vals.mean())
            tol = IDL_TABLE_TOL if (pipeline == "IDL" and key == "in_figure") else POINT_TOL
            if abs(val / float(pt["contrast"]) - 1.0) > tol or len(used) != int(pt["n"]):
                raise SystemExit(f"{name} figure: {pipeline} {data_set} at {sep}\" is {pt['contrast']} [{pt['n']}], "
                                 f"the rows give {val:.4e} [{len(used)}]")
            if key == "in_figure_clip" and len(members) - len(used) != int(pt["n_clipped"]):
                raise SystemExit(f"clipped figure: {pipeline} {data_set} at {sep}\": {pt['n_clipped']} clipped, "
                                 f"the rows give {len(members) - len(used)}")
            for m in used:
                m[key] = "1"
        tol = f"{POINT_TOL:.0e}" + (f", IDL {IDL_TABLE_TOL:.0e}" if key == "in_figure" else "")
        report.append(f"{name} figure: all {len(fig)} plotted points reproduced from the rows "
                      f"(value within {tol}, count exact)")
    return report


def main(argv=None) -> int:
    """Build, check and write the table of measurements.

    Parameters
    ----------
    argv : list of str or None, default None
        Command-line arguments; None reads ``sys.argv[1:]``.

    Returns
    -------
    status : int
        0.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out_csv")
    ap.add_argument("--archive", required=True)
    ap.add_argument("--run", nargs=3, action="append", metavar=("DATA_SET", "TAG", "RUN_DIR"), default=[])
    ap.add_argument("--idl", nargs=3, action="append", metavar=("DATA_SET", "LABEL", "PATH"), default=[])
    ap.add_argument("--figure", required=True)
    ap.add_argument("--figure-clip", required=True)
    ap.add_argument("--clip", type=float, default=3.0)
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    rows: List[dict] = []
    for data_set, tag, run_dir in a.run:
        rows += run_rows(data_set, tag, run_dir, a.archive)
    for data_set, label, path in a.idl:
        rows += idl_rows(data_set, label, path, a.archive)
    report = flag_and_check(rows, read_figure(a.figure), read_figure(a.figure_clip), a.clip)
    with open(a.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    counts = defaultdict(int)
    for r in rows:
        counts[(r["pipeline"], r["data_set"], r["version"])] += 1
    for (p, d, v), n in counts.items():
        mine = [r for r in rows if (r["pipeline"], r["data_set"], r["version"]) == (p, d, v)]
        flagged = sum(r["in_figure"] == "1" for r in mine)
        flagged_clip = sum(r["in_figure_clip"] == "1" for r in mine)
        clipped = sum(r["clipped"] == "1" for r in mine)
        print(f"{p} {d} {v}: {n} rows; in the unclipped figure {flagged}, in the clipped one {flagged_clip}; "
              f"clipped at {a.clip:g} sigma {clipped}")
    for line in report:
        print(line)
    print(f"wrote {a.out_csv} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
