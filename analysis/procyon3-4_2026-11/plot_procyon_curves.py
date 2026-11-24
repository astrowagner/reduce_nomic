#!/usr/bin/env python3
"""Draw KLIP-TPE final-image contrast curves beside the best IDL curves at S/N 5, in one boxed plot.

Each KLIP-TPE run is drawn as a solid line with filled circles at its final-image grid's
separations (the median over position angle in ``final_grid/curve.txt``). Each IDL data set is
drawn as a dashed line with open squares, in the color of the matching run: at each separation of
its table, the deepest of the given columns (IDL versions computed at S/N 5), so a version
computed at another S/N and scaled can be left out by not naming it. The axes are a closed box with
inward major and minor ticks on all four sides, as in the IDL predecessor's contrast plots, with
the projected separation in au on the top axis.

usage: python analysis/procyon3-4_2026-11/plot_procyon_curves.py OUT_PNG
       --run LABEL COLOR RUN_DIR [--run ...] --idl LABEL COLOR CSV COLUMNS [--idl ...]
       [--title TITLE] [--distance-pc D] [--xlim A B] [--ylim A B] [--pdf OUT_PDF] [--csv OUT_CSV]

Parameters
----------
OUT_PNG : str
    PNG to write.
--run LABEL COLOR RUN_DIR : str, repeatable
    A finished KLIP-TPE run: its legend label, its line color (hex) and its directory, holding
    ``final_grid/curve.txt``.
--idl LABEL COLOR CSV COLUMNS : str, repeatable
    An IDL data set: its legend label, its line color (hex), its table as ``idl_curves_compare.py``
    writes it (``separation_arcsec`` and one column per IDL version) and the comma-separated
    columns to take the deepest of at each separation.
--title : str, default "Procyon at S/N 5: KLIP-TPE Final Images and Best IDL Curves"
    The plot's title.
--distance-pc : float, default 3.51
    The star's distance, in pc, for the projected separation on the top axis.
--xlim : float float, default 0.3 2.5
    Angular separation range, in arcsec. IDL points outside it are not drawn.
--ylim : float float, optional
    Contrast range; by default from the curves drawn.
--pdf : str, optional
    A vector copy of the figure, written without a creation date so reruns are byte-identical.
--csv : str, optional
    The plotted values, one row per curve and separation.

Returns
-------
OUT_PNG : file
    The figure, 200 dpi.
OUT_PDF : file
    The same figure as PDF [--pdf].
OUT_CSV : file
    Columns ``curve``, ``separation_arcsec``, ``contrast`` and ``source`` (the run's
    ``final_grid/curve.txt``, or the IDL column the deepest value came from) [--csv].
"""
import argparse
import csv
import math
import os
import sys
from typing import List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import AutoMinorLocator, MultipleLocator  # noqa: E402

#: line widths (pt) and marker size (pt) of the KLIP-TPE and IDL curves
RUN_LW, IDL_LW, MARKER = 1.8, 1.4, 5.0
#: figure size, in inches, and PNG resolution, in dots per inch
FIGSIZE, DPI = (7.2, 5.0), 200


def read_final_curve(run_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read a finished run's final-image curve.

    Parameters
    ----------
    run_dir : str
        The run directory, holding ``final_grid/curve.txt``.

    Returns
    -------
    sep : ndarray, shape (n,)
        Separations of the final-image grid, in arcsec, where n is the number of grid separations.
    contrast : ndarray, shape (n,)
        Median contrast at S/N 5 over position angle at each separation.
    """
    rows = []
    with open(os.path.join(run_dir, "final_grid", "curve.txt")) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            rows.append((float(parts[0]), float(parts[1])))
    if not rows:
        raise SystemExit(f"no rows in {run_dir}/final_grid/curve.txt")
    a = np.array(rows)
    return a[:, 0], a[:, 1]


def read_idl_best(path: str, columns: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Take, at each separation of an IDL table, the deepest of the given columns.

    Parameters
    ----------
    path : str
        The IDL table (CSV with ``separation_arcsec`` and one column per IDL version; an empty
        cell is no value).
    columns : sequence of str
        The columns to compare. Every name must be in the table.

    Returns
    -------
    sep : ndarray, shape (m,)
        Separations where at least one of the columns has a finite, positive value, in arcsec,
        where m is the number of such separations.
    contrast : ndarray, shape (m,)
        The deepest (smallest) of those values at each separation.
    source : list of str
        The column each value came from.
    """
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in columns if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"{path}: no column(s) {', '.join(missing)}")
        sep, best, src = [], [], []
        for row in reader:
            vals = []
            for c in columns:
                cell = (row.get(c) or "").strip()
                if cell:
                    v = float(cell)
                    if math.isfinite(v) and v > 0:
                        vals.append((v, c))
            if vals:
                v, c = min(vals)
                sep.append(float(row["separation_arcsec"]))
                best.append(v)
                src.append(c)
    return np.array(sep), np.array(best), src


def apply_box_style(ax, top_axis=None) -> None:
    """Draw the axes as a closed box with inward major and minor ticks on every side.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The plot's axes.
    top_axis : matplotlib.axis.SecondaryAxis or None, default None
        A secondary x axis on top. With one, the top side carries its ticks and labels instead of
        mirrored ticks of the bottom axis.

    Returns
    -------
    None
        `ax` (and `top_axis`) are changed in place.
    """
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(1.0)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", direction="in", colors="black", top=top_axis is None, right=True,
                   labelsize=10)
    ax.tick_params(which="major", length=6, width=1.0)
    ax.tick_params(which="minor", length=3, width=0.8)
    ax.grid(False)
    if top_axis is not None:
        top_axis.xaxis.set_minor_locator(AutoMinorLocator())
        top_axis.tick_params(which="both", direction="in", colors="black", labelsize=10)
        top_axis.tick_params(which="major", length=6, width=1.0)
        top_axis.tick_params(which="minor", length=3, width=0.8)


def plot(out_png: str, runs: Sequence[Tuple[str, str, str]], idls: Sequence[Tuple[str, str, str, str]],
         title: str, distance_pc: float, xlim: Tuple[float, float], ylim=None, out_pdf=None,
         out_csv=None) -> List[Tuple[str, float, float, str]]:
    """Draw the curves and write the figure (and the table of plotted values).

    Parameters
    ----------
    out_png : str
        PNG to write.
    runs : sequence of (str, str, str)
        Label, color and directory of each KLIP-TPE run, in legend order.
    idls : sequence of (str, str, str, str)
        Label, color, table path and comma-separated columns of each IDL data set, in legend order.
    title : str
        The plot's title.
    distance_pc : float
        The star's distance, in pc, for the top axis.
    xlim : (float, float)
        Angular separation range, in arcsec.
    ylim : (float, float) or None, default None
        Contrast range; None takes 0.7 times the smallest and 1.4 times the largest value drawn.
    out_pdf : str or None, default None
        PDF copy to write, without a creation date.
    out_csv : str or None, default None
        Table of the plotted values to write.

    Returns
    -------
    rows : list of (str, float, float, str)
        Curve label, separation (arcsec), contrast and source of every plotted point; also written
        to `out_csv` when given.
    """
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 10, "axes.linewidth": 1.0, "legend.frameon": False,
                         "savefig.facecolor": "white", "figure.facecolor": "white"})
    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    rows: List[Tuple[str, float, float, str]] = []
    handles = []
    for label, color, run_dir in runs:
        sep, c = read_final_curve(run_dir)
        keep = (sep >= xlim[0]) & (sep <= xlim[1])
        (h,) = ax.plot(sep[keep], c[keep], "-o", color=color, lw=RUN_LW, ms=MARKER, mfc=color, mec=color,
                       label=label, zorder=3)
        handles.append(h)
        rows += [(label, float(s), float(v), "final_grid/curve.txt") for s, v in zip(sep[keep], c[keep])]
    for label, color, path, cols in idls:
        sep, c, src = read_idl_best(path, [x.strip() for x in cols.split(",") if x.strip()])
        keep = (sep >= xlim[0]) & (sep <= xlim[1])
        (h,) = ax.plot(sep[keep], c[keep], "--s", color=color, lw=IDL_LW, ms=MARKER, mfc="white", mec=color,
                       mew=1.2, label=label, zorder=2)
        handles.append(h)
        rows += [(label, float(s), float(v), w) for s, v, w, k in zip(sep, c, src, keep) if k]
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    vals = np.array([r[2] for r in rows])
    ax.set_ylim(*(ylim if ylim else (0.7 * vals.min(), 1.4 * vals.max())))
    ax.set_xlabel("Angular Separation (arcsec)", fontsize=11)
    ax.set_ylabel("Contrast at S/N 5", fontsize=11)
    top = ax.secondary_xaxis("top", functions=(lambda x: x * distance_pc, lambda au: au / distance_pc))
    top.set_xlabel(f"Projected Separation (au) at {distance_pc:g} pc", fontsize=11)
    apply_box_style(ax, top)
    # major ticks every 0.5 arcsec, minor every 0.1 arcsec
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    fig.suptitle(title, fontsize=12)
    # column-major legend: the runs fill the first column, the IDL curves the second
    ax.legend(handles=handles, loc="upper right", ncol=2, fontsize=9, handlelength=2.6, columnspacing=1.4)
    fig.savefig(out_png, dpi=DPI)
    if out_pdf:
        fig.savefig(out_pdf, metadata={"CreationDate": None})
    plt.close(fig)
    if out_csv:
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["curve", "separation_arcsec", "contrast", "source"])
            for label, s, v, src in rows:
                w.writerow([label, f"{s:.4f}", f"{v:.4e}", src])
    return rows


def main(argv=None) -> None:
    """Parse the command line, draw the figure and print the plotted values.

    Parameters
    ----------
    argv : list of str or None, default None
        Command-line arguments; None reads ``sys.argv[1:]``.

    Returns
    -------
    None
        The figure and tables go to the files named; the plotted values to standard output.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out_png")
    ap.add_argument("--run", nargs=3, action="append", metavar=("LABEL", "COLOR", "RUN_DIR"), default=[])
    ap.add_argument("--idl", nargs=4, action="append", metavar=("LABEL", "COLOR", "CSV", "COLUMNS"), default=[])
    ap.add_argument("--title", default="Procyon at S/N 5: KLIP-TPE Final Images and Best IDL Curves")
    ap.add_argument("--distance-pc", type=float, default=3.51)
    ap.add_argument("--xlim", type=float, nargs=2, default=(0.3, 2.5))
    ap.add_argument("--ylim", type=float, nargs=2, default=None)
    ap.add_argument("--pdf", default=None)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args(argv)
    if not a.run and not a.idl:
        raise SystemExit("give at least one --run or --idl")
    rows = plot(a.out_png, a.run, a.idl, a.title, a.distance_pc, tuple(a.xlim), a.ylim, a.pdf, a.csv)
    for label in dict.fromkeys(r[0] for r in rows):
        pts = ", ".join(f'{s:.2f}" {v:.3g}' for lab, s, v, _ in rows if lab == label)
        print(f"{label}: {pts}")
    print(f"wrote {a.out_png}" + (f", {a.pdf}" if a.pdf else "") + (f", {a.csv}" if a.csv else ""))


if __name__ == "__main__":
    main(sys.argv[1:])
