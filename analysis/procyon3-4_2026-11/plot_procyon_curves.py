#!/usr/bin/env python3
"""Draw KLIP-TPE final-image contrast curves beside the best IDL curves at S/N 5, in one boxed plot.

Each KLIP-TPE run is drawn as a solid line with filled circles at its final-image grid's
separations (the median over position angle in ``final_grid/curve.txt``). Each IDL data set is
drawn as a dashed line with open squares, in the color of the matching run: at each separation of
its table, the deepest of the given columns (IDL versions computed at S/N 5), so a version
computed at another S/N and scaled can be left out by not naming it. The axes are a closed box with
inward major and minor ticks on all four sides, as in the IDL predecessor's contrast plots, with
the projected separation in au on the top axis.

With ``--bands``, each curve gets a band of ±1 sample standard deviation (ddof 1) over position
angle, centered on the plotted value: for a run, of its settled (converged or accepted) positions'
contrasts in ``final_grid/curve.json``, whose median the curve is; for an IDL point, of the
position angles that the IDL version it came from averages, read from the data set's scatter table
(``idl_pa_scatter.py``). KLIP-TPE bands are filled and IDL bands hatched, in the curve's color.
Where the value minus the standard deviation is not positive, the band's lower edge runs off the
bottom of the plot (log axis); the CSV keeps the true edge.

With ``--clip SIGMA``, the contrasts of each separation are sigma-clipped over position angle
(``pa_clip.py``: iterative, median-centered) before the curves and bands are taken, for every
curve alike: a run's curve is then the median of its settled contrasts that survive the clip, and
an IDL data set's curve at each separation the deepest of its versions' clipped means, from scatter
tables written with the same ``--clip``.

usage: python analysis/procyon3-4_2026-11/plot_procyon_curves.py OUT_PNG
       --run LABEL COLOR RUN_DIR [--run ...] --idl LABEL COLOR CSV COLUMNS [--idl ...]
       [--bands] [--clip SIGMA] [--idl-scatter LABEL CSV [--idl-scatter ...]]
       [--title TITLE] [--distance-pc D] [--xlim A B] [--ylim A B] [--pdf OUT_PDF] [--csv OUT_CSV]

Parameters
----------
OUT_PNG : str
    PNG to write.
--run LABEL COLOR RUN_DIR : str, repeatable
    A finished KLIP-TPE run: its legend label, its line color (hex) and its directory, holding
    ``final_grid/curve.txt`` (and ``final_grid/curve.json`` for ``--bands`` and ``--clip``).
--idl LABEL COLOR CSV COLUMNS : str, repeatable
    An IDL data set: its legend label, its line color (hex), its table as ``idl_curves_compare.py``
    writes it (``separation_arcsec`` and one column per IDL version) and the comma-separated
    columns to take the deepest of at each separation.
--bands : flag
    Draw the ±1σ bands. Every ``--idl`` then needs an ``--idl-scatter`` of the same label.
--clip : float, optional
    Clipping threshold over position angle, in standard deviations (ddof 0); none by default.
    Every ``--idl`` then needs an ``--idl-scatter`` written with the same ``--clip``.
--idl-scatter LABEL CSV : str, repeatable
    The scatter table of the IDL data set labeled LABEL, as ``idl_pa_scatter.py`` writes it
    (``<column>_mean_all``, ``<column>_mean``, ``<column>_sd``, ``<column>_n`` and
    ``<column>_n_clipped`` per IDL version, and ``clip_sigma``). Its unclipped mean must match the
    IDL table's value within 0.5 %.
--title : str, default "Procyon at S/N 5: KLIP-TPE Final Images and Best IDL Curves"
    The plot's title.
--distance-pc : float, default 3.51
    The star's distance, in pc, for the projected separation on the top axis.
--xlim : float float, default 0.3 2.5
    Angular separation range, in arcsec. IDL points outside it are not drawn.
--ylim : float float, optional
    Contrast range; by default from the curves (and band edges above zero) drawn.
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
    ``final_grid/curve.txt``, or the IDL column the deepest value came from) [--csv]; with
    ``--bands`` or ``--clip`` also ``sd`` (the standard deviation over position angle), ``n``
    (the position angles it is over), ``n_clipped`` (those clipped), ``band_low`` and
    ``band_high`` (contrast minus and plus ``sd``), before ``source``.
"""
import argparse
import csv
import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import AutoMinorLocator, MultipleLocator  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pa_clip import sigma_clip_mask  # noqa: E402

#: line widths (pt) and marker size (pt) of the KLIP-TPE and IDL curves
RUN_LW, IDL_LW, MARKER = 1.8, 1.4, 5.0
#: figure size, in inches, and PNG resolution, in dots per inch
FIGSIZE, DPI = (7.2, 5.0), 200
#: opacity of the KLIP-TPE runs' filled ±1σ bands and of the IDL curves' hatched ones
RUN_BAND_ALPHA, IDL_BAND_ALPHA = 0.18, 0.45
#: hatch pattern of the IDL bands and its line width (pt)
IDL_BAND_HATCH, HATCH_LW = "////", 0.7
#: statuses of the final-grid positions whose contrasts make the curve (klip_tpe.contrast_grid)
SETTLED = ("converged", "accepted")
#: largest relative difference allowed between a scatter table's mean and the plotted IDL value
MEAN_TOL = 0.005


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


def read_final_positions(run_dir: str, sep: np.ndarray, contrast: np.ndarray, clip: Optional[float] = None
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a run's curve, its standard deviation over position angle and the counts, after an optional clip.

    The contrasts are those of the settled (converged or accepted) positions in
    ``final_grid/curve.json``, whose median is the curve.

    Parameters
    ----------
    run_dir : str
        The run directory, holding ``final_grid/curve.json``.
    sep : ndarray, shape (n,)
        The curve's separations from ``final_grid/curve.txt``, in arcsec, where n is the number of
        grid separations.
    contrast : ndarray, shape (n,)
        The curve's medians from ``final_grid/curve.txt``.
    clip : float or None, default None
        Clipping threshold over position angle, in standard deviations (``pa_clip.sigma_clip_mask``);
        None keeps every settled contrast.

    Returns
    -------
    curve : ndarray, shape (n,)
        Median of the settled contrasts kept at each separation; `contrast` itself without a clip.
    sd : ndarray, shape (n,)
        Sample standard deviation (ddof 1) of the settled contrasts kept; NaN where fewer than two.
    count : ndarray of int, shape (n,)
        Number of settled contrasts kept at each separation.
    n_clipped : ndarray of int, shape (n,)
        Number of settled contrasts clipped at each separation.

    Raises
    ------
    SystemExit
        If ``curve.json`` does not match ``curve.txt``: other separations, or a median of the
        settled contrasts that differs from the curve by more than 0.01 %.
    """
    path = os.path.join(run_dir, "final_grid", "curve.json")
    with open(path) as f:
        js = json.load(f)
    r = np.asarray(js["r_as"], float)
    if r.shape != sep.shape or np.max(np.abs(r - sep)) > 1e-3:
        raise SystemExit(f"{path}: separations do not match final_grid/curve.txt")
    curve = np.array(contrast, float)
    sd, count, n_clipped = np.full(sep.size, np.nan), np.zeros(sep.size, int), np.zeros(sep.size, int)
    for i in range(sep.size):
        v = np.array([p["contrast"] for p in js["positions"] if p["sep_index"] == i and p["status"] in SETTLED],
                     float)
        if v.size and abs(np.median(v) / contrast[i] - 1.0) > 1e-4:
            raise SystemExit(f"{path}: median of the settled contrasts at {sep[i]:.2f}\" is {np.median(v):.4e}, "
                             f"not the curve's {contrast[i]:.4e}")
        if clip and v.size:
            keep = sigma_clip_mask(v, clip)
            n_clipped[i] = int(v.size - keep.sum())
            v = v[keep]
            curve[i] = np.median(v)
        count[i] = v.size
        if v.size > 1:
            sd[i] = v.std(ddof=1)
    return curve, sd, count, n_clipped


def read_idl_scatter(path: str, sep: np.ndarray, contrast: np.ndarray, source: Sequence[str],
                     columns: Sequence[str], clip: Optional[float] = None
                     ) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray, np.ndarray]:
    """Return each plotted IDL point's value, version, standard deviation over position angle and counts.

    Parameters
    ----------
    path : str
        The data set's scatter table, as ``idl_pa_scatter.py`` writes it.
    sep : ndarray, shape (m,)
        Separations of the plotted points, in arcsec, where m is their number.
    contrast : ndarray, shape (m,)
        The IDL table's values at those separations (the deepest of `columns`).
    source : sequence of str
        The IDL version (table column) each value came from.
    columns : sequence of str
        The IDL versions to take the deepest of.
    clip : float or None, default None
        The clipping threshold the table must have been written with (its ``clip_sigma``); None
        for a table written without one.

    Returns
    -------
    value : ndarray, shape (m,)
        The point's value: `contrast` without a clip, else the deepest of the versions' clipped
        means.
    version : list of str
        The version each value came from.
    sd : ndarray, shape (m,)
        Sample standard deviation over position angle of that version's values kept; NaN where one
        is kept.
    count : ndarray of int, shape (m,)
        Number of that version's values kept.
    n_clipped : ndarray of int, shape (m,)
        Number of that version's values clipped.

    Raises
    ------
    SystemExit
        If the table was written with another clip, lacks a point's version or separation, or its
        unclipped mean there differs from the IDL table's value by more than `MEAN_TOL`, so that
        the scatter would not be of the values the IDL curve averages.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    clips = {(r.get("clip_sigma") or "").strip() for r in rows}
    want = "" if clip is None else f"{clip:g}"
    if clips != {want}:
        raise SystemExit(f"{path}: written with clip_sigma {sorted(clips)}, not {want or 'none'}")
    by_sep: Dict[float, dict] = {round(float(r["separation_arcsec"]), 2): r for r in rows}

    def _get(row: dict, col: str, field: str) -> str:
        """Return a cell of the scatter table, empty when missing."""
        return (row.get(f"{col}_{field}") or "").strip()

    value, version = np.array(contrast, float), list(source)
    sd, count, n_clipped = np.full(sep.size, np.nan), np.zeros(sep.size, int), np.zeros(sep.size, int)
    for i, (s, c, src) in enumerate(zip(sep, contrast, source)):
        row = by_sep.get(round(float(s), 2))
        if row is None or not _get(row, src, "mean_all"):
            raise SystemExit(f"{path}: no {src} at {s:.2f}\"")
        mean_all = float(_get(row, src, "mean_all"))
        if abs(mean_all / c - 1.0) > MEAN_TOL:
            raise SystemExit(f"{path}: {src} mean at {s:.2f}\" is {mean_all:.4e}, not the IDL table's {c:.4e}")
        if clip:
            have = [(float(_get(row, col, "mean")), col) for col in columns if _get(row, col, "mean")]
            value[i], version[i] = min(have)
        col = version[i]
        count[i], n_clipped[i] = int(_get(row, col, "n")), int(_get(row, col, "n_clipped") or 0)
        if _get(row, col, "sd"):
            sd[i] = float(_get(row, col, "sd"))
    return value, version, sd, count, n_clipped


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
         out_csv=None, bands: bool = False, idl_scatter: Optional[Dict[str, str]] = None,
         clip: Optional[float] = None) -> List[Tuple[str, float, float, str, float, int, int]]:
    """Draw the curves (and their ±1σ bands) and write the figure (and the table of plotted values).

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
        Contrast range; None takes 0.7 times the smallest and 1.4 times the largest value drawn,
        band edges above zero included.
    out_pdf : str or None, default None
        PDF copy to write, without a creation date.
    out_csv : str or None, default None
        Table of the plotted values to write.
    bands : bool, default False
        Draw a band of ±1 standard deviation over position angle around each curve.
    idl_scatter : dict of str to str or None, default None
        IDL data set label to its scatter table (``idl_pa_scatter.py``); needed for every IDL data
        set when `bands` is True or `clip` is given.
    clip : float or None, default None
        Clipping threshold over position angle, in standard deviations, for every curve; None
        keeps every value.

    Returns
    -------
    rows : list of (str, float, float, str, float, int, int)
        Curve label, separation (arcsec), contrast, source, standard deviation over position angle,
        the values it is over and the values clipped, for every plotted point (the last three NaN,
        0 and 0 without `bands` or `clip`); also written to `out_csv` when given.
    """
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 10, "axes.linewidth": 1.0, "legend.frameon": False,
                         "savefig.facecolor": "white", "figure.facecolor": "white"})
    if bands:
        plt.rcParams["hatch.linewidth"] = HATCH_LW
    scatter = bands or bool(clip)
    # each curve: kind, label, color, separations, contrasts, sources, standard deviations, counts, clipped
    curves = []
    for label, color, run_dir in runs:
        sep, c = read_final_curve(run_dir)
        if scatter:
            c, sd, n, nc = read_final_positions(run_dir, sep, c, clip)
        else:
            sd, n, nc = np.full(sep.size, np.nan), np.zeros(sep.size, int), np.zeros(sep.size, int)
        keep = (sep >= xlim[0]) & (sep <= xlim[1])
        curves.append(("run", label, color, sep[keep], c[keep], ["final_grid/curve.txt"] * int(keep.sum()),
                       sd[keep], n[keep], nc[keep]))
    for label, color, path, cols in idls:
        columns = [x.strip() for x in cols.split(",") if x.strip()]
        sep, c, src = read_idl_best(path, columns)
        keep = (sep >= xlim[0]) & (sep <= xlim[1])
        src = [w for w, k in zip(src, keep) if k]
        sep, c = sep[keep], c[keep]
        if scatter:
            if not idl_scatter or label not in idl_scatter:
                raise SystemExit(f"--bands or --clip: no --idl-scatter for {label!r}")
            c, src, sd, n, nc = read_idl_scatter(idl_scatter[label], sep, c, src, columns, clip)
        else:
            sd, n, nc = np.full(sep.size, np.nan), np.zeros(sep.size, int), np.zeros(sep.size, int)
        curves.append(("idl", label, color, sep, c, src, sd, n, nc))
    vals = [float(v) for cu in curves for v in cu[4]]
    if bands:
        for _, _, _, _, c, _, sd, _, _ in curves:
            ok = np.isfinite(sd)
            vals += [float(v) for v in (c + sd)[ok]] + [float(v) for v in (c - sd)[ok] if v > 0]
    vals = np.array(vals)
    ylim = tuple(ylim) if ylim else (0.7 * vals.min(), 1.4 * vals.max())
    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    rows: List[Tuple[str, float, float, str, float, int, int]] = []
    handles = []
    for kind, label, color, sep, c, src, sd, n, nc in curves:
        if kind == "run":
            (h,) = ax.plot(sep, c, "-o", color=color, lw=RUN_LW, ms=MARKER, mfc=color, mec=color, label=label,
                           zorder=3)
        else:
            (h,) = ax.plot(sep, c, "--s", color=color, lw=IDL_LW, ms=MARKER, mfc="white", mec=color, mew=1.2,
                           label=label, zorder=2)
        if bands:
            # a lower edge at or below zero runs off the bottom of the log axis
            lo = np.where(c - sd > 0, c - sd, ylim[0] / 100.0)
            if kind == "run":
                band = dict(facecolor=color, alpha=RUN_BAND_ALPHA, lw=0)
            else:
                band = dict(facecolor="none", edgecolor=color, hatch=IDL_BAND_HATCH, alpha=IDL_BAND_ALPHA, lw=0)
            ax.fill_between(sep, lo, c + sd, where=np.isfinite(sd), zorder=1, **band)
            handles.append((Patch(**band), h))
        else:
            handles.append(h)
        rows += [(label, float(s), float(v), w, float(e), int(k), int(q))
                 for s, v, w, e, k, q in zip(sep, c, src, sd, n, nc)]
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
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
    legend_kw = dict(loc="upper right", ncol=2, fontsize=9, handlelength=2.6, columnspacing=1.4)
    if bands:
        note = "Bands: ±1σ over Position Angle" + (f", {clip:g}σ-Clipped" if clip else "")
    else:
        note = f"{clip:g}σ-Clipped over Position Angle" if clip else ""
    if bands:
        # each entry shows its band behind its line
        ax.legend(handles=handles, labels=[cu[1] for cu in curves], handler_map={tuple: HandlerTuple(ndivide=1)},
                  title=note, title_fontsize=9, **legend_kw)
    elif clip:
        ax.legend(handles=handles, title=note, title_fontsize=9, **legend_kw)
    else:
        ax.legend(handles=handles, **legend_kw)
    fig.savefig(out_png, dpi=DPI)
    if out_pdf:
        fig.savefig(out_pdf, metadata={"CreationDate": None})
    plt.close(fig)
    if out_csv:
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            if scatter:
                def _e(x: float) -> str:
                    """Format a value for the table, empty for NaN."""
                    return f"{x:.4e}" if math.isfinite(x) else ""

                w.writerow(["curve", "separation_arcsec", "contrast", "sd", "n", "n_clipped", "band_low", "band_high",
                            "source"])
                for label, s, v, src, sd, n, nc in rows:
                    w.writerow([label, f"{s:.4f}", f"{v:.4e}", _e(sd), n, nc, _e(v - sd), _e(v + sd), src])
            else:
                w.writerow(["curve", "separation_arcsec", "contrast", "source"])
                for label, s, v, src, _, _, _ in rows:
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
    ap.add_argument("--bands", action="store_true")
    ap.add_argument("--clip", type=float, default=None, metavar="SIGMA")
    ap.add_argument("--idl-scatter", nargs=2, action="append", metavar=("LABEL", "CSV"), default=[])
    ap.add_argument("--title", default="Procyon at S/N 5: KLIP-TPE Final Images and Best IDL Curves")
    ap.add_argument("--distance-pc", type=float, default=3.51)
    ap.add_argument("--xlim", type=float, nargs=2, default=(0.3, 2.5))
    ap.add_argument("--ylim", type=float, nargs=2, default=None)
    ap.add_argument("--pdf", default=None)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args(argv)
    if not a.run and not a.idl:
        raise SystemExit("give at least one --run or --idl")
    if a.clip is not None and not a.clip > 0:
        raise SystemExit("--clip must be positive")
    if a.idl_scatter and not (a.bands or a.clip):
        raise SystemExit("--idl-scatter is for --bands or --clip")
    scatter = dict(a.idl_scatter)
    unknown = sorted(set(scatter) - {lab for lab, _, _, _ in a.idl})
    if unknown:
        raise SystemExit(f"--idl-scatter for no --idl: {', '.join(unknown)}")
    rows = plot(a.out_png, a.run, a.idl, a.title, a.distance_pc, tuple(a.xlim), a.ylim, a.pdf, a.csv,
                bands=a.bands, idl_scatter=scatter, clip=a.clip)
    for label in dict.fromkeys(r[0] for r in rows):
        if a.bands or a.clip:
            pts = ", ".join(f'{s:.2f}" {v:.3g} ± {e:.2g} [{k}' + (f" -{q}]" if q else "]")
                            for lab, s, v, _, e, k, q in rows if lab == label)
        else:
            pts = ", ".join(f'{s:.2f}" {v:.3g}' for lab, s, v, _, _, _, _ in rows if lab == label)
        print(f"{label}: {pts}")
    if a.clip:
        print(f"clipped over position angle at {a.clip:g} sigma (median-centered, iterated); "
              "[n -k]: n values kept, k clipped")
    if a.bands:
        low = [f'{lab} at {s:.2f}" ({v:.3g} - {e:.3g})' for lab, s, v, _, e, _, _ in rows
               if math.isfinite(e) and v - e <= 0]
        print("band below zero (lower edge off the plot): " + ("; ".join(low) if low else "none"))
    print(f"wrote {a.out_png}" + (f", {a.pdf}" if a.pdf else "") + (f", {a.csv}" if a.csv else ""))


if __name__ == "__main__":
    main(sys.argv[1:])
