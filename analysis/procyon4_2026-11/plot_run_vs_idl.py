"""Plot a KLIP-TPE run's final contrast curve against Kevin's IDL curves at S/N 5.

usage: python analysis/procyon4_2026-11/plot_run_vs_idl.py RUN_DIR IDL_CSV OUT [--title TITLE]
       [--xmin XMIN] [--xmax XMAX] [--edges E [E ...]]
       [--run-label LABEL] [--run-short SHORT]
       [--compare RUN_DIR2 [--compare-label LABEL] [--compare-short SHORT]]

The run's curve is the final-image grid's (``final_grid/curve.txt``): the median over position
angle at each separation, drawn with its 16th to 84th percentile band. The IDL curves are the
columns of the table ``idl_curves_compare.py`` writes, each at the points it has, with its label
from :data:`LABELS`. A second run's final curve can be drawn beside the first (``--compare``), solid
with open markers and in the color after the IDL table's curves. Every curve is labeled at its last
point inside the plot as well as in the legend, and the IDL curves are dashed with their own
markers, so no curve is told by color alone.

Parameters
----------
RUN_DIR : str
    The finished KLIP-TPE run directory.
IDL_CSV : str
    The IDL curves at S/N 5: ``separation_arcsec``, one column per curve, ``deepest`` and
    ``deepest_curve``.
OUT : str
    Figure to write; its extension picks the format (``.png`` at 200 dpi, ``.pdf``).
--title : str, default "Contrast at S/N 5: KLIP-TPE Final Image and IDL"
    Title of the plot.
--xmin : float, default 0.3
    Smallest separation shown, in arcsec.
--xmax : float, default 2.6
    Largest separation shown, in arcsec.
--edges : float, optional
    Separations of the run's annulus edges to mark, in arcsec.
--run-label : str, default "KLIP-TPE final image (median, S/N 5)"
    Legend entry of the run's curve.
--run-short : str, default "KLIP-TPE"
    Label written beside the run's last point.
--compare : str, optional
    Another finished run whose final-image curve (median only) is drawn for comparison.
--compare-label : str, default "KLIP-TPE comparison run (median, S/N 5)"
    Legend entry of that curve.
--compare-short : str, default "Comparison"
    Label written beside that curve's last point.

Returns
-------
OUT : file
    The figure.
stdout : text
    The points drawn per curve.
"""
import argparse
import csv
import os
import sys
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

#: Display labels of the IDL table's columns; a column not listed keeps its name.
LABELS = {"dec2025_sav": "IDL Dec 2025 (S/N 5)", "dec2022_5sig": "IDL Dec 2022 (S/N 5)",
          "jan2022_norepl": "IDL Jan 2022 (S/N 3, scaled to 5)", "jan2024_sav": "IDL Jan 2024 (S/N 5)",
          "jan2025_sav": "IDL Jan 2025 (S/N 5)"}

#: Light-mode chart colors: the run (validated categorical slot 1), then the IDL curves in table
#: order (slots 2-5), the band, ink and chrome.
COLORS = {"run": "#2a78d6", "idl": ["#eb6834", "#1baf7a", "#eda100", "#e87ba4"], "band": "#b7d3f6",
          "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
          "surface": "#fcfcfb"}

#: Categorical slots 2-8 of the validated palette; a comparison run takes the slot after the IDL
#: table's curves, so its color never repeats one of theirs.
SERIES = ["#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

#: Markers of the IDL curves, in table order.
MARKERS = ["s", "^", "D", "v"]

#: Short labels written beside each curve's last point, by IDL column (the legend carries the full
#: ones), and their offsets from that point in points where the default (6, 0) would sit on
#: another curve.
SHORT = {"dec2025_sav": "Dec 2025", "dec2022_5sig": "Dec 2022", "jan2022_norepl": "Jan 2022",
         "jan2024_sav": "Jan 2024", "jan2025_sav": "Jan 2025"}
OFFSETS = {"dec2022_5sig": (-58, 18)}


def read_final_curve(run_dir: str) -> Dict[str, np.ndarray]:
    """Read the final-image grid's curve of a finished run.

    Parameters
    ----------
    run_dir : str
        The run directory, holding ``final_grid/curve.txt``.

    Returns
    -------
    curve : dict of str to ndarray, shape (m,)
        ``sep`` (arcsec), ``median``, ``p16`` and ``p84`` (contrast at S/N 5), ``n_settled``,
        ``n_tested`` and ``annulus`` (from 1); m is the number of separations.
    """
    rows = []
    with open(os.path.join(run_dir, "final_grid", "curve.txt")) as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                p = s.split()
                rows.append([float(v) for v in p[:7]])
    a = np.asarray(rows, float)
    keys = ["sep", "median", "p16", "p84", "n_settled", "n_tested", "annulus"]
    return {k: a[:, i] for i, k in enumerate(keys)}


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
        Each curve's contrast at S/N 5 by its column label, NaN where it has no point; the
        ``deepest`` and ``deepest_curve`` columns are left out.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    labels = [k for k in rows[0] if k not in ("separation_arcsec", "deepest", "deepest_curve")]
    sep = np.array([float(r["separation_arcsec"]) for r in rows])
    return sep, {k: np.array([float(r[k]) if r[k] else np.nan for r in rows]) for k in labels}


def plot(run: Dict[str, np.ndarray], sep: np.ndarray, idl: Dict[str, np.ndarray], out: str, title: str,
         xmin: float, xmax: float, edges: Sequence[float] = (),
         compare: Optional[Dict[str, np.ndarray]] = None,
         compare_label: str = "KLIP-TPE comparison run (median, S/N 5)",
         compare_short: str = "Comparison",
         run_label: str = "KLIP-TPE final image (median, S/N 5)", run_short: str = "KLIP-TPE") -> None:
    """Draw the run's final curve and the IDL curves on one log-contrast axis.

    Parameters
    ----------
    run : dict of str to ndarray
        The run's final curve, as :func:`read_final_curve` gives it.
    sep : ndarray, shape (m,)
        The IDL table's separations, in arcsec; m is the number of rows.
    idl : dict of str to ndarray, shape (m,)
        The IDL curves, as :func:`read_idl` gives them.
    out : str
        Figure to write.
    title : str
        Title of the plot.
    xmin : float
        Smallest separation shown, in arcsec.
    xmax : float
        Largest separation shown, in arcsec.
    edges : sequence of float, default ()
        Annulus edges to mark, in arcsec.
    compare : dict of str to ndarray or None, default None
        A second run's final curve, as :func:`read_final_curve` gives it; its median is drawn.
    compare_label : str, default "KLIP-TPE comparison run (median, S/N 5)"
        Legend entry of the second curve.
    compare_short : str, default "Comparison"
        Label beside the second curve's last point.
    run_label : str, default "KLIP-TPE final image (median, S/N 5)"
        Legend entry of the run's curve.
    run_short : str, default "KLIP-TPE"
        Label beside the run's last point.

    Returns
    -------
    None
        The figure goes to ``out``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})
    fig, ax = plt.subplots(figsize=(7.6, 4.8), facecolor=COLORS["surface"])
    ax.set_facecolor(COLORS["surface"])
    ends = []                                   # (x, y, text, offset) of each curve's label
    for i, (lab, vals) in enumerate(idl.items()):
        ok = np.isfinite(vals) & (sep >= xmin - 1e-9) & (sep <= xmax + 1e-9)
        if not ok.any():
            continue
        c = COLORS["idl"][i % len(COLORS["idl"])]
        ax.plot(sep[ok], vals[ok], color=c, lw=1.6, ls=(0, (5, 3)), marker=MARKERS[i % len(MARKERS)],
                ms=6.5, mfc=c, mec=COLORS["surface"], mew=1.2, label=LABELS.get(lab, lab), zorder=3)
        ends.append((sep[ok][-1], vals[ok][-1], SHORT.get(lab, lab), OFFSETS.get(lab, (6, 0))))
        print(f"{LABELS.get(lab, lab)}: " + ", ".join(f"{s:.2f}\" {v:.3g}" for s, v in zip(sep[ok], vals[ok])))
    if compare is not None:
        c = SERIES[len(idl) % len(SERIES)]
        rc = compare["sep"]
        ax.plot(rc, compare["median"], color=c, lw=1.6, marker="o", ms=6.5, mfc=COLORS["surface"],
                mec=c, mew=1.6, label=compare_label, zorder=3)
        ends.append((rc[-1], compare["median"][-1], compare_short, (6, 0)))
        print(f"{compare_label}: " + ", ".join(f"{s:.2f}\" {v:.3g}" for s, v in zip(rc, compare["median"])))
    r = run["sep"]
    ax.fill_between(r, run["p16"], run["p84"], color=COLORS["band"], alpha=0.55, lw=0, zorder=1,
                    label="KLIP-TPE 16th-84th percentile over position angle")
    ax.plot(r, run["median"], color=COLORS["run"], lw=2.0, marker="o", ms=7, mfc=COLORS["run"],
            mec=COLORS["surface"], mew=1.2, label=run_label, zorder=4)
    ends.append((r[-1], run["median"][-1], run_short, (6, 0)))
    print(f"{run_short} final image: " + ", ".join(f"{s:.2f}\" {v:.3g}" for s, v in zip(r, run["median"])))
    for e in edges:
        ax.axvline(e, color=COLORS["muted"], lw=0.9, ls=":", zorder=2)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(2.0, 3.0, 5.0)))

    def _minor(v: float, _pos: int) -> str:
        """Label a minor tick as m x 10^e, or nothing where it is not a 2, 3 or 5."""
        e = int(np.floor(np.log10(v) + 1e-9))
        m = v / 10.0 ** e
        return f"${m:.0f}\\times10^{{{e}}}$" if round(m) in (2, 3, 5) else ""

    ax.yaxis.set_minor_formatter(FuncFormatter(_minor))
    ax.tick_params(axis="y", which="minor", labelsize=8)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Separation (arcsec)", color=COLORS["ink2"])
    ax.set_ylabel("Contrast at S/N 5", color=COLORS["ink2"])
    ax.set_title(title, color=COLORS["ink"], fontsize=11.5, loc="left")
    ax.grid(True, which="major", color=COLORS["grid"], lw=0.6)
    ax.grid(True, which="minor", axis="y", color=COLORS["grid"], lw=0.4, alpha=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLORS["axis"])
    ax.tick_params(colors=COLORS["muted"], which="both")
    for x, y, text, off in ends:
        ax.annotate(text, (x, y), xytext=off, textcoords="offset points", va="center", ha="left",
                    fontsize=8.5, color=COLORS["ink"], zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", fc=COLORS["surface"], ec="none", alpha=0.8))
    leg = ax.legend(loc="upper right", fontsize=8.5, frameon=True, framealpha=0.9, edgecolor=COLORS["grid"])
    for t in leg.get_texts():
        t.set_color(COLORS["ink"])
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=COLORS["surface"])
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Read the curves and write the figure.

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
    ap.add_argument("out")
    ap.add_argument("--title", default="Contrast at S/N 5: KLIP-TPE Final Image and IDL")
    ap.add_argument("--xmin", type=float, default=0.3)
    ap.add_argument("--xmax", type=float, default=2.6)
    ap.add_argument("--edges", type=float, nargs="+", default=[])
    ap.add_argument("--run-label", default="KLIP-TPE final image (median, S/N 5)")
    ap.add_argument("--run-short", default="KLIP-TPE")
    ap.add_argument("--compare")
    ap.add_argument("--compare-label", default="KLIP-TPE comparison run (median, S/N 5)")
    ap.add_argument("--compare-short", default="Comparison")
    a = ap.parse_args(argv)
    run = read_final_curve(a.run_dir)
    sep, idl = read_idl(a.idl_csv)
    compare = read_final_curve(a.compare) if a.compare else None
    plot(run, sep, idl, a.out, a.title, a.xmin, a.xmax, a.edges, compare, a.compare_label,
         a.compare_short, a.run_label, a.run_short)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
