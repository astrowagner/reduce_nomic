"""Read the contrast curves drawn in an IDL contrast-curve PDF (the plot of contrastcurve_nomic.pro).

usage: python analysis/procyon4_2026-11/idl_curve_pdf.py PDF [PDF ...] [--csv OUT]

contrastcurve_nomic.pro plots contrast on a log axis against angular separation in arcsec: one
gray curve per position angle, their mean in black (a thicker line), and model curves in color.
This reads the plot's vector paths with PyMuPDF. The frame's major ticks calibrate both axes:
separation from the ticks matched to their numeric labels, log contrast from the decade ticks
matched to their 10^k labels. The black and gray paths are then converted to data. The page may
be rotated; each axis is found from where its labels lie. Needs PyMuPDF (``pip install pymupdf``).

On ``Procyon_4_contrast_curve.pdf`` the gray curves' medians, minima and maxima per separation
equal those of the save file written with it, and the black curve is within 3 % of the save
file's mean over position angles (the plot is drawn while the curves are still being refined).

Parameters
----------
PDF : str
    One or more contrast-curve PDFs.
--csv : str, optional
    CSV to write, one row per curve vertex: file, curve ("mean" or "pa<j>", the gray curves in
    drawing order), separation (arcsec) and contrast.

Returns
-------
stdout : text
    Per PDF: the axis calibration (page units per arcsec, per decade, and the largest tick
    residual), the title's test counter and S/N, and per separation the black curve's contrast
    and the gray curves' median, minimum, maximum and count.
"""
import argparse
import csv
import re
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pymupdf

BLACK = (0.0, 0.0, 0.0)


def segments(page) -> List[Tuple[Tuple[float, ...], float, List[Tuple[float, float]]]]:
    """Return every stroked path of a page as (color, width, vertices).

    Parameters
    ----------
    page : pymupdf.Page
        The page.

    Returns
    -------
    paths : list of tuple
        ``(color, width, points)``: the stroke color rounded to 2 decimals, the line width in
        page units, and the path's vertices in drawing order, each ``(x, y)`` in page units.
    """
    out = []
    for d in page.get_drawings():
        if not d.get("color"):
            continue
        pts: List[Tuple[float, float]] = []
        for it in d["items"]:
            if it[0] != "l":
                continue
            a, b = (it[1].x, it[1].y), (it[2].x, it[2].y)
            if not pts or (abs(pts[-1][0] - a[0]) > 1e-6 or abs(pts[-1][1] - a[1]) > 1e-6):
                pts.append(a)
            pts.append(b)
        if pts:
            out.append((tuple(round(v, 2) for v in d["color"]), float(d.get("width") or 0.0), pts))
    return out


def frame(paths) -> Tuple[float, float, float, float]:
    """Find the plot's frame: the rectangle of the longest black axis-parallel segments.

    Parameters
    ----------
    paths : list of tuple
        Output of :func:`segments`.

    Returns
    -------
    x0, x1, y0, y1 : float
        The frame's edges in page units, x0 < x1 and y0 < y1.
    """
    xs, ys = [], []
    for col, _, pts in paths:
        if col != BLACK:
            continue
        for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
            if abs(ax - bx) < 1e-3 and abs(ay - by) > 150:
                xs.append(ax)
            if abs(ay - by) < 1e-3 and abs(ax - bx) > 150:
                ys.append(ay)
    if len(set(np.round(xs, 2))) < 2 or len(set(np.round(ys, 2))) < 2:
        raise ValueError("no frame found")
    return min(xs), max(xs), min(ys), max(ys)


def labels(page) -> List[Tuple[str, float, float, Tuple[float, float, float, float]]]:
    """Return the page's text spans as (text, x center, y center, bbox), each span once.

    Parameters
    ----------
    page : pymupdf.Page
        The page.

    Returns
    -------
    spans : list of tuple
        ``(text, xc, yc, bbox)`` in page units.
    """
    seen, out = set(), []
    for b in page.get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            for s in ln["spans"]:
                t = s["text"].strip()
                key = (t, tuple(round(v, 1) for v in s["bbox"]))
                if t and key not in seen:
                    seen.add(key)
                    x0, y0, x1, y1 = s["bbox"]
                    out.append((t, 0.5 * (x0 + x1), 0.5 * (y0 + y1), (x0, y0, x1, y1)))
    return out


def ticks_on(paths, edge_axis: str, edge: float) -> List[Tuple[float, float]]:
    """Return the tick marks standing on one frame edge, as (position along the edge, length).

    Parameters
    ----------
    paths : list of tuple
        Output of :func:`segments`.
    edge_axis : {"x", "y"}
        "x" for an edge at constant page x, "y" for an edge at constant page y.
    edge : float
        The edge's page coordinate.

    Returns
    -------
    ticks : list of tuple of 2 floats
        Position along the edge and tick length, page units, for black segments under 25 units
        perpendicular to the edge with one end on it.
    """
    out = set()
    for col, _, pts in paths:
        if col != BLACK:
            continue
        for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
            if edge_axis == "x" and abs(ay - by) < 1e-3 and 0 < abs(ax - bx) < 25 and min(abs(ax - edge), abs(bx - edge)) < 0.05:
                out.add((round(ay, 3), round(abs(ax - bx), 2)))
            if edge_axis == "y" and abs(ax - bx) < 1e-3 and 0 < abs(ay - by) < 25 and min(abs(ay - edge), abs(by - edge)) < 0.05:
                out.add((round(ax, 3), round(abs(ay - by), 2)))
    return sorted(out)


def calibrate(page, paths):
    """Calibrate both axes from the frame's major ticks and their labels.

    Parameters
    ----------
    page : pymupdf.Page
        The page.
    paths : list of tuple
        Output of :func:`segments`.

    Returns
    -------
    to_sep : callable
        Page point (x, y) to separation, arcsec.
    to_logc : callable
        Page point (x, y) to log10 contrast.
    info : dict
        ``px_per_arcsec``, ``px_per_decade`` and ``max_resid`` (largest tick residual, page units).
    """
    x0, x1, y0, y1 = frame(paths)
    spans = labels(page)
    num = [(float(t), xc, yc) for t, xc, yc, _ in spans if re.fullmatch(r"\d+(\.\d+)?", t)]
    tens = [(xc, yc, bb) for t, xc, yc, bb in spans if t == "10"]
    exps = [(int(t.replace("−", "-")), xc, yc, bb) for t, xc, yc, bb in spans
            if re.fullmatch(r"[−-]\d+", t)]
    # Separation labels sit outside the frame along one edge, at near-constant page x or page y.
    # The plot has a second, projected-separation (au) axis on the opposite edge, so take the
    # labeled edge nearest the "Angular Separation (arcsec)" title.
    title = next(((xc, yc) for t, xc, yc, _ in spans if t.startswith("Angular Separation")), None)
    if title is None:
        raise ValueError("no 'Angular Separation' axis title")
    best = None
    # axis "y": a frame edge at constant page y, labels spread along page x; "x": the reverse.
    for axis, edges in (("y", (y0, y1)), ("x", (x0, x1))):
        for e in edges:
            near = [(v, xc, yc) for v, xc, yc in num if abs((yc if axis == "y" else xc) - e) < 25
                    and (x0 - 5 <= xc <= x1 + 5 if axis == "y" else y0 - 5 <= yc <= y1 + 5)]
            if len(near) < 3:
                continue
            perp = np.mean([yc if axis == "y" else xc for _, xc, yc in near])
            gap = abs(perp - (title[1] if axis == "y" else title[0]))
            if best is None or gap < best[3]:
                best = (axis, e, near, gap)
    if best is None:
        raise ValueError("no separation labels found")
    s_axis, s_edge, s_lab, _ = best
    s_ticks = ticks_on(paths, s_axis, s_edge)
    tmax = max(t[1] for t in s_ticks)
    s_major = [p for p, ln in s_ticks if ln > 0.75 * tmax]
    pairs = []
    for v, xc, yc in s_lab:
        c = xc if s_axis == "y" else yc
        p = min(s_major, key=lambda q: abs(q - c))
        if abs(p - c) < 12:
            pairs.append((v, p))
    pairs = sorted(set(pairs))
    pv = np.array(pairs)
    sa, sb = np.polyfit(pv[:, 1], pv[:, 0], 1)
    resid_s = np.abs(np.polyval([sa, sb], pv[:, 1]) - pv[:, 0]).max() / abs(sa)
    # Contrast labels: "10" with an exponent, outside the frame edge perpendicular to the first.
    c_axis = "x" if s_axis == "y" else "y"
    cands = []
    for e in ((x0, x1) if c_axis == "x" else (y0, y1)):
        lab = []
        for k, xc, yc, bb in exps:
            d = (xc if c_axis == "x" else yc) - e
            if abs(d) < 40:
                ten = min(tens, key=lambda t: np.hypot(t[0] - xc, t[1] - yc))
                along = 0.5 * (min(bb[1], ten[2][1]) + max(bb[3], ten[2][3])) if c_axis == "x" else \
                    0.5 * (min(bb[0], ten[2][0]) + max(bb[2], ten[2][2]))
                lab.append((k, along))
        if len(lab) >= 2:
            cands.append((e, lab))
    if not cands:
        raise ValueError("no contrast labels found")
    c_edge, c_lab = max(cands, key=lambda t: len(t[1]))
    c_ticks = ticks_on(paths, c_axis, c_edge)
    tmax = max(t[1] for t in c_ticks)
    c_major = sorted(p for p, ln in c_ticks if ln > 0.75 * tmax)
    pairs = []
    for k, along in c_lab:
        p = min(c_major, key=lambda q: abs(q - along))
        pairs.append((k, p))
    pairs = sorted(set(pairs))
    cv = np.array(pairs, float)
    ca, cb = np.polyfit(cv[:, 1], cv[:, 0], 1)
    resid_c = np.abs(np.polyval([ca, cb], cv[:, 1]) - cv[:, 0]).max() / abs(ca)
    # On a log axis the major ticks are a decade apart, so a label matched to the wrong tick
    # shows as unequal spacing: check every major tick against the fit.
    for p in c_major:
        kk = np.polyval([ca, cb], p)
        if abs(kk - round(kk)) > 0.02:
            raise ValueError(f"major tick at {p:.2f} sits at log10 c = {kk:.3f}, not a decade")
    if s_axis == "y":
        def to_sep(pt):
            return float(np.polyval([sa, sb], pt[0]))

        def to_logc(pt):
            return float(np.polyval([ca, cb], pt[1]))
    else:
        def to_sep(pt):
            return float(np.polyval([sa, sb], pt[1]))

        def to_logc(pt):
            return float(np.polyval([ca, cb], pt[0]))
    info = {"px_per_arcsec": abs(1.0 / sa), "px_per_decade": abs(1.0 / ca),
            "max_resid": float(max(resid_s, resid_c))}
    return to_sep, to_logc, info


def read_curves(path: str) -> Dict:
    """Read one PDF's black (mean) and gray (per position angle) curves.

    Parameters
    ----------
    path : str
        The PDF.

    Returns
    -------
    out : dict
        ``mean``: list of (separation, contrast); ``pa``: list of such lists, one per gray
        curve; ``info``: the calibration (:func:`calibrate`); ``title``: the "Test ... SNR"
        text, or "".
    """
    page = pymupdf.open(path)[0]
    paths = segments(page)
    to_sep, to_logc, info = calibrate(page, paths)
    x0, x1, y0, y1 = frame(paths)
    thick = [p for p in paths if p[0] == BLACK and len(p[2]) > 3 and
             not all(abs(a[0] - b[0]) < 1e-3 or abs(a[1] - b[1]) < 1e-3 for a, b in zip(p[2][:-1], p[2][1:]))]
    if not thick:
        raise ValueError("no black curve")
    w = max(p[1] for p in thick)
    blacks = [p for p in thick if abs(p[1] - w) < 1e-3]
    if len(blacks) != 1:
        raise ValueError(f"{len(blacks)} black curves of width {w:.3f}")
    conv = lambda pts: [(round(to_sep(q), 4), 10 ** to_logc(q)) for q in pts]
    grays = [conv(p[2]) for p in paths if p[0] != BLACK and abs(p[0][0] - p[0][1]) < 0.01
             and abs(p[0][1] - p[0][2]) < 0.01 and 0.3 < p[0][0] < 0.95]
    title = next((t for t, *_ in labels(page) if t.startswith("Test")), "")
    return {"mean": conv(blacks[0][2]), "pa": grays, "info": info, "title": title}


def main(argv=None) -> int:
    """Print each PDF's curves and optionally write them to a CSV.

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
    ap.add_argument("pdf", nargs="+")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args(argv)
    rows = []
    for f in a.pdf:
        r = read_curves(f)
        i = r["info"]
        print(f"{f}\n  {r['title']}; {i['px_per_arcsec']:.3f} page units per arcsec, "
              f"{i['px_per_decade']:.3f} per decade, largest tick residual {i['max_resid']:.3f} units; "
              f"{len(r['pa'])} gray curves")
        bysep = defaultdict(list)
        for k, curve in enumerate(r["pa"]):
            for s, c in curve:
                bysep[round(s, 2)].append(c)
                rows.append((f, f"pa{k}", s, c))
        for s, c in r["mean"]:
            g = np.array(bysep.get(round(s, 2), [np.nan]))
            print(f"  {s:5.2f}\": mean curve {c:.3g}; gray median {np.nanmedian(g):.3g}, "
                  f"min {np.nanmin(g):.3g}, max {np.nanmax(g):.3g}, n {np.count_nonzero(np.isfinite(g))}")
            rows.append((f, "mean", s, c))
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "curve", "separation_arcsec", "contrast"])
            w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
