"""Convert a target's contrast curves to limits on stellar companions, with tables and a figure.

usage: python tools/companion_limits.py SPEC --archive ROOT --out DIR [--no-plot]

``SPEC`` is a JSON file (example: ``config/companion_limits/vega_idl.json``) giving the star
(parallax, apparent magnitude in the band), the band, the dwarf-sequence range, the curves (paths
relative to ``--archive``, one entry per curve with ``id``, ``night``, ``date``, ``side``,
``method``, ``cap``, ``plot`` and ``kind``: ``idl_sav`` (default; per position angle) or
``klip_tpe`` (a KLIP-TPE ``contrast_curve.txt``, azimuthally combined, counted in both envelopes);
``"optional": true`` leaves a missing file out instead of stopping), the companion types to mark, the projected separations to mark
(``marks_au``) and ``min_pa_worst``, the fewest position angles a curve needs to count in the
worst-position-angle envelope. Each curve's limits are summarized over position angle (median and
worst), converted with ``reduce_nomic.companions`` to the limiting absolute N' magnitude, spectral
type and mass, and combined into the lowest limit at each separation over all curves (median) and
over the curves with at least ``min_pa_worst`` position angles (worst position angle).

Parameters
----------
SPEC : str
    The JSON specification.
--archive : str
    Root the curve paths are relative to: the NOMIC archive, or a staged copy of its curve files.
--out : str
    Output folder, created if missing.
--no-plot : flag
    Skip the figure (no matplotlib needed).

Returns
-------
<out>/<name>_per_curve.csv : file
    One row per curve and separation: limits over position angle and the companion limits they
    give (columns described in ``per_curve_rows``).
<out>/<name>_combined.csv : file
    The two envelopes at every separation (``combined_rows``).
<out>/<name>_sequence.csv : file
    The dwarf sequence used, with each subtype's contrast at the star's distance.
<out>/<name>_summary.txt : file
    The numbers a reader needs first: the band, the distance, the marked separations in mas and
    λ/D, the contrast of the shaded companion types, and the envelopes.
<out>/<name>_spec.json : file
    Copy of the specification.
<out>/<name>_limits.png, <out>/<name>_limits.pdf : file
    Contrast limits by night (left) and the limiting companion mass against projected separation
    (right). Not written with ``--no-plot``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from typing import Dict, List, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reduce_nomic import companions as cp  # noqa: E402

#: Categorical slots (light mode) of the reference palette, in order; checked with the dataviz
#: validator for five lines (all checks pass; three slots need visible labels, which the legend and
#: tables give).
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
#: Marker per night, the secondary encoding beside color.
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def per_curve_rows(spec: dict, curves: Dict[str, cp.ContrastCurve], seq: cp.DwarfSequence) -> List[dict]:
    """Tabulate each curve's limits and the companion limits they give.

    Parameters
    ----------
    spec : dict
        The specification (``star`` and ``curves`` are used).
    curves : dict of str to ContrastCurve
        The curves by ``id``.
    seq : DwarfSequence
        Dwarf sequence in the band.

    Returns
    -------
    rows : list of dict
        One per curve and separation with: ``id``, ``night``, ``date``, ``side``, ``method``,
        ``folder``, ``sigma``, ``n_pa`` (position angles with any limit), ``sep_arcsec``,
        ``sep_au``, ``n_ok`` (position angles with a limit at this separation), ``contrast_median``,
        ``contrast_mean``, ``contrast_worst``, ``contrast_best``, and for the median and the worst
        position angle: ``dmag_*``, ``m_abs_*`` (mag), ``spt_*``, ``mass_*`` (solar masses) and
        ``beyond_*`` (``brighter``/``fainter`` when outside the sequence).
    """
    st = spec["star"]
    rows = []
    for c in spec["curves"]:
        cur = curves[c["id"]]
        s = cur.stats()
        lim = {k: cp.companion_limits(s[k], st["mag"], st["parallax_mas"], seq) for k in ("median", "worst")}
        for j, sep in enumerate(cur.separations):
            r = {"id": c["id"], "night": c["night"], "date": c["date"], "side": c["side"], "method": c["method"],
                 "folder": c.get("folder", ""), "sigma": f"{cur.sigma:g}", "n_pa": len(cur.pas),
                 "sep_arcsec": f"{sep:.3f}", "sep_au": f"{cp.projected_au(sep, st['parallax_mas']):.3f}",
                 "n_ok": int(s["n_ok"][j])}
            for k in ("median", "mean", "worst", "best"):
                r[f"contrast_{k}"] = _fmt(s[k][j], ".3e")
            for k in ("median", "worst"):
                r[f"dmag_{k}"] = _fmt(lim[k]["dmag"][j], ".3f")
                r[f"m_abs_{k}"] = _fmt(lim[k]["m_abs"][j], ".3f")
                r[f"spt_{k}"] = lim[k]["spt"][j]
                r[f"mass_{k}"] = _fmt(lim[k]["mass"][j], ".3f")
                r[f"beyond_{k}"] = lim[k]["beyond"][j]
            rows.append(r)
    return rows


def envelopes(spec: dict, curves: Dict[str, cp.ContrastCurve]) -> Dict[str, np.ndarray]:
    """Lowest limit at each separation: median over all curves, worst position angle over the
    curves with at least ``min_pa_worst`` position angles.

    Parameters
    ----------
    spec : dict
        The specification (``curves`` and ``min_pa_worst`` are used).
    curves : dict of str to ContrastCurve
        The curves by ``id``.

    Returns
    -------
    env : dict of str to ndarray, each shape (n_grid,)
        ``grid`` (every separation of any curve, arcsec), ``median`` and ``median_from`` (lowest
        median limit and the curve id giving it), ``worst`` and ``worst_from`` (lowest worst-angle
        limit among the well-sampled curves and its curve id); NaN and ``""`` where there is none.
        Each curve is interpolated in log10(contrast) between its measured separations
        (``companions.on_grid``), so curves on different grids are compared everywhere.
    """
    med = {c["id"]: (curves[c["id"]].separations, curves[c["id"]].stats()["median"]) for c in spec["curves"]}
    grid, m, mf = cp.best_envelope(med)
    nmin = int(spec.get("min_pa_worst", 10))
    wor = {c["id"]: (curves[c["id"]].separations, curves[c["id"]].stats()["worst"])
           for c in spec["curves"] if len(curves[c["id"]].pas) >= nmin or c.get("kind") == "klip_tpe"}
    _, w, wf = cp.best_envelope(wor, grid) if wor else (grid, np.full(len(grid), np.nan), np.array([""] * len(grid)))
    return {"grid": grid, "median": m, "median_from": mf, "worst": w, "worst_from": wf}


def combined_rows(spec: dict, env: Dict[str, np.ndarray], seq: cp.DwarfSequence) -> List[dict]:
    """Tabulate the envelopes and the companion limits they give.

    Parameters
    ----------
    spec : dict
        The specification (``star`` and ``companions`` are used).
    env : dict of str to ndarray
        Output of ``envelopes``.
    seq : DwarfSequence
        Dwarf sequence in the band.

    Returns
    -------
    rows : list of dict
        One per separation with ``sep_arcsec``, ``sep_au`` and, for ``median`` and ``worst``: the
        contrast, the curve it comes from, ``spt_*``, ``mass_*`` and ``excludes_<A>_<B>_*`` (True
        when the limit is below the contrast of the later end of the shaded types, so that a
        companion of any shaded type would have been seen).
    """
    st = spec["star"]
    lo, hi = spec["companions"]["shade"]
    c_late = cp.absmag_to_contrast(seq.m_w3_of(hi), st["mag"], st["parallax_mas"])
    rows = []
    lim = {k: cp.companion_limits(env[k], st["mag"], st["parallax_mas"], seq) for k in ("median", "worst")}
    for j, sep in enumerate(env["grid"]):
        r = {"sep_arcsec": f"{sep:.3f}", "sep_au": f"{cp.projected_au(sep, st['parallax_mas']):.3f}"}
        for k in ("median", "worst"):
            v = env[k][j]
            r[f"contrast_{k}"] = _fmt(v, ".3e")
            r[f"from_{k}"] = env[f"{k}_from"][j]
            r[f"spt_{k}"] = lim[k]["spt"][j] or lim[k]["beyond"][j]
            r[f"mass_{k}"] = _fmt(lim[k]["mass"][j], ".3f")
            r[f"excludes_{lo}_{hi}_{k}"] = "" if not np.isfinite(v) else str(bool(v < c_late))
        rows.append(r)
    return rows


def _fmt(v: float, spec: str) -> str:
    """Format a number for a table, empty for NaN.

    Parameters
    ----------
    v : float
        Value.
    spec : str
        Format specification, e.g. ``".3f"``.

    Returns
    -------
    text : str
        The formatted value, or ``""`` when it is not finite.
    """
    return format(float(v), spec) if np.isfinite(v) else ""


def write_csv(path: str, rows: List[dict]) -> None:
    """Write rows of dicts to a CSV file.

    Parameters
    ----------
    path : str
        Output path.
    rows : list of dict
        Rows with the same keys; the first row's keys give the columns.

    Returns
    -------
    None
        The file is written.
    """
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def plot(spec: dict, curves: Dict[str, cp.ContrastCurve], env: Dict[str, np.ndarray],
         seq: cp.DwarfSequence, out_base: str) -> None:
    """Draw the contrast limits by night and the limiting companion mass.

    Parameters
    ----------
    spec : dict
        The specification.
    curves : dict of str to ContrastCurve
        The curves by ``id``.
    env : dict of str to ndarray
        Output of ``envelopes``.
    seq : DwarfSequence
        Dwarf sequence in the band.
    out_base : str
        Output path without extension.

    Returns
    -------
    None
        ``<out_base>.png`` (200 dpi) and ``<out_base>.pdf`` are written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st, band = spec["star"], spec["band"]
    par = st["parallax_mas"]
    lod = cp.lambda_over_d_arcsec(band["lambda_um"], band["aperture_m"])
    to_c = lambda spt: float(cp.absmag_to_contrast(seq.m_w3_of(spt), st["mag"], par))  # noqa: E731
    n_plot = sum(1 for c in spec["curves"] if c.get("plot") and c["id"] in curves)
    # Heights in inches: titles and the top axis, the panels, the x labels, the legend (a title row
    # and one row per two curves), the footnote; the figure grows with the legend.
    top_in, plot_in, xlab_in, foot_in = 1.3, 4.0, 0.6, 0.45
    leg_in = 0.2 * (1 + (n_plot + 1) // 2) + 0.1
    height = top_in + plot_in + xlab_in + leg_in + foot_in
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.0, height), gridspec_kw={"width_ratios": [1.25, 1.0]})
    fig.subplots_adjust(left=0.065, right=0.985, top=1 - top_in / height,
                        bottom=(xlab_in + leg_in + foot_in) / height, wspace=0.22)
    plain = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
    for a in (ax, bx):
        a.set_facecolor("#fcfcfb")
        a.grid(True, which="major", color=GRID, lw=0.6)
        a.set_axisbelow(True)
        for sp in a.spines.values():
            sp.set_color("#c3c2b7")
        a.tick_params(colors=INK2, labelsize=9)

    # left: contrast limits by night
    xlo, xhi = 0.04, 7.0
    lo, hi = spec["companions"]["shade"]
    ax.axhspan(to_c(hi), to_c(lo), color=MUTED, alpha=0.18, lw=0)
    ax.text(0.075, np.sqrt(to_c(hi) * to_c(lo)), f"{spec['companions']['shade_label']} Companions",
            ha="left", va="center", fontsize=8.5, color=INK2)
    for spt in spec["companions"]["lines"]:
        c = to_c(spt)
        ax.axhline(c, color=MUTED, lw=0.8, ls=(0, (4, 3)))
        ax.text(xlo * 1.12, c * 1.04, spt, ha="left", va="bottom", fontsize=8, color=INK2)
    nights = []
    for c in spec["curves"]:
        if c.get("plot") and c["night"] not in nights:
            nights.append(c["night"])
    for c in spec["curves"]:
        if not c.get("plot"):
            continue
        k = nights.index(c["night"])
        cur = curves[c["id"]]
        s = cur.stats()
        n_pa = len(cur.pas)
        azim = c.get("kind") == "klip_tpe"
        well = azim or n_pa >= int(spec.get("min_pa_worst", 10))
        ok = np.isfinite(s["median"])
        lab = (f"{c['date']} {c.get('short', c['method'].split(' (')[0])}" +
               (", Azimuthal" if azim else f", {n_pa} PA{'s' if n_pa != 1 else ''}"))
        rdi = "RDI" in c["method"]                   # same night, same color and marker: dash-dot tells RDI apart
        ls = (0, (7, 2, 1.5, 2)) if rdi else ("-" if well else (0, (5, 2.5)))
        ax.plot(cur.separations[ok], s["median"][ok], color=SLOTS[k], lw=2.0 if well else 1.4,
                ls=ls, marker=MARKERS[k], ms=6 if well else 5,
                mfc=SLOTS[k] if well else "white", mec=SLOTS[k], mew=1.4, label=lab, zorder=3)
        if well and not azim:
            ax.fill_between(cur.separations[ok], s["median"][ok], s["worst"][ok], color=SLOTS[k],
                            alpha=0.14, lw=0, zorder=2)
    x05 = [float(cp.arcsec_of_au(a, par)) for a in spec.get("marks_au", [])]
    for a, x in zip(spec.get("marks_au", []), x05):
        ax.axvline(x, color=INK, lw=1.0, ls=(0, (1, 2)))
        ax.text(x * 1.06, 1.5e-4, f"{a:g} au ({x * 1000:.0f} mas)", rotation=90, ha="left", va="bottom",
                fontsize=8.5, color=INK)
    ax.axvline(lod, color=INK2, lw=0.8)
    ax.text(lod * 1.04, 1.5e-4, f"λ/D = {lod:.2f}″", rotation=90, ha="left", va="bottom",
            fontsize=8.5, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(1e-4, 0.12)
    ax.set_xlabel("Angular Separation (arcsec)", fontsize=10, color=INK)
    ax.set_ylabel(f"5σ Contrast at {band['name']}", fontsize=10, color=INK)
    ax.set_title("Contrast Limits by Night (Line: Median PA; Band: To Worst PA)", fontsize=10.5, color=INK)
    top = ax.secondary_xaxis("top", functions=(lambda x: cp.projected_au(x, par), lambda a: cp.arcsec_of_au(a, par)))
    top.set_xlabel("Projected Separation (au)", fontsize=10, color=INK)
    top.tick_params(colors=INK2, labelsize=9)
    top.set_xticks([0.5, 1, 2, 5, 10, 20, 50])
    top.xaxis.set_major_formatter(plain)
    top.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([0.05, 0.1, 0.2, 0.5, 1, 2, 5])
    ax.xaxis.set_major_formatter(plain)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    leg = ax.legend(loc="upper left", bbox_to_anchor=(0.065, (leg_in + foot_in) / height), bbox_transform=fig.transFigure,
                    ncol=2, fontsize=8.5, frameon=False, borderaxespad=0.0,
                    handlelength=2.6, title="Night, Method, Position Angles With Limits", title_fontsize=8.5,
                    alignment="left")
    leg.get_title().set_color(INK2)

    # right: limiting mass against projected separation
    au = cp.projected_au(env["grid"], par)
    m_med = seq.mass_at(cp.contrast_to_absmag(env["median"], st["mag"], par))
    m_wor = seq.mass_at(cp.contrast_to_absmag(env["worst"], st["mag"], par))
    ylo_m, yhi_m = 0.05, 1.2
    mlo, mhi = float(seq.mass[np.isclose(seq.sptnum, cp.spt_to_num(hi))][0]), float(seq.mass[np.isclose(seq.sptnum, cp.spt_to_num(lo))][0])
    bx.axhspan(mlo, mhi, color=MUTED, alpha=0.18, lw=0)
    bx.text(0.33, np.sqrt(mlo * mhi), f"{spec['companions']['shade_label']}", ha="left", va="center", fontsize=8.5, color=INK2)
    for spt in spec["companions"]["lines"]:
        mm = float(seq.mass[np.isclose(seq.sptnum, cp.spt_to_num(spt))][0])
        bx.axhline(mm, color=MUTED, lw=0.8, ls=(0, (4, 3)))
        bx.text(0.33, mm * 1.03, spt, ha="left", va="bottom", fontsize=8, color=INK2)
    okm, okw = np.isfinite(m_med), np.isfinite(m_wor)
    bx.plot(au[okm], m_med[okm], color=INK, lw=2.0, marker="o", ms=5.5, label="Best Night at Each Separation, Median PA", zorder=3)
    if okw.any():
        bx.plot(au[okw], m_wor[okw], color=INK2, lw=1.4, ls=(0, (5, 2.5)), marker="s", ms=5, mfc="white",
                label=f"Best Night with ≥{int(spec.get('min_pa_worst', 10))} PAs, Worst PA", zorder=3)
    for a in spec.get("marks_au", []):
        bx.axvline(a, color=INK, lw=1.0, ls=(0, (1, 2)))
        bx.text(a * 1.06, ylo_m * 1.15, f"{a:g} au", rotation=90, ha="left", va="bottom", fontsize=8.5, color=INK)
    inner = np.nanmin(au[okm]) if okm.any() else np.nan
    if np.isfinite(inner):
        bx.axvspan(0.3, inner, color="#f0efec", lw=0, zorder=0)
        bx.text(np.sqrt(0.3 * inner), 0.3, f"No Imaging Limits\nInside {inner:.1f} au", ha="center", va="center",
                fontsize=8.5, color=INK2)
    bx.set_xscale("log")
    bx.set_yscale("log")
    bx.set_xlim(0.3, 45)
    bx.set_ylim(ylo_m, yhi_m)
    bx.set_xlabel("Projected Separation (au)", fontsize=10, color=INK)
    bx.set_ylabel("Mass of Companion That Could Be Missed (M☉)", fontsize=10, color=INK)
    bx.set_title("Limiting Companion Mass (Main-Sequence Dwarfs)", fontsize=10.5, color=INK)
    bx.legend(loc="lower right", fontsize=8, frameon=False, handlelength=2.6)
    bx.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    bx.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    bx.set_yticks([0.05, 0.1, 0.2, 0.5, 1.0])
    bx.set_xticks([0.5, 1, 2, 5, 10, 20, 40])
    bx.xaxis.set_major_formatter(plain)
    bx.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    fig.suptitle(spec["title"], fontsize=12.5, color=INK, y=0.985)
    miss = "; ".join(f"{m['date']}: {m['note'].split(':')[0]}" for m in spec.get("missing", []))
    foot = (f"d = {1000 / par:.2f} pc ({st['parallax_mas']} ± {st['parallax_err_mas']} mas); "
            f"m({band['name']}) = {st['mag']:.2f}; companion magnitudes: W3 as the {band['name']} proxy, "
            f"M_Ks and masses from Mamajek (2022.04.16), Ks−W3 from Mamajek through K5V and Cifuentes et al. (2020) from K7V. "
            f"Open markers and dashed lines: 1–2 position angles." +
            (" Dash-dot lines: RDI." if any("RDI" in c["method"] for c in spec["curves"] if c["id"] in curves) else "") +
            (f" {miss}." if miss else ""))
    fig.text(0.065, 0.1 / height, foot, fontsize=7.5, color=INK2, wrap=True, ha="left", va="bottom")
    fig.savefig(out_base + ".png", dpi=200, facecolor="white")
    fig.savefig(out_base + ".pdf", facecolor="white")
    plt.close(fig)


def summary_text(spec: dict, curves: Dict[str, cp.ContrastCurve], env: Dict[str, np.ndarray],
                 seq: cp.DwarfSequence) -> str:
    """Write the summary a reader needs first.

    Parameters
    ----------
    spec : dict
        The specification.
    curves : dict of str to ContrastCurve
        The curves by ``id``.
    env : dict of str to ndarray
        Output of ``envelopes``.
    seq : DwarfSequence
        Dwarf sequence in the band.

    Returns
    -------
    text : str
        Several lines of plain text.
    """
    st, band = spec["star"], spec["band"]
    par = st["parallax_mas"]
    lod = cp.lambda_over_d_arcsec(band["lambda_um"], band["aperture_m"])
    lines = [spec["title"], "",
             f"distance {1000 / par:.3f} pc (parallax {par} +/- {st['parallax_err_mas']} mas, {st['parallax_ref']})",
             f"absolute = apparent + {cp.distance_modulus_term(par):.4f} mag; m({band['name']}) of {st['name']} = {st['mag']} +/- {st['mag_err']}",
             f"lambda/D at {band['lambda_um']} um on {band['aperture_m']} m = {lod * 1000:.0f} mas", ""]
    for a in spec.get("marks_au", []):
        x = float(cp.arcsec_of_au(a, par))
        lines.append(f"{a:g} au = {x * 1000:.1f} mas = {x / lod:.2f} lambda/D")
    lines.append("")
    lines.append(f"contrast at {band['name']} of a dwarf companion at {1000 / par:.2f} pc (M_W3 from the sequence):")
    for spt in ["K0V", "K5V", "K7V", "M0V", "M2V", "M4V", "M5V", "M6V"]:
        m = seq.m_w3_of(spt)
        lines.append(f"  {spt:5s} M_W3 = {m:6.3f}  contrast = {float(cp.absmag_to_contrast(m, st['mag'], par)):.4f}  "
                     f"mass = {float(seq.mass[np.isclose(seq.sptnum, cp.spt_to_num(spt))][0]):.3f}")
    lines.append("")
    lines.append("curves (PAs with any limit; separations with a limit):")
    for c in spec["curves"]:
        cur = curves[c["id"]]
        s = cur.stats()
        lines.append(f"  {c['id']:24s} {c['date']} {c['side']:5s} {len(cur.pas):2d} PAs, sigma {cur.sigma:g}, "
                     f"limits at {int(np.isfinite(s['median']).sum())}/{len(cur.separations)} separations "
                     f"({c.get('folder', '')}, {c['method']})")
    for m in spec.get("missing", []):
        lines.append(f"  {m['night']}: {m['note']}")
    lines.append("")
    lines.append("lowest limit at each separation (median PA over all curves | worst PA over curves with "
                 f">= {spec.get('min_pa_worst', 10)} PAs; each curve interpolated in log contrast between its "
                 "measured separations):")
    lim = {k: cp.companion_limits(env[k], st["mag"], par, seq) for k in ("median", "worst")}
    for j, sep in enumerate(env["grid"]):
        parts = []
        for k in ("median", "worst"):
            v = env[k][j]
            parts.append("--" if not np.isfinite(v) else
                         f"{v:.2e} {lim[k]['spt'][j] or lim[k]['beyond'][j]:>7s} {lim[k]['mass'][j]:.2f} Msun ({env[k + '_from'][j]})")
        lines.append(f"  {sep:5.2f}\" {float(cp.projected_au(sep, par)):5.2f} au | {parts[0]:44s} | {parts[1]}")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str]) -> int:
    """Run the conversion from the command line.

    Parameters
    ----------
    argv : sequence of str
        Command-line arguments without the program name (see the module docstring).

    Returns
    -------
    status : int
        0 on success. The files listed in the module docstring are written.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("spec")
    ap.add_argument("--archive", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args(argv)
    with open(a.spec) as fh:
        spec = json.load(fh)
    seq = cp.dwarf_sequence(spec["sequence"]["first"], spec["sequence"]["last"])
    readers = {"idl_sav": cp.read_idl_curve, "klip_tpe": cp.read_klip_tpe_curve}
    present = []
    for c in spec["curves"]:
        if os.path.exists(os.path.join(a.archive, c["path"])):
            present.append(c)
        elif c.get("optional"):
            print(f"{c['id']}: {c['path']} not found; left out (optional)")
        else:
            raise SystemExit(f"{c['id']}: {c['path']} not found")
    if not present:
        raise SystemExit("no curves found")
    spec = dict(spec, curves=present)
    curves = {c["id"]: readers[c.get("kind", "idl_sav")](os.path.join(a.archive, c["path"])) for c in spec["curves"]}
    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.out, spec["name"])
    write_csv(base + "_per_curve.csv", per_curve_rows(spec, curves, seq))
    env = envelopes(spec, curves)
    write_csv(base + "_combined.csv", combined_rows(spec, env, seq))
    st = spec["star"]
    write_csv(base + "_sequence.csv", [
        {"spt": seq.spt[i], "sptnum": f"{seq.sptnum[i]:g}", "M_Ks": f"{seq.m_ks[i]:.3f}", "Ks_W3": f"{seq.ks_w3[i]:.3f}",
         "Ks_W3_source": seq.ks_w3_source[i], "M_W3": f"{seq.m_w3[i]:.3f}", "mass_msun": f"{seq.mass[i]:.3f}",
         "contrast": f"{float(cp.absmag_to_contrast(seq.m_w3[i], st['mag'], st['parallax_mas'])):.4e}"}
        for i in range(len(seq.spt))])
    with open(base + "_summary.txt", "w") as fh:
        fh.write(summary_text(spec, curves, env, seq))
    shutil.copyfile(a.spec, base + "_spec.json")
    if not a.no_plot:
        plot(spec, curves, env, seq, base + "_limits")
    print(f"wrote {base}_*.csv, _summary.txt, _spec.json" + ("" if a.no_plot else ", _limits.png/.pdf"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
