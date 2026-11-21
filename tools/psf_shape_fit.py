"""Limits on companions inside λ/D from the PSF shape, turning with the parallactic angle.

usage: python tools/psf_shape_fit.py CUTOUT [CUTOUT ...] --out DIR --name NAME [--sigma-w 3]
       [--spins 1 2] [--drift 1] [--seps-mas 30,40,50,65,80,100,130,160,200,270] [--inject 0.02]
       [--n-phi 8] [--parallax 130.23] [--star-mag 0.0] [--no-plot]

Each ``CUTOUT`` is a file from ``tools/extract_psf_cutouts.py``: the registered core of every
binned frame of one night, side and chop, with parallactic angles. For each, the Gaussian-weighted
moments of spin 1 (lopsidedness) and spin 2 (elongation) are fitted as a static term, a drift
linear in frame order (``--drift``) and a term turning with the parallactic angle
(``reduce_nomic.psfshape``), in both senses: +1, the sense of the IDL derotation
(``rot(frame, -angle - truenorth)``: a sky-fixed source sits at detector angle φ - q), and -1, a
null no sky-fixed source follows. Injected copies of the frames' own PSF calibrate the turning
amplitude per unit flux ratio at each separation. The frame sets of one night are combined
(``psfshape.combine``), and the flux ratio above which a companion would have been detected
(false-alarm probability of a one-sided 5σ, ``psfshape.rayleigh_threshold``) is turned into a
spectral type and mass with ``reduce_nomic.companions``.

Parameters
----------
CUTOUT : str
    Cutout files, named ``<night>_<side>_cube<n>_<product>_core<size>.fits``.
--out : str
    Output folder, created if missing.
--name : str
    Prefix of the output files.
--sigma-w : float, default 3.0
    Width of the Gaussian weight, pixels.
--spins : int list, default 1 2
    Moment spins to use.
--drift : int, default 1
    Polynomial order of the drift in frame order.
--seps-mas : str, default "30,40,50,65,80,100,130,160,200,270"
    Separations to calibrate, mas.
--inject : float, default 0.02
    Flux ratio of the injected companions.
--n-phi : int, default 8
    Sky angles per separation.
--parallax : float, default 130.23
    Parallax, mas (for projected separations and companion types).
--star-mag : float, default 0.0
    The star's magnitude in the band (N').
--no-plot : flag
    Skip the figure.

Returns
-------
<out>/<name>_fits.csv : file
    One row per cutout and spin: frames, angle range, turning amplitude and its error in both
    senses, residual autocorrelation and static term.
<out>/<name>_response.csv : file
    One row per cutout, spin and separation: response (moment per unit flux ratio), its spread
    over angles and the phase scatter.
<out>/<name>_limits.csv : file
    One row per night, spin and separation: combined flux-ratio estimate, its error, SNR, χ²,
    the detection limit and the companion type and mass at the limit, and the null (sense -1) SNR.
<out>/<name>_summary.txt : file
    Settings, rotation per night, and the limits at each separation.
<out>/<name>_limits.png, .pdf : file
    Flux-ratio limit against separation per night and spin (not with ``--no-plot``).
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reduce_nomic import companions as cp  # noqa: E402
from reduce_nomic import psfshape as ps  # noqa: E402

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
NAME_RE = re.compile(r"^(?P<night>.+?)_(?P<side>left|right)_cube(?P<cube>\d)_")


def load_cutout(path: str) -> Dict:
    """Read one cutout file.

    Parameters
    ----------
    path : str
        File from ``tools/extract_psf_cutouts.py``.

    Returns
    -------
    cut : dict
        ``frames`` (ndarray, shape (n, ny, nx)), ``q`` (parallactic angles, degrees, shape (n,)),
        ``x0``, ``y0`` (star position, pixels), ``pxscale`` (arcsec per pixel), ``night``,
        ``side``, ``cube`` and ``label`` (``<night> <side> chop A/B``).
    """
    m = NAME_RE.match(os.path.basename(path))
    if not m:
        raise ValueError(f"cannot read night, side and cube from {os.path.basename(path)}")
    with fits.open(path) as hd:
        fr = np.asarray(hd[0].data, dtype=np.float64)
        q = np.asarray(hd["ANGLES"].data, dtype=np.float64)
        h = hd[0].header
        x0, y0, px = float(h["XCEN"]), float(h["YCEN"]), float(h.get("PXSCALE", 0.0179))
    chop = "A" if m.group("cube") == "1" else "B"
    return {"frames": fr, "q": q, "x0": x0, "y0": y0, "pxscale": px, "night": m.group("night"),
            "side": m.group("side"), "cube": int(m.group("cube")), "label": f"{m.group('night')} {m.group('side')} chop {chop}"}


def write_csv(path: str, rows: List[dict]) -> None:
    """Write rows of dicts to a CSV file.

    Parameters
    ----------
    path : str
        Output path.
    rows : list of dict
        Rows with the same keys.

    Returns
    -------
    None
        The file is written.
    """
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv: Sequence[str]) -> int:
    """Run the PSF-shape fit from the command line.

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
    ap.add_argument("cutouts", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--sigma-w", type=float, default=3.0)
    ap.add_argument("--spins", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--drift", type=int, default=1)
    ap.add_argument("--seps-mas", default="30,40,50,65,80,100,130,160,200,270")
    ap.add_argument("--inject", type=float, default=0.02)
    ap.add_argument("--n-phi", type=int, default=8)
    ap.add_argument("--parallax", type=float, default=130.23)
    ap.add_argument("--star-mag", type=float, default=0.0)
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args(argv)
    seps = [float(x) for x in a.seps_mas.split(",")]
    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.out, a.name)
    bg = None
    fit_rows, resp_rows = [], []
    per = defaultdict(list)                        # (night, spin) -> list of (b, sigma_b, r(sep) dict, label, null fit)
    rot = {}
    for path in a.cutouts:
        c = load_cutout(path)
        n, ny, nx = c["frames"].shape
        bg = (min(ny, nx) * 0.36, min(ny, nx) * 0.49)
        mom = ps.weighted_moments(c["frames"], a.sigma_w, c["x0"], c["y0"], bg_annulus=bg)
        span = float(np.ptp(c["q"]))
        rot.setdefault(c["night"], []).append((c["label"], len(c["q"]), float(c["q"].min()), float(c["q"].max()), span))
        fits_s = {}
        for m in a.spins:
            fp = ps.fit_rotating(mom[f"spin{m}"], c["q"], m, +1, a.drift)
            fn = ps.fit_rotating(mom[f"spin{m}"], c["q"], m, -1, a.drift)
            fits_s[m] = fp
            fit_rows.append({"cutout": c["label"], "night": c["night"], "side": c["side"], "chop": "AB"[c["cube"] - 1],
                             "spin": m, "n": fp.n, "q_min": f"{c['q'].min():.2f}", "q_max": f"{c['q'].max():.2f}",
                             "q_span": f"{span:.2f}", "b_abs": f"{abs(fp.b):.3e}", "sigma_b": f"{fp.sigma_b:.3e}",
                             "snr": f"{abs(fp.b) / fp.sigma_b:.2f}", "rho": f"{fp.rho:.2f}",
                             "b_abs_null": f"{abs(fn.b):.3e}", "sigma_b_null": f"{fn.sigma_b:.3e}",
                             "snr_null": f"{abs(fn.b) / fn.sigma_b:.2f}", "static_re": f"{fp.static.real:.4e}",
                             "static_im": f"{fp.static.imag:.4e}", "resid_rms": f"{fp.resid_rms:.3e}"})
        resp = {}
        for s in seps:
            cal = ps.response(c["frames"], c["q"], a.sigma_w, c["x0"], c["y0"], a.spins, s / 1000.0 / c["pxscale"],
                              a.inject, a.n_phi, a.drift, +1, bg, base=fits_s)
            resp[s] = cal
            for m in a.spins:
                resp_rows.append({"cutout": c["label"], "spin": m, "sep_mas": f"{s:g}", "r": f"{cal[m]['r']:.4e}",
                                  "r_spread": f"{cal[m]['r_spread']:.4e}", "phase_sd_deg": f"{cal[m]['phase_sd']:.1f}"})
        for m in a.spins:
            fn = ps.fit_rotating(mom[f"spin{m}"], c["q"], m, -1, a.drift)
            per[(c["night"], m)].append((fits_s[m], {s: resp[s][m]["r"] for s in seps}, c["label"], fn))
        print(f"{c['label']}: {n} frames, rotation {span:.1f} deg, "
              + ", ".join(f"spin {m} |b|/sigma {abs(fits_s[m].b) / fits_s[m].sigma_b:.2f}" for m in a.spins), flush=True)
    write_csv(base + "_fits.csv", fit_rows)
    write_csv(base + "_response.csv", resp_rows)

    seq = cp.dwarf_sequence()
    t1 = ps.rayleigh_threshold()
    lim_rows = []
    for (night, m), items in sorted(per.items()):
        for s in seps:
            comb = ps.combine([it[0].b for it in items], [it[0].sigma_b for it in items], [it[1][s] for it in items])
            null = ps.combine([it[3].b for it in items], [it[3].sigma_b for it in items], [it[1][s] for it in items])
            flim = t1 * comb["sigma_f"]
            lim = cp.companion_limits(np.array([flim]), a.star_mag, a.parallax, seq)
            lim_rows.append({"night": night, "spin": m, "sets": len(items), "sep_mas": f"{s:g}",
                             "sep_au": f"{float(cp.projected_au(s / 1000.0, a.parallax)):.3f}",
                             "f_est": f"{comb['f_abs']:.4e}", "sigma_f": f"{comb['sigma_f']:.4e}",
                             "snr": f"{comb['snr']:.2f}", "chi2": f"{comb['chi2']:.1f}", "dof_chi2": 2 * (len(items) - 1),
                             "f_limit": f"{flim:.4e}", "spt_at_limit": lim["spt"][0] or lim["beyond"][0],
                             "mass_at_limit": f"{lim['mass'][0]:.3f}" if np.isfinite(lim["mass"][0]) else "",
                             "snr_null": f"{null['snr']:.2f}"})
    write_csv(base + "_limits.csv", lim_rows)

    lines = [f"PSF-shape fit: sigma_w {a.sigma_w} px, spins {a.spins}, drift order {a.drift}, injected flux ratio {a.inject}, "
             f"{a.n_phi} angles per separation; detection threshold {t1:.2f} x sigma per component "
             "(false-alarm probability of a one-sided 5 sigma); sense +1 = IDL derotation, sense -1 = null", ""]
    for night, sets in sorted(rot.items()):
        lines.append(f"{night}:")
        for lab, n, qmin, qmax, span in sets:
            lines.append(f"  {lab}: {n} frames, q {qmin:.1f} .. {qmax:.1f} deg (rotation {span:.1f} deg)")
    lines.append("")
    c_k7 = float(cp.absmag_to_contrast(seq.m_w3_of("K7V"), a.star_mag, a.parallax))
    c_k0 = float(cp.absmag_to_contrast(seq.m_w3_of("K0V"), a.star_mag, a.parallax))
    lines.append(f"K0V-K7V companions: flux ratio {c_k7:.4f}-{c_k0:.4f} at N'")
    lines.append("night, spin, separation: flux-ratio limit (companion at the limit) | estimate +/- sigma, SNR | null SNR | chi2/dof")
    for r in lim_rows:
        lines.append(f"  {r['night']} spin {r['spin']} {float(r['sep_mas']):5.0f} mas ({float(r['sep_au']):.2f} au): "
                     f"{float(r['f_limit']):.4f} ({r['spt_at_limit']}, {r['mass_at_limit']} Msun) | "
                     f"{float(r['f_est']):.4f} +/- {float(r['sigma_f']):.4f}, SNR {r['snr']} | null {r['snr_null']} | "
                     f"{r['chi2']}/{r['dof_chi2']}")
    with open(base + "_summary.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    if not a.no_plot:
        plot(lim_rows, rot, a, seq, base + "_limits")
    print(f"wrote {base}_fits.csv, _response.csv, _limits.csv, _summary.txt" + ("" if a.no_plot else ", _limits.png/.pdf"))
    return 0


def plot(lim_rows: List[dict], rot: Dict, a: argparse.Namespace, seq: cp.DwarfSequence, out_base: str) -> None:
    """Draw the flux-ratio limit against separation for each night and spin.

    Parameters
    ----------
    lim_rows : list of dict
        Rows of ``<name>_limits.csv``.
    rot : dict
        Rotation per night: list of (label, frames, q_min, q_max, span) per frame set.
    a : argparse.Namespace
        Command-line settings (``parallax``, ``star_mag``, ``sigma_w``, ``name``).
    seq : DwarfSequence
        Dwarf sequence in the band.
    out_base : str
        Output path without extension.

    Returns
    -------
    None
        ``<out_base>.png`` and ``<out_base>.pdf`` are written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    to_c = lambda spt: float(cp.absmag_to_contrast(seq.m_w3_of(spt), a.star_mag, a.parallax))  # noqa: E731
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.84, bottom=0.25)
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.axhspan(to_c("K7V"), to_c("K0V"), color=MUTED, alpha=0.18, lw=0)
    ax.text(21, np.sqrt(to_c("K7V") * to_c("K0V")), "K0V–K7V", ha="left", va="center", fontsize=8.5, color=INK2)
    for spt in ("M0V", "M2V", "M4V"):
        ax.axhline(to_c(spt), color=MUTED, lw=0.8, ls=(0, (4, 3)))
        ax.text(21, to_c(spt) * 1.04, spt, ha="left", va="bottom", fontsize=8, color=INK2)
    x05 = float(cp.arcsec_of_au(0.5, a.parallax)) * 1000
    ax.axvline(x05, color=INK, lw=1.0, ls=(0, (1, 2)))
    ax.text(x05 * 1.03, 0.62, f"0.5 au ({x05:.0f} mas)", rotation=90, ha="left", va="top", fontsize=8.5, color=INK)
    nights = sorted({r["night"] for r in lim_rows})
    for k, night in enumerate(nights):
        span = max(s[4] for s in rot[night])
        for m, ls in ((1, "-"), (2, (0, (5, 2.5)))):
            rr = [r for r in lim_rows if r["night"] == night and int(r["spin"]) == m]
            if not rr:
                continue
            x = [float(r["sep_mas"]) for r in rr]
            y = [float(r["f_limit"]) for r in rr]
            ax.plot(x, y, color=SLOTS[k % len(SLOTS)], ls=ls, lw=1.8 if m == 1 else 1.3, marker=MARKERS[k % len(MARKERS)],
                    ms=5, mfc=SLOTS[k % len(SLOTS)] if m == 1 else "white", mec=SLOTS[k % len(SLOTS)],
                    label=f"{night[-8:-4]}-{night[-4:-2]}-{night[-2:]}, Spin {m} ({'Lopsidedness' if m == 1 else 'Elongation'}), "
                          f"{span:.0f}° Rotation")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(20, 300)
    ax.set_ylim(1e-3, 1.0)
    ax.set_xticks([20, 30, 50, 65, 100, 150, 200, 300])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Angular Separation (mas)", fontsize=10, color=INK)
    ax.set_ylabel("Flux Ratio That Would Be Detected at N′", fontsize=10, color=INK)
    top = ax.secondary_xaxis("top", functions=(lambda x: cp.projected_au(x / 1000.0, a.parallax),
                                               lambda u: cp.arcsec_of_au(u, a.parallax) * 1000.0))
    top.set_xlabel("Projected Separation (au)", fontsize=10, color=INK)
    top.set_xticks([0.2, 0.3, 0.5, 1, 2])
    top.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_title(f"Companions Inside λ/D from the PSF Shape ({a.name})", fontsize=11, color=INK, pad=34)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.12), ncol=1, fontsize=8, frameon=False)
    fig.text(0.1, 0.01, f"Limit: {ps.rayleigh_threshold():.2f}σ per component of the turning term (false-alarm "
             f"probability of a one-sided 5σ); weight σ = {a.sigma_w:g} px; responses from injected copies of each frame's PSF.",
             fontsize=7.5, color=INK2)
    fig.savefig(out_base + ".png", dpi=200, facecolor="white")
    fig.savefig(out_base + ".pdf", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
