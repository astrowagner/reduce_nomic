"""Compare Kevin's IDL contrast curves of one night at a common S/N and pick the deepest.

usage: python analysis/procyon4_2026-11/idl_curves_compare.py OUT_CSV SNR_REF LABEL:SNR:PATH [LABEL:SNR:PATH ...]

Each curve is a PDF plotted by contrastcurve_nomic.pro (read with ``idl_curve_pdf.py``: the
black line, the mean over position angles) or the ``*_contrast_output.sav`` it saves (the mean
over position angles of ``curves``). In a save file only finite, positive contrasts count: NaN is a
test not recovered within the trial cap and 0 a test that failed (Procyon_3's Dec 2025 curve has 0
at 19 of 20 position angles at 0.40 and 0.50"), and a separation with none is left out. Where a
mean covers fewer than all position angles, the printed table gives the count, since the
position angles left out are the hardest. A curve that targeted S/N ``SNR`` is scaled to
``SNR_REF`` by ``SNR_REF / SNR``: the injection-recovery measurement is linear in the injected
flux, so the contrast at a threshold scales with the threshold. The table gives every curve at
every separation any of them measured, and the deepest value with its curve.

Parameters
----------
OUT_CSV : str
    Table to write: separation (arcsec), one column per curve (contrast at SNR_REF, empty where
    the curve has no point), the deepest contrast and its curve's label.
SNR_REF : float
    The common S/N, e.g. 5.
LABEL:SNR:PATH : str
    A curve: a short label, the S/N it targeted and its PDF or save file.

Returns
-------
stdout : text
    The table, with ``[n/N]`` after a save file's mean over n of its N position angles, and per
    curve the number of separations where it is the deepest.
OUT_CSV : file
    The table (values only).
"""
import csv
import os
import sys
from collections import Counter
from typing import Dict, Tuple

import numpy as np
from scipy.io import readsav

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from idl_curve_pdf import read_curves  # noqa: E402


def load(path: str) -> Tuple[Dict[float, float], Dict[float, Tuple[int, int]]]:
    """Return a curve's mean contrast over position angles at each separation.

    Parameters
    ----------
    path : str
        A contrast-curve PDF or IDL save file.

    Returns
    -------
    curve : dict of float to float
        Separation (arcsec, 2 decimals) to contrast, at the S/N the curve targeted; a save file's
        separations without a finite, positive contrast are left out.
    counts : dict of float to tuple of 2 int
        For a save file, per separation in ``curve``, the number of position angles averaged and
        the number there are; empty for a PDF.
    """
    if path.lower().endswith(".sav"):
        s = readsav(path, python_dict=True)
        c, sep, pa = np.asarray(s["curves"], float), np.asarray(s["separations"], float), np.asarray(s["pas"])
        if c.shape != (pa.size, sep.size):
            c = c.T
        curve, counts = {}, {}
        for j, r in enumerate(sep):
            ok = np.isfinite(c[:, j]) & (c[:, j] > 0)
            if ok.any():
                curve[round(float(r), 2)] = float(np.mean(c[ok, j]))
                counts[round(float(r), 2)] = (int(ok.sum()), int(pa.size))
        return curve, counts
    return {round(float(r), 2): float(v) for r, v in read_curves(path)["mean"]}, {}


def main(argv=None) -> int:
    """Build, write and print the comparison table.

    Parameters
    ----------
    argv : list of str or None, default None
        OUT_CSV SNR_REF LABEL:SNR:PATH ...; None reads ``sys.argv[1:]``.

    Returns
    -------
    status : int
        0.
    """
    args = argv or sys.argv[1:]
    out, ref = args[0], float(args[1])
    curves, counts = [], {}
    for spec in args[2:]:
        label, snr, path = spec.split(":", 2)
        k = ref / float(snr)
        curve, cnt = load(path)
        curves.append((label, {s: v * k for s, v in curve.items()}))
        counts[label] = cnt
    seps = sorted({s for _, c in curves for s in c})
    wins = Counter()

    def _cell(lab: str, s: float, v) -> str:
        """Format one printed cell, with ``[n/N]`` where fewer than all position angles were averaged."""
        if v is None:
            return f"{'':>18s}"
        n = counts[lab].get(s)
        tag = f" [{n[0]}/{n[1]}]" if n and n[0] < n[1] else ""
        return f"{f'{v:.3g}{tag}':>18s}"

    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["separation_arcsec"] + [lab for lab, _ in curves] + ["deepest", "deepest_curve"])
        print("sep(\")  " + "  ".join(f"{lab:>18s}" for lab, _ in curves) + "  deepest")
        for s in seps:
            vals = [c.get(s) for _, c in curves]
            have = [(v, lab) for v, (lab, _) in zip(vals, curves) if v is not None]
            best, who = min(have)
            wins[who] += 1
            w.writerow([f"{s:.2f}"] + ["" if v is None else f"{v:.4g}" for v in vals] + [f"{best:.4g}", who])
            print(f"{s:5.2f}  " + "  ".join(_cell(lab, s, v) for v, (lab, _) in zip(vals, curves))
                  + f"  {best:.3g} ({who})")
    print("deepest at: " + ", ".join(f"{lab} {wins[lab]}" for lab, _ in curves) + f" of {len(seps)} separations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
