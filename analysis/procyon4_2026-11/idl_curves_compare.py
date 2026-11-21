"""Compare Kevin's IDL contrast curves of one night at a common S/N and pick the deepest.

usage: python analysis/procyon4_2026-11/idl_curves_compare.py OUT_CSV SNR_REF LABEL:SNR:PATH [LABEL:SNR:PATH ...]

Each curve is a PDF plotted by contrastcurve_nomic.pro (read with ``idl_curve_pdf.py``: the
black line, the mean over position angles) or the ``*_contrast_output.sav`` it saves (the mean
over position angles of ``curves``, NaN left out). A curve that targeted S/N ``SNR`` is scaled to
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
    The table, and per curve the number of separations where it is the deepest.
"""
import csv
import os
import sys
from collections import Counter

import numpy as np
from scipy.io import readsav

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from idl_curve_pdf import read_curves  # noqa: E402


def load(path: str) -> dict:
    """Return a curve as {separation (arcsec, 2 decimals): mean contrast over position angles}.

    Parameters
    ----------
    path : str
        A contrast-curve PDF or IDL save file.

    Returns
    -------
    curve : dict
        Separation to contrast, at the S/N the curve targeted.
    """
    if path.lower().endswith(".sav"):
        s = readsav(path, python_dict=True)
        c, sep, pa = np.asarray(s["curves"], float), np.asarray(s["separations"], float), np.asarray(s["pas"])
        if c.shape != (pa.size, sep.size):
            c = c.T
        return {round(float(r), 2): float(np.nanmean(c[:, j])) for j, r in enumerate(sep)}
    return {round(float(r), 2): float(v) for r, v in read_curves(path)["mean"]}


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
    curves = []
    for spec in args[2:]:
        label, snr, path = spec.split(":", 2)
        k = ref / float(snr)
        curves.append((label, {s: v * k for s, v in load(path).items()}))
    seps = sorted({s for _, c in curves for s in c})
    wins = Counter()
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["separation_arcsec"] + [lab for lab, _ in curves] + ["deepest", "deepest_curve"])
        print("sep(\")  " + "  ".join(f"{lab:>12s}" for lab, _ in curves) + "  deepest")
        for s in seps:
            vals = [c.get(s) for _, c in curves]
            have = [(v, lab) for v, (lab, _) in zip(vals, curves) if v is not None]
            best, who = min(have)
            wins[who] += 1
            w.writerow([f"{s:.2f}"] + ["" if v is None else f"{v:.4g}" for v in vals] + [f"{best:.4g}", who])
            print(f"{s:5.2f}  " + "  ".join(f"{'':>12s}" if v is None else f"{v:12.3g}" for v in vals)
                  + f"  {best:.3g} ({who})")
    print("deepest at: " + ", ".join(f"{lab} {wins[lab]}" for lab, _ in curves) + f" of {len(seps)} separations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
