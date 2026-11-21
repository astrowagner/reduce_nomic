"""Summarize an IDL contrast-curve save file per separation.

usage: python analysis/procyon4_2026-11/idl_curve_summary.py SAV

Reads the ``curves`` (position angle by separation), ``separations`` (arcsec), ``pas`` (degrees)
and ``threshold`` (S/N) of a ``*_contrast_output.sav`` written by the IDL contrast-curve code and
prints, per separation, the median, smallest and largest contrast over the position angles and the
number of position angles with a finite value (a position angle off the detector has none).

Parameters
----------
SAV : str
    The IDL save file, e.g. ``Procyon_20211220/IDL/contrast/Procyon_4_contrast_output.sav``.

Returns
-------
stdout : text
    The threshold, the position angles, and one line per separation.
"""
import sys

import numpy as np
from scipy.io import readsav


def main(argv=None) -> int:
    """Print the summary.

    Parameters
    ----------
    argv : list of str or None, default None
        SAV; None reads ``sys.argv[1:]``.

    Returns
    -------
    status : int
        0.
    """
    s = readsav((argv or sys.argv[1:])[0], python_dict=True)
    c = np.asarray(s["curves"], float)
    sep = np.asarray(s["separations"], float)
    pa = np.asarray(s["pas"], float)
    if c.shape != (pa.size, sep.size):
        c = c.T
    print(f"threshold S/N {float(s['threshold']):g}; {pa.size} position angles {pa.min():g}-{pa.max():g} deg")
    for j, r in enumerate(sep):
        v = c[:, j]
        f = np.isfinite(v)
        print(f"{r:4.2f}\": median {np.median(v[f]):.3g}, min {v[f].min():.3g}, max {v[f].max():.3g}, "
              f"finite at {f.sum()} of {pa.size} position angles")
    return 0


if __name__ == "__main__":
    sys.exit(main())
