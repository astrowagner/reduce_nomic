"""Compare two KLIP-TPE runs' final-image contrast curves separation by separation.

usage: python analysis/procyon3-4_2026-11/compare_final_curves.py RUN_A RUN_B [--labels A B]

Each run's curve is its final-image grid's (``final_grid/curve.txt``): the median contrast at S/N 5
over position angle at each separation, with its 16th and 84th percentiles. The two runs are
compared at the separations both grids hold (to 0.001 arcsec); a ratio A / B below 1 means run A
goes deeper there.

Parameters
----------
RUN_A : str
    The first finished run directory.
RUN_B : str
    The second finished run directory.
--labels : str, default "A B"
    Names of the two runs in the table's header.

Returns
-------
stdout : text
    One row per common separation: the separation (arcsec), each run's median with its 16th-84th
    percentile range, the ratio of the medians A / B and the depth gained by A, 1 - A / B, in
    percent.
"""
import argparse
import os
import sys
from typing import Dict, Optional, Sequence

import numpy as np


def read_curve(run_dir: str) -> Dict[float, tuple]:
    """Read a finished run's final-image curve.

    Parameters
    ----------
    run_dir : str
        The run directory, holding ``final_grid/curve.txt``.

    Returns
    -------
    curve : dict of float to tuple of float
        Separation rounded to 0.001 arcsec mapped to (median, p16, p84), contrasts at S/N 5.
    """
    out = {}
    with open(os.path.join(run_dir, "final_grid", "curve.txt")) as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                p = s.split()
                out[round(float(p[0]), 3)] = (float(p[1]), float(p[2]), float(p[3]))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Read both curves and print the comparison.

    Parameters
    ----------
    argv : sequence of str or None, default None
        The command-line arguments (see the module's usage); None reads ``sys.argv``.

    Returns
    -------
    status : int
        0 on success, 1 when the grids share no separation.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--labels", nargs=2, default=["A", "B"])
    a = ap.parse_args(argv)
    ca, cb = read_curve(a.run_a), read_curve(a.run_b)
    common = sorted(set(ca) & set(cb))
    if not common:
        print("the two grids share no separation", file=sys.stderr)
        return 1
    la, lb = a.labels
    print(f"A = {la}: {a.run_a}")
    print(f"B = {lb}: {a.run_b}")
    print("final-image curves, median contrast at S/N 5 over position angle (16th-84th percentile)")
    print(f"{'sep':>6}  {'A median':>9} {'(p16-p84)':>19}  {'B median':>9} {'(p16-p84)':>19}  "
          f"{'A/B':>5}  {'gain':>6}")
    ratios = []
    for s in common:
        ma, a16, a84 = ca[s]
        mb, b16, b84 = cb[s]
        r = ma / mb
        ratios.append(r)
        print(f"{s:6.3f}  {ma:9.3e} ({a16:8.2e}-{a84:8.2e})  {mb:9.3e} ({b16:8.2e}-{b84:8.2e})  "
              f"{r:5.2f}  {100 * (1 - r):+5.0f}%")
    print(f"{len(common)} common separations; A/B from {min(ratios):.2f} to {max(ratios):.2f}, "
          f"median {float(np.median(ratios)):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
