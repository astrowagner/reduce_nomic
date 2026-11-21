"""Derive photospheric Ks - W3 colors of K5V to M9.5V dwarfs from the Cifuentes et al. (2020) catalog.

usage: python tools/derive_ks_w3_colors.py MOTHER_CSV OUT_CSV [--sigma 2.5] [--source-note TEXT]

Mamajek's dwarf sequence (``reduce_nomic/data/EEM_dwarf_UBVIJHK_colors_Teff.txt``) gives W1 - W3 only
through K5V, so the N'-band (W3) magnitudes of later dwarfs need another source. ``MOTHER_CSV`` is
``cif20.Mother.v01.csv`` from github.com/ccifuentesr/CARMENES-V, the full Table A.3 of Cifuentes
et al. (2020, A&A 642, A115): photometry and Gaia DR2 distances of 2483 K5V to L8 stars. Stars are
kept with the paper's quality cuts for its Tables 6 and 7 (``Bool_delta``, ``Bool_young``,
``Bool_dphot`` and ``Bool_RUWE`` all false: no resolved companion within 5 arcsec, not
overluminous, consistent photometry, good astrometry), with clean Ks, W1 and W3 photometry
(magnitude and error given, ``Qf_<band>`` false), and without a flagged infrared excess
(``Bool_excess`` false). For each spectral subtype the colors are sigma-clipped about the median
and their median and standard deviation written out. Distances do not enter, since only colors
are used.

Parameters
----------
MOTHER_CSV : str
    Path of ``cif20.Mother.v01.csv``.
OUT_CSV : str
    Output table. Comment lines (``#``) record the input's SHA-256, the cuts and the clipping.
--sigma : float, default 2.5
    Clipping threshold in standard deviations about the median, as in the paper.
--source-note : str, default ""
    Free text added to the header, e.g. the CARMENES-V commit the file came from.

Returns
-------
OUT_CSV : file
    One row per subtype with at least one star: ``spt`` (``K5V``, ``M3.5V``), ``sptnum``
    (catalog code: K5 = -2, K7 = -1, M0.0 = 0, ..., M9.5 = 9.5, L0 = 10), ``n`` (stars kept),
    ``ks_w3`` and ``ks_w3_sd`` (median and standard deviation of Ks - W3, mag), ``w1_w3`` and
    ``ks_w1`` (medians, mag), ``n_clipped`` (stars left after clipping Ks - W3).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from datetime import datetime, timezone
from typing import Dict, List, Sequence, Tuple

import numpy as np
from astropy.stats import sigma_clip

#: Catalog flags that must all be ``false`` for a star to be used (the paper's Tables 6 and 7).
QUALITY_FLAGS = ("Bool_delta", "Bool_young", "Bool_dphot", "Bool_RUWE")


def sptnum_name(sptnum: float) -> str:
    """Name a subtype from its catalog code.

    Parameters
    ----------
    sptnum : float
        Catalog code: K5 = -2, K7 = -1, M0.0 = 0 to M9.5 = 9.5, L0 = 10 and on.

    Returns
    -------
    name : str
        ``K5V``, ``K7V``, ``M0V``, ``M3.5V``, ``L1.5`` (L types carry no luminosity class).
    """
    if sptnum == -2:
        return "K5V"
    if sptnum == -1:
        return "K7V"
    if sptnum < 10:
        return f"M{sptnum:g}V"
    return f"L{sptnum - 10:g}"


def clean_row(row: Dict[str, str]) -> bool:
    """Tell whether a catalog row passes the quality and photometry cuts.

    Parameters
    ----------
    row : dict of str to str
        One row of the catalog as read by ``csv.DictReader``.

    Returns
    -------
    ok : bool
        True when every flag in ``QUALITY_FLAGS`` and ``Bool_excess`` is ``false`` and Ks, W1 and
        W3 each have a magnitude, an error and ``Qf_<band>`` equal to ``false``.
    """
    if any(row[f] != "false" for f in QUALITY_FLAGS) or row["Bool_excess"] == "true":
        return False
    return all(row[f"{b}_mag"] != "" and row[f"e{b}_mag"] != "" and row[f"Qf_{b}"] == "false"
               for b in ("Ks", "W1", "W3"))


def subtype_colors(rows: Sequence[Dict[str, str]], sigma: float = 2.5) -> List[dict]:
    """Median photospheric colors per subtype.

    Parameters
    ----------
    rows : sequence of dict of str to str
        Catalog rows (all of them; the cuts are applied here).
    sigma : float, default 2.5
        Clipping threshold in standard deviations about the median (up to 5 iterations).

    Returns
    -------
    table : list of dict
        One entry per subtype with stars left, ordered by ``sptnum``, with the keys described for
        ``OUT_CSV`` in the module docstring.
    """
    groups: Dict[float, List[Tuple[float, float, float]]] = {}
    for r in rows:
        if not clean_row(r):
            continue
        ks, w1, w3 = (float(r[f"{b}_mag"]) for b in ("Ks", "W1", "W3"))
        groups.setdefault(float(r["SpTnum"]), []).append((ks - w3, w1 - w3, ks - w1))
    table = []
    for s in sorted(groups):
        a = np.asarray(groups[s])
        clipped = [sigma_clip(a[:, k], sigma=sigma, cenfunc="median", maxiters=5) for k in range(3)]
        table.append({"spt": sptnum_name(s), "sptnum": s, "n": len(a),
                      "ks_w3": float(np.ma.median(clipped[0])), "ks_w3_sd": float(np.ma.std(clipped[0])),
                      "w1_w3": float(np.ma.median(clipped[1])), "ks_w1": float(np.ma.median(clipped[2])),
                      "n_clipped": int(clipped[0].count())})
    return table


def main(argv: Sequence[str]) -> int:
    """Run the derivation from the command line.

    Parameters
    ----------
    argv : sequence of str
        Command-line arguments without the program name (see the module docstring).

    Returns
    -------
    status : int
        0 on success. ``OUT_CSV`` is written.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mother_csv")
    ap.add_argument("out_csv")
    ap.add_argument("--sigma", type=float, default=2.5)
    ap.add_argument("--source-note", default="")
    a = ap.parse_args(argv)
    with open(a.mother_csv, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    with open(a.mother_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    table = subtype_colors(rows, a.sigma)
    keys = ["spt", "sptnum", "n", "n_clipped", "ks_w3", "ks_w3_sd", "w1_w3", "ks_w1"]
    with open(a.out_csv, "w", newline="") as fh:
        fh.write("# Photospheric Ks - W3 colors of K5V-M9.5V dwarfs from Cifuentes et al. (2020, A&A 642, A115)\n")
        fh.write(f"# made by tools/derive_ks_w3_colors.py on {datetime.now(timezone.utc):%Y-%m-%d} from "
                 f"cif20.Mother.v01.csv (sha256 {digest})\n")
        if a.source_note:
            fh.write(f"# source: {a.source_note}\n")
        fh.write(f"# cuts: {', '.join(QUALITY_FLAGS)} and Bool_excess false; Ks, W1, W3 with magnitude, "
                 f"error and Qf false; {a.sigma:g}-sigma clipping about the median (5 iterations)\n")
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for t in table:
            w.writerow({k: (f"{t[k]:.4f}" if isinstance(t[k], float) and k not in ("sptnum",) else t[k])
                        for k in keys})
    print(f"wrote {a.out_csv}: {len(table)} subtypes from {sum(t['n'] for t in table)} stars")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
