#!/usr/bin/env python3
"""Replay the split stage's cut positions of finished runs under the old and the new rules.

usage: python3 tools/replay_split_positions.py OUT_CSV SUMMARY_JSON [SUMMARY_JSON ...]

Each run's summary holds the star positions the split stage found per chunk (``positions``).
For both chop positions this recomputes where each chunk was cut before the change of
2026-11-20 (IDL's edge-check bounds [radius, 2 radius], no outlier filter) and where it is cut
now (``geometry.split_cut_positions`` with the night's settings), so that the runs whose products
change can be listed without rerunning them. The night's settings are the defaults, its IDL object
block (``config/nights_idl.json``) and the port's per-night decisions (``config/nights_port.json``),
looked up by the summary's ``night_old``.

Parameters
----------
OUT_CSV : str
    Report written, one row per run and chop; an existing file is never overwritten.
SUMMARY_JSON : str list
    ``<night>_<side>_summary.json`` files of finished runs.

Returns
-------
OUT_CSV : file
    Columns ``summary``, ``night_old``, ``side``, ``chop`` (1 = A, 2 = B), ``chunks``, ``radius``,
    ``framew``, ``old_far`` (chunks whose old cut was more than 20 px from the found star),
    ``old_max_px`` (largest such distance), ``new_far`` and ``new_max_px`` (the same for the new
    cut, measured from the star as found), ``replaced`` (positions the outlier filter replaced) and
    ``changed`` (chunks whose cut moves by more than 2 px).
stdout : text
    The rows with any old or new distance above 20 px, or any changed chunk.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from typing import Dict, List

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from reduce_nomic import config, geometry  # noqa: E402


def night_settings(old: str) -> Dict:
    """Merge the split settings of one night.

    Parameters
    ----------
    old : str
        The night's folder name before the 2026-11-20 renaming.

    Returns
    -------
    st : dict
        ``config.DEFAULTS`` and ``config.PORT_DEFAULTS``, updated with the night's IDL object block
        (values that are not None) and its port decisions.
    """
    st = dict(config.DEFAULTS)
    st.update(config.PORT_DEFAULTS)
    with open(os.path.join(HERE, "config", "nights_idl.json")) as fh:
        idl = json.load(fh)
    with open(os.path.join(HERE, "config", "nights_port.json")) as fh:
        port = json.load(fh)
    st.update({k: v for k, v in idl.get(old, {}).get("settings", {}).items() if v is not None})
    st.update(port.get(old, {}).get("settings", {}))
    return st


def main(argv: List[str]) -> int:
    """Write the replay report.

    Parameters
    ----------
    argv : list of str
        ``[OUT_CSV, SUMMARY_JSON, ...]``.

    Returns
    -------
    status : int
        0 on success.
    """
    if len(argv) < 2:
        raise SystemExit(__doc__.split("\n\n")[1])
    out, paths = argv[0], argv[1:]
    rows = []
    for p in paths:
        with open(p) as fh:
            s = json.load(fh)
        raw = s.get("positions") or []
        if not raw:
            continue
        old = s.get("night_old") or s.get("night")
        st = night_settings(old)
        _, _, cut_old = geometry.split_cut_positions(raw, dict(st, edgecheck_idl_bounds=True, pos_outlier_px=0.0))
        _, replaced, cut_new = geometry.split_cut_positions(raw, dict(st, edgecheck_idl_bounds=False))
        for cn in (1, 2):
            found = np.asarray([r[cn - 1] for r in raw], float)
            d_old = np.max(np.abs(np.asarray(cut_old[cn], float) - found), axis=1)
            d_new = np.max(np.abs(np.asarray(cut_new[cn], float) - found), axis=1)
            moved = np.max(np.abs(np.asarray(cut_new[cn], float) - np.asarray(cut_old[cn], float)), axis=1)
            rows.append({"summary": os.path.basename(p), "night_old": old, "side": s.get("side"), "chop": cn,
                         "chunks": len(raw), "radius": st["radius"], "framew": st["framew"],
                         "old_far": int(np.sum(d_old > 20)), "old_max_px": float(d_old.max()),
                         "new_far": int(np.sum(d_new > 20)), "new_max_px": float(d_new.max()),
                         "replaced": sum(1 for c, k in replaced if k == cn), "changed": int(np.sum(moved > 2))})
    with open(out, "x", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        if r["old_far"] or r["new_far"] or r["changed"]:
            print(r)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
