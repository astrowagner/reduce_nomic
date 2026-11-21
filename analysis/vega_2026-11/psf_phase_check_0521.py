"""Phases of the per-set spin-2 turning terms of Vega 2022-05-21 at 65 mas (sense +1 and -1, drift 1 and 2).

A sky-fixed companion gives the same complex b_k / r_k in every set of one night; this prints each
set's value (amplitude in flux-ratio units, phase, error), so that agreement in phase as well as
amplitude can be checked. Frames whose angle was averaged across the ±180° wrap are left out.

usage: python psf_phase_check_0521.py [CUTOUT_DIR]
    run in the folder holding ``vega_psf_20220521_w3_d1_response.csv`` and ``..._d2_response.csv``
    from ``tools/psf_shape_fit.py`` (Gaussian weight 3 px), whose 65 mas responses r_k are used.
"""
import csv
import sys

import numpy as np

sys.path.insert(0, "/home/claude/reduce_nomic")
sys.path.insert(0, "/home/claude/reduce_nomic/tools")
import psf_shape_fit as t  # noqa: E402
from reduce_nomic import psfshape as ps  # noqa: E402
from reduce_nomic.angles import wrap_outliers  # noqa: E402

C = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/NOMIC/_analysis/vega_2026-11/psf_cutouts"
names = ["Vega_20220521_left_cube1", "Vega_20220521_right_cube1", "Vega_20220521_right_cube2"]
for drift in (1, 2):
    resp = {r["cutout"]: float(r["r"]) for r in csv.DictReader(open(f"vega_psf_20220521_w3_d{drift}_response.csv"))
            if r["spin"] == "2" and r["sep_mas"] == "65"}
    print(f"drift order {drift}: b/r at 65 mas (flux-ratio units), amplitude and phase")
    for nm in names:
        c = t.load_cutout(f"{C}/{nm}_clean_core96.fits")
        bad = wrap_outliers(c["q"])
        keep = np.setdiff1d(np.arange(c["q"].size), bad)
        fr, q = c["frames"][keep], c["q"][keep]
        n, ny, nx = fr.shape
        bg = (min(ny, nx) * 0.36, min(ny, nx) * 0.49)
        z = ps.weighted_moments(fr, 3.0, c["x0"], c["y0"], bg_annulus=bg)["spin2"]
        r = resp[c["label"]]
        out = []
        for sense in (+1, -1):
            f = ps.fit_rotating(z, q, 2, sense, drift)
            v = f.b / r
            out.append(f"sense {sense:+d}: {abs(v):.3f} at {np.degrees(np.angle(v)):7.1f} deg (+/- {f.sigma_b / r:.3f})")
        print(f"   {c['label']:28s} " + " | ".join(out))
