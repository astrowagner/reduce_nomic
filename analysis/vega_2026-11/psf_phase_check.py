"""Print the complex spin-m turning term of every PSF-shape cutout at one separation, in flux-ratio units.

usage: python psf_phase_check.py CUTOUT_DIR RESPONSE_CSV [--sep-mas 65] [--spin 2] [--drift 1]
       [--sigma-w 3] [--cutouts NAME ...]

A sky-fixed companion gives the same complex b_k / r_k, amplitude and phase, in every frame set of
one night (both apertures and both chop positions see the same sky at the same parallactic angle),
and nearly the same on consecutive nights when its orbit is slow; the sense -1 term (the null, which
no sky-fixed source follows) gives noise only. This prints both for every cutout, so that agreement
in phase as well as amplitude can be judged. It generalizes ``psf_phase_check_0521.py`` (2022-05-21
only) to any cutouts and response table.

Parameters
----------
CUTOUT_DIR : str
    Folder of ``<night>_<side>_cube<n>_clean_core96.fits`` files from
    ``tools/extract_psf_cutouts.py``.
RESPONSE_CSV : str
    ``<name>_response.csv`` from ``tools/psf_shape_fit.py``, run on the same cutouts with the same
    Gaussian weight and drift order; its response r (moment per unit flux ratio) at ``--sep-mas``
    and ``--spin`` scales each cutout's turning term.
--sep-mas : float, default 65
    Separation whose response is used, mas (65 mas = 0.5 au for Vega).
--spin : int, default 2
    Moment spin (1 lopsidedness, 2 elongation).
--drift : int, default 1
    Polynomial order of the drift in frame order, as in the response table's run.
--sigma-w : float, default 3.0
    Width of the Gaussian weight, pixels, as in the response table's run.
--cutouts : str list, optional
    Cutout file names to use (default every ``*_core96.fits`` in CUTOUT_DIR whose label is in the
    response table).

Returns
-------
stdout : text
    One line per cutout: frames used, and for sense +1 and -1 the amplitude |b / r| (flux ratio),
    its phase (degrees) and its 1σ error (flux ratio).
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))
import psf_shape_fit as t  # noqa: E402
from reduce_nomic import psfshape as ps  # noqa: E402
from reduce_nomic.angles import wrap_outliers  # noqa: E402


def main(argv):
    """Print the per-cutout turning terms.

    Parameters
    ----------
    argv : sequence of str
        Command-line arguments without the program name (see the module docstring).

    Returns
    -------
    status : int
        0 when at least one cutout was measured, 1 otherwise.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cutout_dir")
    ap.add_argument("response_csv")
    ap.add_argument("--sep-mas", type=float, default=65.0)
    ap.add_argument("--spin", type=int, default=2)
    ap.add_argument("--drift", type=int, default=1)
    ap.add_argument("--sigma-w", type=float, default=3.0)
    ap.add_argument("--cutouts", nargs="*", default=None)
    a = ap.parse_args(argv)
    resp = {}
    for r in csv.DictReader(open(a.response_csv)):
        if int(r["spin"]) == a.spin and abs(float(r["sep_mas"]) - a.sep_mas) < 1e-6:
            resp[r["cutout"]] = float(r["r"])
    files = ([os.path.join(a.cutout_dir, c) for c in a.cutouts] if a.cutouts
             else sorted(glob.glob(os.path.join(a.cutout_dir, "*_core96.fits"))))
    print(f"spin {a.spin}, {a.sep_mas:g} mas, drift order {a.drift}, sigma_w {a.sigma_w} px: b / r in flux-ratio units")
    done = 0
    for path in files:
        c = t.load_cutout(path)
        if c["label"] not in resp:
            continue
        bad = wrap_outliers(c["q"])
        keep = np.setdiff1d(np.arange(c["q"].size), bad)
        fr, q = c["frames"][keep], c["q"][keep]
        n, ny, nx = fr.shape
        bg = (min(ny, nx) * 0.36, min(ny, nx) * 0.49)
        z = ps.weighted_moments(fr, a.sigma_w, c["x0"], c["y0"], bg_annulus=bg)[f"spin{a.spin}"]
        r = resp[c["label"]]
        out = []
        for sense in (+1, -1):
            f = ps.fit_rotating(z, q, a.spin, sense, a.drift)
            v = f.b / r
            out.append(f"sense {sense:+d}: {abs(v):.3f} at {np.degrees(np.angle(v)):7.1f} deg (+/- {f.sigma_b / r:.3f})")
        print(f"   {c['label']:28s} {n:4d} frames  " + " | ".join(out))
        done += 1
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
