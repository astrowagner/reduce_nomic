"""Spin-2 turning term of Vega 2022-05-21 against the drift model and the weight width.

For each frame set and drift order 1-4 (Gaussian weight 3 and 4 px), fits the spin-2 moment with a
term turning in the sense of the IDL derotation (+1) and the opposite sense (-1, a null), and with
terms following other rotation laws q' = alpha q (alpha = -3 ... 3, nulls whose temporal frequency
differs from the companion's). Prints |b| / sigma for each, the median and maximum of the null
laws, |b|, sigma and the lag-1 autocorrelation of the residuals. Frames whose angle was averaged
across the ±180° wrap are left out.

usage: python psf_drift_study_0521.py CUTOUT_DIR
"""
import sys

import numpy as np
from astropy.io import fits

sys.path.insert(0, "/home/claude/reduce_nomic")
from reduce_nomic import psfshape as ps  # noqa: E402
from reduce_nomic.angles import wrap_outliers  # noqa: E402

D = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/NOMIC/_analysis/vega_2026-11/psf_cutouts"
names = ["Vega_20220521_left_cube1", "Vega_20220521_right_cube1", "Vega_20220521_right_cube2"]
data = {}
for nm in names:
    with fits.open(f"{D}/{nm}_clean_core96.fits") as hd:
        fr = hd[0].data.astype(float)
        q = hd["ANGLES"].data.astype(float)
        keep = np.setdiff1d(np.arange(q.size), wrap_outliers(q))
        data[nm] = (fr[keep], q[keep], hd[0].header["XCEN"], hd[0].header["YCEN"])
alphas = [-3, -2, -1.5, -1, -0.5, 0.5, 1.5, 2, 3]
for sw in (3.0, 4.0):
    print(f"\nsigma_w = {sw} px")
    for nm in names:
        fr, q, x0, y0 = data[nm]
        m = ps.weighted_moments(fr, sw, x0, y0, bg_annulus=(35, 47))
        z = m["spin2"]
        qu = np.rad2deg(np.unwrap(np.deg2rad(q)))
        line = []
        for d in (1, 2, 3, 4):
            fp = ps.fit_rotating(z, q, 2, +1, d)
            fn = ps.fit_rotating(z, q, 2, -1, d)
            nulls = []
            for al in alphas:
                fa = ps.fit_rotating(z, al * qu, 2, +1, d)
                nulls.append(abs(fa.b) / fa.sigma_b)
            line.append(f"d{d}: snr+ {abs(fp.b)/fp.sigma_b:4.1f} snr- {abs(fn.b)/fn.sigma_b:4.1f} "
                        f"null med {np.median(nulls):4.1f} max {np.max(nulls):4.1f} |b| {abs(fp.b):.1e} "
                        f"sig {fp.sigma_b:.1e} rho {fp.rho:+.2f}")
        print(f"  {nm} ({len(q)} frames)\n" + "\n".join("     " + x for x in line))
