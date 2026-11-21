"""Test of tools/extract_psf_cutouts.py: the cut, and the frames it leaves out."""
import os
import subprocess
import sys

import numpy as np
from astropy.io import fits

TOOL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "extract_psf_cutouts.py")


def test_cutouts_leave_out_nan_and_off_center_frames(tmp_path):
    """A frame with NaN in the cut and a frame with the star 12 px off are left out and listed;
    the kept frames keep their angles and the star at the cut's center."""
    red = tmp_path / "reduced_py"
    red.mkdir()
    n, size = 8, 200
    yy, xx = np.mgrid[0:size, 0:size]
    star = np.exp(-((xx - 99.5) ** 2 + (yy - 99.5) ** 2) / 50.0).astype(np.float32)
    off = np.exp(-((xx - 111.5) ** 2 + (yy - 99.5) ** 2) / 50.0).astype(np.float32)
    cube = np.repeat(star[None], n, axis=0)
    cube[2, 60:70, :] = np.nan                     # padding reaching into the 80 px cut (rows 60-139)
    cube[5] = off
    cube[6, :20, :] = np.nan                       # NaN outside the cut: kept
    h = fits.Header()
    h["XCEN"], h["YCEN"] = 99.5, 99.5
    fits.PrimaryHDU(cube, h).writeto(red / "Fake_20220522_left_cube1_align_cen_clean_bin.fits")
    fits.PrimaryHDU(np.linspace(-80.0, -100.0, n)).writeto(red / "Fake_20220522_left_angles1_clean_bin.fits")
    out = tmp_path / "cut"
    r = subprocess.run([sys.executable, TOOL, str(red), "Fake_20220522", "left", str(out), "--half", "40"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "left out 1 frame(s) with NaN in the cut and 1 with the star more than 5.0 px off: [2, 5]" in r.stdout
    with fits.open(out / "Fake_20220522_left_cube1_clean_core80.fits") as hd:
        d, ang, dropped, head = hd[0].data, hd["ANGLES"].data, hd["DROPPED"].data, hd[0].header
    assert d.shape == (6, 80, 80) and np.all(np.isfinite(d))
    assert dropped.tolist() == [2, 5] and head["NDROPNAN"] == 1 and head["NDROPOFF"] == 1
    assert np.allclose(ang, np.delete(np.linspace(-80.0, -100.0, n), [2, 5]))
    assert (head["XCEN"], head["YCEN"]) == (39.5, 39.5)
    img = d[0]
    yy2, xx2 = np.mgrid[0:80, 0:80]
    assert abs((img * xx2).sum() / img.sum() - 39.5) < 1e-3
