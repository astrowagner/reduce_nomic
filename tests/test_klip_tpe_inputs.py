"""Test of tools/klip_tpe_inputs.py on a synthetic archive."""
import json
import os
import subprocess
import sys

import numpy as np
from astropy.io import fits

TOOL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "klip_tpe_inputs.py")


def _night(root, night, n):
    """Write one side's port products for a night: a centered star in 300 px frames."""
    red = os.path.join(root, night, "reduced_py")
    os.makedirs(red)
    yy, xx = np.mgrid[0:300, 0:300]
    star = np.exp(-((xx - 149.5) ** 2 + (yy - 149.5) ** 2) / 72.0).astype(np.float32)
    h = fits.Header()
    h["XCEN"], h["YCEN"] = 149.5, 149.5
    for cn in (1, 2):
        pre = f"{night}_left"
        for tag in ("", "_clean"):
            fits.PrimaryHDU(np.repeat(star[None], n, axis=0), h).writeto(os.path.join(red, f"{pre}_cube{cn}_align_cen{tag}_bin.fits"))
            fits.PrimaryHDU(np.linspace(-10, 10, n)).writeto(os.path.join(red, f"{pre}_angles{cn}{tag}_bin.fits"))
        fits.PrimaryHDU(star, h).writeto(os.path.join(red, f"{pre}_cube{cn}_align_cen_clean_median_PSF.fits"))


def test_inputs_centered_with_references(tmp_path):
    """Cut cubes and PSFs keep the star at the array center; references stack; the command lists every partition."""
    root = str(tmp_path)
    _night(root, "Fake_20220101", 10)
    _night(root, "Fake_20220102", 7)
    r = subprocess.run([sys.executable, TOOL, root, "Fake_20220101", "--tag", "rdi", "--half", "40", "--psf-half", "10",
                        "--ref", "Fake_20220102", "Missing_1"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    d = os.path.join(root, "Fake_20220101", "klip_tpe", "inputs_rdi")
    cube = fits.getdata(os.path.join(d, "Fake_20220101_left_cube1_all_c80.fits"))
    ref = fits.getdata(os.path.join(d, "Fake_20220101_left_cube1_ref_c80.fits"))
    psf = fits.getdata(os.path.join(d, "Fake_20220101_left_cube1_psf20.fits"))
    assert cube.shape == (10, 80, 80) and ref.shape == (7, 80, 80) and psf.shape == (20, 20)
    yy, xx = np.mgrid[0:80, 0:80]
    img = cube[0]
    assert abs((img * xx).sum() / img.sum() - 39.5) < 1e-4 and abs((img * yy).sum() / img.sum() - 39.5) < 1e-4
    m = json.load(open(os.path.join(d, "manifest.json")))
    assert [p["name"] for p in m["partitions"]] == ["left_A", "left_B"]
    assert m["partitions"][0]["ref_nights_used"] == ["Fake_20220102"]
    sh = open(os.path.join(root, "Fake_20220101", "klip_tpe", "run_rdi.sh")).read()
    assert "set -euo pipefail" in sh and "--ref-cube" in sh and "--names left_A left_B" in sh
    again = subprocess.run([sys.executable, TOOL, root, "Fake_20220101", "--tag", "rdi"], capture_output=True, text=True)
    assert again.returncode != 0                         # an existing tag is refused
