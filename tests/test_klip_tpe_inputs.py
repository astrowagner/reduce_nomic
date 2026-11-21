"""Test of tools/klip_tpe_inputs.py on a synthetic archive."""
import json
import os
import subprocess
import sys

import numpy as np
from astropy.io import fits

TOOL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "klip_tpe_inputs.py")


def _night(root, night, n, n_by_cube=None, angles=None):
    """Write one side's port products for a night: a centered star in 300 px frames.

    Parameters
    ----------
    root : str
        Archive root.
    night : str
        Night folder name.
    n : int
        Frames per cube.
    n_by_cube : dict, optional
        Frames per cube number, overriding ``n`` for the cubes it lists.
    angles : ndarray, optional
        Angles for every cube (degrees), sliced to each cube's frame count; default a linear
        ramp from -10 to 10.
    """
    red = os.path.join(root, night, "reduced_py")
    os.makedirs(red)
    yy, xx = np.mgrid[0:300, 0:300]
    star = np.exp(-((xx - 149.5) ** 2 + (yy - 149.5) ** 2) / 72.0).astype(np.float32)
    h = fits.Header()
    h["XCEN"], h["YCEN"] = 149.5, 149.5
    for cn in (1, 2):
        pre = f"{night}_left"
        k = (n_by_cube or {}).get(cn, n)
        ang = np.linspace(-10, 10, k) if angles is None else np.asarray(angles[:k], dtype=float)
        for tag in ("", "_clean"):
            fits.PrimaryHDU(np.repeat(star[None], k, axis=0), h).writeto(os.path.join(red, f"{pre}_cube{cn}_align_cen{tag}_bin.fits"))
            fits.PrimaryHDU(ang).writeto(os.path.join(red, f"{pre}_angles{cn}{tag}_bin.fits"))
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
    assert '--workers "${WORKERS:-auto}"' in sh
    smoke = open(os.path.join(root, "Fake_20220101", "klip_tpe", "run_rdi_smoke.sh")).read()
    assert "--n-iter 10" in smoke and "smoke_rdi" in smoke and "--ann-edges 11 34 " in smoke
    again = subprocess.run([sys.executable, TOOL, root, "Fake_20220101", "--tag", "rdi"], capture_output=True, text=True)
    assert again.returncode != 0                         # an existing tag is refused
    r2 = subprocess.run([sys.executable, TOOL, root, "Fake_20220101", "--tag", "adi", "--half", "40", "--psf-half", "10",
                         "--command-archive", "/mnt/RAID64TB/NOMIC"], capture_output=True, text=True)
    assert r2.returncode == 0, r2.stderr
    sh2 = open(os.path.join(root, "Fake_20220101", "klip_tpe", "run_adi.sh")).read()
    assert "/mnt/RAID64TB/NOMIC/Fake_20220101/klip_tpe/inputs_adi/Fake_20220101_left_cube1_all_c80.fits" in sh2
    assert root not in sh2
    assert "--show" in sh2 and "--no-display" not in sh2                 # the live window by default
    r3 = subprocess.run([sys.executable, TOOL, root, "Fake_20220101", "--tag", "quiet", "--half", "40", "--psf-half", "10",
                         "--display", "none"], capture_output=True, text=True)
    assert r3.returncode == 0, r3.stderr
    sh3 = open(os.path.join(root, "Fake_20220101", "klip_tpe", "run_quiet.sh")).read()
    smoke3 = open(os.path.join(root, "Fake_20220101", "klip_tpe", "run_quiet_smoke.sh")).read()
    assert "--no-display" in sh3 and "--show" not in sh3 and "--no-display" in smoke3


def test_transit_angles_unwrapped_and_short_cube_skipped(tmp_path):
    """Angles crossing ±180° are written without the jump, a bin averaged across the wrap is left
    out, and a cube below --min-frames is left out."""
    root = str(tmp_path)
    wrong = 61.0                                         # a bin averaged across the wrap
    full = np.array([170.0, 175.0, 179.5, wrong, -178.0, -172.0, -165.0, -160.0, -150.0, -140.0, -130.0, -120.0, -110.0])
    ang = np.delete(full, 3)
    _night(root, "Fake_20220521", 13, n_by_cube={2: 1}, angles=full)
    r = subprocess.run([sys.executable, TOOL, root, "Fake_20220521", "--tag", "adi", "--half", "40", "--psf-half", "10"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    d = os.path.join(root, "Fake_20220521", "klip_tpe", "inputs_adi")
    out = fits.getdata(os.path.join(d, "Fake_20220521_left_angles1_all.fits"))
    assert np.allclose(out, np.where(ang < 0, ang + 360.0, ang))        # 170 ... 179.5, 182 ... 250
    assert np.all(np.diff(out) > 0)
    m = json.load(open(os.path.join(d, "manifest.json")))
    assert [p["name"] for p in m["partitions"]] == ["left_A"]
    p = m["partitions"][0]
    assert p["unwrapped"] and abs(p["rotation_deg"] - 80.0) < 1e-9 and p["angle_first"] == 170.0
    assert p["frames"] == 12 and p["dropped_wrap_averaged"] == [{"index": 3, "angle_deg": wrong}]
    assert fits.getdata(os.path.join(d, "Fake_20220521_left_cube1_all_c80.fits")).shape[0] == 12
    assert m["skipped"] == [{"name": "left_B", "frames": 1, "reason": "fewer than 10 frames"}]
    assert not os.path.exists(os.path.join(d, "Fake_20220521_left_cube2_all_c80.fits"))
    sh = open(os.path.join(root, "Fake_20220521", "klip_tpe", "run_adi.sh")).read()
    assert "--names left_A \\" in sh and "cube2" not in sh
