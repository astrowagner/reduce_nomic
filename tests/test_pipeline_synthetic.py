"""End-to-end test of the pipeline on a synthetic night with a chopping star."""
import json
import os

import numpy as np
from astropy.io import fits

from reduce_nomic import idl
from reduce_nomic.config import load_settings
from reduce_nomic.pipeline import reduce


def make_night(root, n=240, period=6, star=((150.3, 192.6), (152.1, 320.4)), fwhm=15.0, seed=3):
    """Write a synthetic NOMIC night and a master bad-pixel map under ``root``.

    Parameters
    ----------
    root : str
        Archive root to create.
    n : int, default 240
        Number of raw frames.
    period : int, default 6
        Frames per chop position.
    star : tuple of tuple of float, default ((150.3, 192.6), (152.1, 320.4))
        ``(x, y)`` of the star in chop A and chop B, in detector pixels, mid-channel so that the
        NaN channel edges stay clear of the core.
    fwhm : float, default 15.0
        Star FWHM in pixels.
    seed : int, default 3
        Random seed of the noise.

    Returns
    -------
    angles : ndarray, shape (n,)
        The parallactic angle written to each frame, in degrees.
    """
    rng = np.random.default_rng(seed)
    raw = os.path.join(root, "Fake_1", "raw")
    os.makedirs(raw)
    os.makedirs(os.path.join(root, "calib"))
    bad = np.ones((512, 512), np.float32)
    bad[rng.integers(70, 512, 40), rng.integers(0, 512, 40)] = 0
    fits.PrimaryHDU(bad).writeto(os.path.join(root, "calib", "master_badmap_2022.fits"))
    yy, xx = np.mgrid[0:512, 0:512]
    sd = fwhm / 2.3548
    angles = np.linspace(-20, 25, n)
    for i in range(n):
        x0, y0 = star[(i // period) % 2]
        img = 9000 + 3 * xx / 512 + rng.normal(0, 20, (512, 512))
        img += 30000 * np.exp(-0.5 * ((xx - x0) ** 2 + (yy - y0) ** 2) / sd ** 2)
        h = fits.Header()
        h["LBT_PARA"] = float(angles[i])
        h["EXPTIME"] = 0.0824
        h["DATE-OBS"] = "2023-01-01"
        h["TIME-END"] = f"05:{i // 60:02d}:{i % 60:02d}.000"
        h["OBJNAME"] = "fake star"
        fits.PrimaryHDU(img.astype(np.float32), header=h).writeto(os.path.join(raw, f"n_230101_{i:06d}.fits"))
    return angles


def test_synthetic_night(tmp_path):
    root = str(tmp_path / "archive")
    make_night(root)
    st = load_settings("Fake_1", nights_json=str(tmp_path / "none.json"), overrides={
        "archive": root, "obj": "fakestar", "prebin": 5, "background_limit": 50000.0,
        "pre_corr_thresh_1": 0.5, "pre_corr_thresh_2": 0.5, "corr_thresh_1": 0.5, "corr_thresh_2": 0.5,
        "pre_noise_thresh_1": 3, "pre_noise_thresh_2": 3, "noise_thresh_1": 3, "noise_thresh_2": 3,
        "pre_peak_thresh_1": 0.5, "pre_peak_thresh_2": 0.5, "peak_thresh_1": 0.5, "peak_thresh_2": 0.5})
    out = str(tmp_path / "out")
    summary = reduce(st, "left", workers=2, out_dir=out)
    for cn in (1, 2):
        cube = fits.getdata(os.path.join(out, f"Fake_1_left_cube{cn}_align_cen_clean_bin.fits"))
        ang = fits.getdata(os.path.join(out, f"Fake_1_left_angles{cn}_clean_bin.fits"))
        psf = fits.getdata(os.path.join(out, f"Fake_1_left_cube{cn}_align_cen_clean_median_PSF.fits"))
        assert cube.shape[1:] == (1000, 1000) and cube.shape[0] == ang.size >= 2
        assert np.all(np.diff(ang) > 0)
        xc, yc = idl.cntrd(np.nan_to_num(psf.astype(float)), 499, 499, 15.0)
        assert abs(xc - 499.5) < 0.3 and abs(yc - 499.5) < 0.3, (xc, yc)
        assert np.isclose(np.nanmax(psf), 1.0, atol=0.05)
    assert summary["cubes"][1]["final"] >= 2
    assert json.load(open(os.path.join(out, "Fake_1_left_summary.json")))["n_raw"] == 240


def test_synthetic_night_display_and_refusal(tmp_path, monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    root = str(tmp_path / "archive")
    make_night(root, n=120)
    over = {"archive": root, "obj": "fakestar", "prebin": 5, "background_limit": 50000.0, "display": True,
            "pre_corr_thresh_1": 0.5, "pre_corr_thresh_2": 0.5, "corr_thresh_1": 0.5, "corr_thresh_2": 0.5,
            "pre_noise_thresh_1": 3, "pre_noise_thresh_2": 3, "noise_thresh_1": 3, "noise_thresh_2": 3,
            "pre_peak_thresh_1": 0.5, "pre_peak_thresh_2": 0.5, "peak_thresh_1": 0.5, "peak_thresh_2": 0.5}
    st = load_settings("Fake_1", nights_json=str(tmp_path / "none.json"), overrides=over)
    out = str(tmp_path / "out")
    reduce(st, "left", workers=2, out_dir=out)
    png = os.path.join(out, "Fake_1_left_display.png")
    assert os.path.getsize(png) > 10000
    import pytest
    with pytest.raises(FileExistsError):
        reduce(st, "left", workers=2, out_dir=out)


def test_work_lock_refuses_second_run(tmp_path):
    import fcntl
    import pytest
    root = str(tmp_path / "archive")
    make_night(root, n=60)
    work = tmp_path / "work"
    work.mkdir()
    held = open(work / "lock", "a")
    fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    st = load_settings("Fake_1", nights_json=str(tmp_path / "none.json"), overrides={"archive": root, "obj": "fakestar"})
    with pytest.raises(RuntimeError, match="another run"):
        reduce(st, "left", workers=1, out_dir=str(tmp_path / "out"), work_dir=str(work))
