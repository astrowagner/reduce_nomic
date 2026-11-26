"""Tests for analysis/procyon3-4_2026-11/feature_check.py on synthetic images.

The images are white noise of unit sigma, 320 x 320 px with the star at (159.5, 159.5) as in
KLIP-TPE, with Gaussian point sources of FWHM 15.653 px (NOMIC's) placed by separation and PA,
north up and east left.
"""
import importlib.util
import os

import numpy as np
import pytest
from astropy.io import fits

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = importlib.util.spec_from_file_location(
    "feature_check", os.path.join(HERE, "..", "analysis", "procyon3-4_2026-11", "feature_check.py"))
fc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fc)

FWHM, PXS = 15.653, 0.0179


def gaussian_source(shape, center, rho, pa, amp):
    """Return a Gaussian point source of peak ``amp`` at (rho [arcsec], pa [deg]).

    Parameters
    ----------
    shape : tuple of int
        Image shape (ny, nx).
    center : tuple of float
        Star position (cy, cx), in pixels.
    rho, pa : float
        Separation (arcsec) and position angle (degrees east of north, east left).
    amp : float
        Peak value.

    Returns
    -------
    ndarray, shape (ny, nx)
        The source.
    """
    dy, dx = fc.offset_of(rho / PXS, pa)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    s = FWHM / (2 * np.sqrt(2 * np.log(2)))
    return amp * np.exp(-((yy - center[0] - dy) ** 2 + (xx - center[1] - dx) ** 2) / (2 * s * s))


def test_parse_image_spec():
    s = fc.parse_image_spec("idl:P4 ADI=/x/y/P4_adi.fits")
    assert (s.label, s.path, s.plane, s.idl) == ("P4 ADI", "/x/y/P4_adi.fits", None, True)
    s = fc.parse_image_spec("P4 A=/x/best_clean_partitions.fits:left_A")
    assert (s.path, s.plane, s.idl) == ("/x/best_clean_partitions.fits", "left_A", False)
    s = fc.parse_image_spec("e=/x/eval0001_clean.fits.gz")
    assert (s.path, s.plane) == ("/x/eval0001_clean.fits.gz", None)
    with pytest.raises(ValueError):
        fc.parse_image_spec("no-equals-sign")


def test_geometry_conventions():
    assert fc.star_center((320, 320), idl=False) == (159.5, 159.5)
    assert fc.star_center((1000, 1000), idl=True) == (500.0, 500.0)
    dy, dx = fc.offset_of(10.0, 90.0)                     # east is to the left: -x
    assert dy == pytest.approx(0.0, abs=1e-9) and dx == pytest.approx(-10.0)
    rho, pa = fc.polar(159.5 + 76.2, 159.5 + 81.7, (159.5, 159.5), PXS)
    assert rho == pytest.approx(2.0, abs=0.01) and pa == pytest.approx(313.0, abs=0.3)
    k = fc.aperture_kernel(FWHM / 2)
    assert k.sum() == pytest.approx(np.pi * (FWHM / 2) ** 2, rel=0.01)


def test_measure_finds_a_source_with_the_expected_snr():
    rng = np.random.default_rng(3)
    c = (159.5, 159.5)
    img = rng.normal(size=(320, 320)) + gaussian_source((320, 320), c, 2.0, 313.0, 1.0)
    common, per = fc.measure_image(img, c, [(2.0, 313.0)], FWHM, PXS, 1.65, 2.35, 0.5)
    res = per[0]
    # half the flux of a Gaussian falls inside FWHM/2: 0.5 * 2 pi s^2 = 138.8 per unit peak, over
    # the noise of a 192 px aperture, sqrt(192) = 13.9, for S/N ~ 10
    assert 7.0 < res["snr"] < 13.0
    assert res["offset_px"] < 2.5
    assert res["rank"] == 1 and common["n_max"] > 5
    assert res["peak_rho"] == pytest.approx(2.0, abs=0.04)
    assert res["peak_pa"] == pytest.approx(313.0, abs=1.5)


def test_noise_only_gives_snr_near_unit_scatter():
    rng = np.random.default_rng(4)
    c = (159.5, 159.5)
    s, _ = fc.snr_map(rng.normal(size=(320, 320)), c, FWHM, 1.65 / PXS, 2.35 / PXS)
    v = s[np.isfinite(s)]
    assert abs(np.mean(v)) < 0.15 and 0.8 < np.std(v) < 1.25


def test_contrast_from_injections(tmp_path):
    rng = np.random.default_rng(5)
    c = (159.5, 159.5)
    clean = rng.normal(scale=0.05, size=(320, 320)) + gaussian_source((320, 320), c, 2.0, 313.0, 0.5)
    inj = clean.copy()
    hdr = fits.Header()
    srcs = [(1.8, 30.0), (2.0, 150.0), (2.2, 210.0), (1.0, 90.0)]   # the last lies outside the annulus
    hdr["INJNSRC"] = len(srcs)
    for k, (r, t) in enumerate(srcs, start=1):
        inj += gaussian_source((320, 320), c, r, t, 1.0)
        hdr[f"INJRHO{k}"], hdr[f"INJPA{k}"] = r, t
    path = tmp_path / "inj.fits"
    fits.writeto(path, inj, hdr)
    per_c, n = fc.flux_per_contrast(clean, str(path), 2e-4, c, FWHM, PXS, 1.65 / PXS, 2.35 / PXS)
    assert n == 3
    res = fc.measure_image(clean, c, [(2.0, 313.0)], FWHM, PXS, 1.65, 2.35, 0.5)[1][0]
    assert res["flux"] / per_c == pytest.approx(1e-4, rel=0.05)    # half the injected peak
    # per-source contrasts from the header win, and then no contrast is needed
    for k in range(1, len(srcs) + 1):
        hdr[f"INJCON{k}"] = 4e-4
    path2 = tmp_path / "inj_con.fits"
    fits.writeto(path2, inj, hdr)
    per_c2, _ = fc.flux_per_contrast(clean, str(path2), None, c, FWHM, PXS, 1.65 / PXS, 2.35 / PXS)
    assert res["flux"] / per_c2 == pytest.approx(2e-4, rel=0.05)
    hdr_none = fits.Header({"INJNSRC": 1, "INJRHO1": 2.0, "INJPA1": 150.0})
    path3 = tmp_path / "inj_none.fits"
    fits.writeto(path3, inj, hdr_none)
    with pytest.raises(ValueError):
        fc.flux_per_contrast(clean, str(path3), None, c, FWHM, PXS, 1.65 / PXS, 2.35 / PXS)


def test_parse_inj():
    assert fc.parse_inj("J=/x/inj.fits:1.03e-4") == ("J", "/x/inj.fits", 1.03e-4)
    assert fc.parse_inj("J=/x/inj.fits") == ("J", "/x/inj.fits", None)
    with pytest.raises(ValueError):
        fc.parse_inj("/x/inj.fits")


def test_main_writes_outputs_and_refuses_to_overwrite(tmp_path):
    rng = np.random.default_rng(6)
    c = (159.5, 159.5)
    a = rng.normal(size=(320, 320)) + gaussian_source((320, 320), c, 2.0, 313.0, 0.6)
    cube = np.stack([rng.normal(size=(320, 320)), a])
    h = fits.Header()
    h["NIGHT1"], h["NIGHT2"] = "left_A", "left_B"
    fits.writeto(tmp_path / "cube.fits", cube, h)
    b = np.full((1000, 1000), np.nan)
    b[340:660, 340:660] = a                # star at 499.5, half a pixel from the IDL convention's 500
    fits.writeto(tmp_path / "idl.fits", b)
    out = str(tmp_path / "chk")
    rc = fc.main([out, "B=" + str(tmp_path / "cube.fits") + ":left_B", "-",
                  "idl:IDL=" + str(tmp_path / "idl.fits"), "--pos", "2.0", "313", "--pos", "1.8", "40"])
    assert rc == 0
    for ext in (".csv", ".txt", ".png"):
        assert os.path.getsize(out + ext) > 0
    rows = open(out + ".csv").read().splitlines()
    assert len(rows) == 5 and rows[1].startswith("B,") and rows[4].startswith("IDL,")
    assert ",1,2.000,313.0," in rows[1] and ",2,1.800,40.0," in rows[2]
    with pytest.raises(SystemExit):
        fc.main([out, "B=" + str(tmp_path / "cube.fits") + ":left_B", "--pos", "2.0", "313"])


def test_profile_zero_radius():
    yy, xx = np.mgrid[:101, :101]
    r2 = (yy - 50) ** 2 + (xx - 50) ** 2
    s1, s2 = 4.0, 8.0
    # a narrow Gaussian minus a broad one with the same integral (a high-pass filtered point
    # source) crosses zero where the two are equal
    narrow, broad = np.exp(-r2 / (2 * s1 ** 2)), (s1 / s2) ** 2 * np.exp(-r2 / (2 * s2 ** 2))
    expect = np.sqrt(np.log((s2 / s1) ** 2) / (1 / (2 * s1 ** 2) - 1 / (2 * s2 ** 2)))
    z = fc.profile_zero_radius(narrow - broad, 50, 50, 30)
    assert z == pytest.approx(expect, abs=0.6)
    assert np.isnan(fc.profile_zero_radius(narrow, 50, 50, 30))


def test_psf_zero_radius(tmp_path):
    c = (159.5, 159.5)
    clean = np.zeros((320, 320))
    inj = clean.copy()
    hdr = fits.Header()
    s1, s2 = 6.0, 12.0
    srcs = [(1.8, 30.0), (2.0, 150.0), (2.2, 270.0)]
    hdr["INJNSRC"] = len(srcs)
    yy, xx = np.mgrid[:320, :320]
    for k, (r, t) in enumerate(srcs, start=1):
        dy, dx = fc.offset_of(r / PXS, t)
        r2 = (yy - c[0] - dy) ** 2 + (xx - c[1] - dx) ** 2
        inj += np.exp(-r2 / (2 * s1 ** 2)) - (s1 / s2) ** 2 * np.exp(-r2 / (2 * s2 ** 2))
        hdr[f"INJRHO{k}"], hdr[f"INJPA{k}"] = r, t
    path = tmp_path / "inj.fits"
    fits.writeto(path, inj, hdr)
    z, n = fc.psf_zero_radius(clean, str(path), c, FWHM, PXS, 1.65 / PXS, 2.35 / PXS)
    expect = np.sqrt(np.log((s2 / s1) ** 2) / (1 / (2 * s1 ** 2) - 1 / (2 * s2 ** 2)))
    assert n == 3 and z == pytest.approx(expect, abs=0.8)


def test_fake_zero_radii_recover_the_template_shape(tmp_path):
    rng = np.random.default_rng(7)
    c = (159.5, 159.5)
    clean = rng.normal(scale=0.02, size=(320, 320))
    inj = clean.copy()
    hdr = fits.Header()
    s1, s2 = 6.0, 12.0
    srcs = [(1.8, 30.0), (2.0, 150.0), (2.2, 270.0)]
    hdr["INJNSRC"] = len(srcs)
    yy, xx = np.mgrid[:320, :320]
    for k, (r, t) in enumerate(srcs, start=1):
        dy, dx = fc.offset_of(r / PXS, t)
        r2 = (yy - c[0] - dy) ** 2 + (xx - c[1] - dx) ** 2
        inj += np.exp(-r2 / (2 * s1 ** 2)) - (s1 / s2) ** 2 * np.exp(-r2 / (2 * s2 ** 2))
        hdr[f"INJRHO{k}"], hdr[f"INJPA{k}"] = r, t
    path = tmp_path / "inj.fits"
    fits.writeto(path, inj, hdr)
    expect = np.sqrt(np.log((s2 / s1) ** 2) / (1 / (2 * s1 ** 2) - 1 / (2 * s2 ** 2)))
    z, found = fc.fake_zero_radii(clean, str(path), c, FWHM, PXS, 1.65 / PXS, 2.35 / PXS, 20.0, [], 8, seed=3)
    assert z.shape == (8,) and found.shape == (8,)
    assert np.nanmedian(z) == pytest.approx(expect, abs=1.0)        # bright fakes keep the template's shape
    assert 14.0 < np.median(found) < 28.0                            # found near the S/N they were given


def test_main_with_fakes(tmp_path):
    rng = np.random.default_rng(8)
    c = (159.5, 159.5)
    clean = rng.normal(size=(320, 320)) + gaussian_source((320, 320), c, 2.0, 313.0, 1.5)
    inj = clean.copy()
    hdr = fits.Header()
    srcs = [(1.8, 30.0), (2.0, 150.0), (2.2, 210.0)]
    hdr["INJNSRC"] = len(srcs)
    for k, (r, t) in enumerate(srcs, start=1):
        inj += gaussian_source((320, 320), c, r, t, 3.0)
        hdr[f"INJRHO{k}"], hdr[f"INJPA{k}"], hdr[f"INJCON{k}"] = r, t, 1e-4
    fits.writeto(tmp_path / "clean.fits", clean)
    fits.writeto(tmp_path / "inj.fits", inj, hdr)
    out = str(tmp_path / "f")
    assert fc.main([out, "C=" + str(tmp_path / "clean.fits"), "--pos", "2.0", "313",
                    "--inj", "C=" + str(tmp_path / "inj.fits"), "--fakes", "4"]) == 0
    import csv as _csv
    row = next(_csv.DictReader(open(out + ".csv")))
    assert row["n_fakes"] == "4" and row["fake_zero_median_px"] != "" and row["contrast"] != ""
    assert float(row["contrast"]) == pytest.approx(0.5e-4, rel=0.2)  # 1.5 against 3.0 per 1e-4
