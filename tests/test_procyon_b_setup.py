"""Tests for procyon_b_position.py and position_coverage.py (analysis/procyon3-4_2026-11).

The orbit code is checked against Bond et al. (2015)'s HST measurements of Procyon B, and the
coverage code on synthetic frames whose data fill half of each frame.
"""
import importlib.util
import os

import numpy as np
import pytest
from astropy.io import fits

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "analysis", "procyon3-4_2026-11")


def _load(name):
    """Import one of the analysis scripts as a module.

    Parameters
    ----------
    name : str
        Script name without ``.py``.

    Returns
    -------
    module
        The imported script.
    """
    spec = importlib.util.spec_from_file_location(name, os.path.join(D, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pb = _load("procyon_b_position")
pc = _load("position_coverage")


def test_orbit_matches_the_hst_measurements():
    lines, max_drho, max_dpa = pb.check()
    assert len(lines) == len(pb.HST) == 16
    assert max_drho < 0.005 and max_dpa < 0.1          # the paper's residuals are of this size


def test_decimal_year_and_the_nomic_epoch():
    assert pb.decimal_year("2000-01-01T12:00") == pytest.approx(2000.0)
    assert pb.decimal_year("2021.97") == 2021.97
    rho, pa = pb.position(pb.decimal_year("2021-12-20T06:00"))
    assert rho == pytest.approx(4.910, abs=0.002) and pa == pytest.approx(333.04, abs=0.05)


def _half_frames(n_frames, size, angle):
    """Frames with data only where y > the center (north in an unrotated frame).

    Parameters
    ----------
    n_frames : int
        Number of frames.
    size : int
        Frame side, pixels.
    angle : float
        Angle of every frame, degrees.

    Returns
    -------
    cube : ndarray, shape (n_frames, size, size)
        Ones where there is data, NaN elsewhere.
    angles : ndarray, shape (n_frames,)
        The angles.
    """
    cube = np.full((n_frames, size, size), np.nan, dtype=np.float32)
    cube[:, size // 2 + 1:, :] = 1.0
    return cube, np.full(n_frames, angle)


def test_coverage_of_a_half_filled_frame(tmp_path):
    cube, ang = _half_frames(4, 200, 0.0)
    fits.writeto(tmp_path / "c.fits", cube)
    fits.writeto(tmp_path / "a.fits", ang)
    m = pc.sky_coverage.frame_masks(str(tmp_path / "c.fits"), 90, 1)
    r = 60.0
    north, south = pc.sky_coverage.coverage(m, ang, r, np.array([0.0, 180.0]))
    assert north == 1.0 and south == 0.0
    assert pc.disk_coverage(m, ang, r, 0.0, 10.0) == 1.0
    assert pc.disk_coverage(m, ang, r, 90.0, 10.0) == 0.0  # a disk astride the edge is not whole
    # a frame turned by 180 degrees has its data in the south of the sky instead
    m2 = np.concatenate([m, m])
    a2 = np.concatenate([ang, ang + 180.0])
    north, south = pc.sky_coverage.coverage(m2, a2, r, np.array([0.0, 180.0]))
    assert north == pytest.approx(0.5) and south == pytest.approx(0.5)


def test_covered_arc():
    theta = np.arange(0.0, 360.0, 1.0)
    frac = np.where((theta >= 300) | (theta <= 20), 0.8, 0.1)
    s, e, whole = pc.covered_arc(frac, theta, 0.5, 333.0)
    assert (s, e, whole) == (300.0, 20.0, False)
    s, e, whole = pc.covered_arc(frac, theta, 0.5, 100.0)
    assert np.isnan(s) and not whole
    assert pc.covered_arc(np.ones(360), theta, 0.5, 0.0) == (0.0, 360.0, True)


def test_main_writes_a_summary(tmp_path):
    cube, ang = _half_frames(3, 200, 0.0)
    fits.writeto(tmp_path / "c.fits", cube)
    fits.writeto(tmp_path / "a.fits", ang)
    out = tmp_path / "cov.txt"
    rc = pc.main(["--part", "P", str(tmp_path / "c.fits"), str(tmp_path / "a.fits"), "--rho", "1.0",
                  "--pa", "0", "--half", "90", "--radii", "0.5", "1.0", "--out", str(out)])
    assert rc == 0
    text = out.read_text()
    assert "data at the position in 1.000" in text and "the covered run holding PA 0.0" in text
    with pytest.raises(SystemExit):
        pc.main(["--part", "P", str(tmp_path / "c.fits"), str(tmp_path / "a.fits"), "--rho", "1.0",
                 "--pa", "0", "--half", "90", "--out", str(out)])
