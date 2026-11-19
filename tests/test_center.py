"""Tests of the center stage's estimators and of the split stage's median position frames."""
import numpy as np
import pytest

from reduce_nomic import geometry, pipeline

YY, XX = np.mgrid[0:256, 0:256]


def _psf(a1: float, a2: float, xc: float = 127.5, yc: float = 127.5) -> np.ndarray:
    """Make a broad core (FWHM 18 px) with two small bumps 4 px to either side.

    Parameters
    ----------
    a1, a2 : float
        Relative heights of the bumps at (-4, +2.4) and (+4, -2.4) px from the core.
    xc, yc : float, default 127.5
        Center of the core in pixels.

    Returns
    -------
    img : ndarray, shape (256, 256)
        The image: a unit Gaussian core plus bumps of 0.15 x a1 and 0.15 x a2 (sigma 2.5 px).
    """
    s = 18 / 2.3548
    img = np.exp(-((XX - xc) ** 2 + (YY - yc) ** 2) / (2 * s * s))
    for bx, by, a in ((-4.0, 2.4, a1), (4.0, -2.4, a2)):
        img += 0.15 * a * np.exp(-((XX - xc - bx) ** 2 + (YY - yc - by) ** 2) / (2 * 2.5 ** 2))
    return img


def test_gauss_center_finds_a_gaussian_exactly():
    """A clean Gaussian's center comes back to within 0.005 px."""
    s = 16 / 2.3548
    img = np.exp(-((XX - 131.3) ** 2 + (YY - 124.8) ** 2) / (2 * s * s)) + 0.02
    xc, yc = geometry.gauss_center(img)
    assert abs(xc - 131.3) < 0.005 and abs(yc - 124.8) < 0.005


def test_idl_estimator_jumps_where_the_gaussian_does_not():
    """Swapping which bump is brighter moves the IDL centroid by pixels, the Gaussian by < 0.1 px."""
    p, q = _psf(1.0, 0.8), _psf(0.8, 1.0)
    jump_idl = np.hypot(*np.subtract(geometry.idl_center(p), geometry.idl_center(q)))
    jump_gauss = np.hypot(*np.subtract(geometry.gauss_center(p), geometry.gauss_center(q)))
    assert jump_idl > 2.0
    assert jump_gauss < 0.1


def test_center_cube_places_the_star_and_records_both_estimators():
    """The Gaussian method puts the core at (radius - 0.5, radius - 0.5) and fills the diagnostics."""
    r = 128
    rng = np.random.default_rng(1)
    cube = np.stack([_psf(1.0, 1.0, 130.2, 125.7) + rng.normal(0, 0.002, (256, 256)) for _ in range(9)]).astype(np.float32)
    diag = {}
    out, (xc, yc) = geometry.center_cube(cube.copy(), {"radius": r, "center_method": "gauss"}, rng, diag=diag)
    assert abs(xc - 130.2) < 0.05 and abs(yc - 125.7) < 0.05
    x2, y2 = geometry.gauss_center(np.median(out, axis=0).astype(float))
    assert abs(x2 - (r - 0.5)) < 0.05 and abs(y2 - (r - 0.5)) < 0.05
    assert diag["center_gauss"] is not None and diag["center_idl"] is not None
    _, c_idl = geometry.center_cube(cube.copy(), {"radius": r, "center_method": "idl"}, np.random.default_rng(1))
    assert np.allclose(c_idl, diag["center_idl"], atol=1.0)
    with pytest.raises(ValueError):
        geometry.center_cube(cube.copy(), {"radius": r, "center_method": "other"}, rng)


def test_start_median_takes_the_chunks_first_frames_of_a_chop():
    """One frame reproduces IDL's single frames; more give their median; no such chop gives the start."""
    mm = np.arange(6 * 4 * 4, dtype=np.float32).reshape(6, 4, 4)
    mm = np.repeat(np.repeat(mm, 128, axis=1), 128, axis=2)[:, :512, :512]
    chops = ["CHOP_A", "CHOP_B", "CHOP_A", "CHOP_B", "CHOP_A", "CHOP_B"]
    assert np.array_equal(pipeline._start_median(mm, 1, 6, chops, "CHOP_B", 1), mm[1])
    assert np.array_equal(pipeline._start_median(mm, 0, 6, chops, "CHOP_A", 3), np.median(mm[[0, 2, 4]], axis=0))
    assert np.array_equal(pipeline._start_median(mm, 2, 6, chops, "CHOP_C", 5), mm[2])
