"""Tests of the IDL-compatible primitives in reduce_nomic.idl."""
import numpy as np
import pytest

from reduce_nomic import idl


def test_idl_round_half_away_from_zero():
    assert idl.idl_round(2.5) == 3
    assert idl.idl_round(-2.5) == -3
    assert idl.idl_round(1.49) == 1
    assert list(idl.idl_round(np.array([0.5, -0.5, 1.5]))) == [1, -1, 2]


def test_smooth_interior_edges_and_even_width():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(40, 50))
    out = idl.smooth(a, 4)          # even width -> 5
    assert np.allclose(out[:2, :], a[:2, :]) and np.allclose(out[:, -2:], a[:, -2:])
    assert np.isclose(out[10, 20], a[8:13, 18:23].mean())
    out7 = idl.smooth(a, 7)
    assert np.isclose(out7[10, 20], a[7:14, 17:24].mean())


def test_smooth_nan_handling():
    a = np.ones((30, 30))
    a[15, 15] = np.nan
    with_nan = idl.smooth(a, 5, nan=True)
    assert np.isclose(with_nan[15, 15], 1.0) and np.isfinite(with_nan).all()
    no_nan = idl.smooth(a, 5)
    assert np.isnan(no_nan[14, 14]) and np.isnan(no_nan[17, 17]) and np.isfinite(no_nan[12, 12])
    allnan = np.full((20, 20), np.nan)
    assert np.isnan(idl.smooth(allnan, 5, nan=True)[10, 10])


def test_median_even_odd_and_nan():
    assert idl.median([1, 2, 3, 4]) == 3
    assert idl.median([1, 2, 3, 4], even=True) == 2.5
    assert idl.median([1, np.nan, 3]) == 3   # two finite values, upper middle
    cube = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [np.nan, 12]]], dtype=float)
    m = idl.median(cube, axis=0)
    assert m[0, 0] == 5 and m[1, 0] == 7     # (3, 7) -> upper middle 7
    me = idl.median(cube, axis=0, even=True)
    assert me[1, 0] == 5.0
    assert np.isnan(idl.median(np.full((3, 2), np.nan), axis=0)).all()


def test_fshift_integer_and_fraction():
    a = np.arange(64, dtype=float).reshape(8, 8)
    assert np.array_equal(idl.fshift(a, 2, -1), np.roll(a, (-1, 2), axis=(0, 1)))
    ramp = np.tile(np.arange(32, dtype=float), (16, 1))
    s = idl.fshift(ramp, 0.25, 0.0)
    assert np.allclose(s[:, 5:20], ramp[:, 5:20] - 0.25)
    s = idl.fshift(ramp, -0.25, 0.0)
    assert np.allclose(s[:, 5:20], ramp[:, 5:20] + 0.25)


def _gauss(n, x0, y0, fwhm):
    sd = fwhm / 2.3548200450309493
    yy, xx = np.mgrid[0:n, 0:n]
    return np.exp(-0.5 * ((xx - x0) ** 2 + (yy - y0) ** 2) / sd ** 2)


def test_crosscorr_normalized_and_peak():
    g = _gauss(64, 32.0, 32.0, 6.0)
    out = idl.crosscorr(g, g)
    assert np.isclose(out.max(), 1.0, atol=1e-9)
    assert np.unravel_index(np.argmax(out), out.shape) == (32, 32)
    h = _gauss(64, 35.3, 30.6, 6.0)
    _, (px, py) = idl.crosscorr(g, h, pmax=True)
    # The peak sits at n/2 minus the second image's displacement; above n/2, IDL subtracts n.
    assert np.isclose(px, 32 - 3.3, atol=0.05) and np.isclose(py, 32 + 1.4 - 64, atol=0.05)


def test_align_formula_recenters_star():
    """The IDL align step (dx from pmax, shift by -dx) moves the star onto the reference."""
    n, r = 200, 100
    ref = _gauss(n, r, r, 14.0)
    frame = _gauss(n, r + 2.6, r - 1.7, 14.0)
    _, (px, py) = idl.crosscorr(ref, frame, cut=True, pmax=True)
    dx = r - px if px >= 0 else -r + abs(px)
    dy = r - py if py >= 0 else -r + abs(py)
    moved = idl.fshift(frame, -dx, -dy)
    xc, yc = idl.cntrd(moved, r, r, 14.0)
    assert abs(xc - r) < 0.1 and abs(yc - r) < 0.1


def test_cntrd_gaussian():
    g = _gauss(101, 50.0, 47.0, 7.0)
    assert idl.cntrd(g, 50, 47, 7.0) == pytest.approx((50.0, 47.0), abs=1e-9)
    g = _gauss(101, 50.3, 47.8, 7.0)
    xc, yc = idl.cntrd(g, 50, 48, 7.0)
    # The derivative centroid of CNTRD is biased by a few hundredths of a pixel off center.
    assert abs(xc - 50.3) < 0.1 and abs(yc - 47.8) < 0.1
    assert idl.cntrd(g, 1, 1, 7.0) == (-1.0, -1.0)


def test_polyfit2d_max_paraboloid():
    yy, xx = np.mgrid[0:5, 0:5]
    patch = 10 - (xx - 2.3) ** 2 - 0.5 * (yy - 1.8) ** 2
    x, y = idl.polyfit2d_max(patch)
    assert np.isclose(x, 2.3, atol=1e-5) and np.isclose(y, 1.8, atol=1e-5)


def test_surface_fit_exact_quadratic():
    yy, xx = np.mgrid[0:20, 0:30].astype(float)
    z = 3 + 0.5 * xx - 0.2 * yy + 0.01 * xx * yy + 0.003 * xx ** 2 * yy ** 2
    sf = idl.SurfaceFit(z.shape, degree=2)
    assert np.allclose(sf.surface(z), z, atol=1e-8)


def test_fixpix_interior_and_trimmed_mean():
    img = np.arange(100, dtype=float).reshape(10, 10)
    bad = np.ones((10, 10))
    bad[5, 5] = 0
    fp = idl.FixPix(bad)
    out = fp.apply(img)
    # 8 nearest good pixels of (5, 5): the 3x3 ring; their trimmed mean on this plane is 55.
    assert np.isclose(out[5, 5], 55.0)
    assert np.array_equal(np.delete(out.ravel(), 55), np.delete(img.ravel(), 55))


def test_fixpix_uses_original_values():
    img = np.ones((12, 12))
    img[6, 6] = 1000.0
    bad = np.ones((12, 12))
    bad[6, 6] = 0
    bad[6, 7] = 0
    out = idl.FixPix(bad).apply(img)
    assert np.isclose(out[6, 6], 1.0) and np.isclose(out[6, 7], 1.0)


def test_psf_gaussian_peak_and_center():
    p = idl.psf_gaussian(100, 14.0)
    assert p.dtype == np.float32
    assert np.isclose(p[49:51, 49:51].mean(), p[49, 49]) and p.max() < 1.0
