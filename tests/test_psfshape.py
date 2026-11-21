"""Tests of the PSF-shape companion test (reduce_nomic.psfshape) on synthetic frames."""
import numpy as np
import pytest

from reduce_nomic import psfshape as ps


def _frames(n=60, size=48, sig=4.7, q_span=120.0, noise=0.0, seed=0):
    """Circular Gaussian PSFs at the frame center with parallactic angles over q_span degrees."""
    yy, xx = np.mgrid[0:size, 0:size]
    c = size / 2 - 0.5
    f = np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * sig ** 2))
    rng = np.random.default_rng(seed)
    fr = np.repeat(f[None], n, axis=0) + noise * rng.standard_normal((n, size, size))
    q = np.linspace(-q_span / 2, q_span / 2, n)
    return fr, q, c


def test_moments_of_a_round_star_vanish():
    """A circular Gaussian has no elongation or lopsidedness, and its centroid is its center."""
    fr, q, c = _frames(n=3)
    m = ps.weighted_moments(fr, 3.0, c + 0.7, c - 0.4)
    assert np.allclose(m["xc"], c, atol=1e-6) and np.allclose(m["yc"], c, atol=1e-6)
    for k in ("spin1", "spin2", "spin3"):
        assert np.all(np.abs(m[k]) < 1e-6)


def test_fit_rotating_recovers_a_turning_term():
    """A static term, a drift and a term turning with q are separated when the rotation is large."""
    rng = np.random.default_rng(3)
    n = 200
    q = np.linspace(-80, 80, n)
    t = np.linspace(-1, 1, n)
    b_true = 0.004 * np.exp(1j * 0.7)
    z = (0.01 - 0.003j) + 0.002 * t + b_true * np.exp(-1j * 2 * np.deg2rad(q)) \
        + 0.001 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    fit = ps.fit_rotating(z, q, 2, +1, 1)
    assert abs(fit.b - b_true) < 4 * fit.sigma_b
    assert fit.sigma_b < 5e-4
    z0 = z - b_true * np.exp(-1j * 2 * np.deg2rad(q))           # noise, static and drift only
    null = ps.fit_rotating(z0, q, 2, -1, 1)
    assert abs(null.b) < 4 * null.sigma_b


def test_injected_companion_follows_its_sky_angle():
    """Injection gives a positive response whose phase follows spin x angle, and shifts move content as stated."""
    fr, q, c = _frames()
    cal = ps.response(fr, q, 3.0, c, c, [1, 2], 3.6, 0.02, n_phi=8)
    assert cal[1]["r"] > 0 and cal[2]["r"] > 0
    assert cal[1]["phase_sd"] < 10 and cal[2]["phase_sd"] < 10
    sh = ps.shift_frames(fr[:1], np.array([2.0]), np.array([0.0]))
    iy, ix = np.unravel_index(np.argmax(sh[0]), sh[0].shape)
    assert ix == int(c + 0.5) + 2 or ix == int(c - 0.5) + 2


def test_combine_and_threshold():
    """Equal sets average to their common value with an error smaller by sqrt(2); 5.49 sigma for one try."""
    out = ps.combine([0.01 + 0.0j, 0.01 + 0.0j], [0.001, 0.001], [0.5, 0.5])
    assert out["f_abs"] == pytest.approx(0.02)
    assert out["sigma_f"] == pytest.approx(0.002 / np.sqrt(2))
    assert out["chi2"] == pytest.approx(0.0)
    assert ps.rayleigh_threshold() == pytest.approx(5.489, abs=1e-3)
    assert ps.rayleigh_threshold(trials=2) > ps.rayleigh_threshold()


def test_small_sample_threshold_and_empirical_scale():
    """The F(2, nu) threshold tends to the Rayleigh one and has the right tail; the scale is the
    pooled reduced chi-square of the noise-only quantities."""
    from reduce_nomic import psfshape as ps
    p = 2.866515719e-7
    assert abs(ps.small_sample_threshold(10 ** 7, p) - ps.rayleigh_threshold(p)) < 1e-3
    t = ps.small_sample_threshold(10, p)
    assert abs(t - 13.91) < 0.01
    x = t ** 2 / 2.0
    assert abs((1 + 2 * x / 10) ** (-5) - p) < 1e-12
    s, nu = ps.empirical_scale(4.0, [1.0, 1.0, np.sqrt(2.0)])
    assert nu == 10 and abs(s - np.sqrt((4.0 + 1 + 1 + 2) / 10)) < 1e-12
    rng = np.random.default_rng(0)                  # Gaussian noise of scale 3 gives s near 3
    vals = []
    for _ in range(2000):
        z = 3.0 * (rng.normal(size=3) + 1j * rng.normal(size=3))
        bp = 3.0 * (rng.normal(size=3) + 1j * rng.normal(size=3)) + 0.5
        chi2 = float(np.sum(np.abs(bp - bp.mean()) ** 2))
        vals.append(ps.empirical_scale(chi2, np.abs(z))[0] ** 2)
    assert abs(np.mean(vals) - 9.0) < 0.3
