"""Tests of analysis/procyon3-4_2026-11/edge_profile.py on synthetic frames with a NaN edge.

The frames are white noise of unit sigma, NaN from a column on, with a 3 px rim of triple noise
next to the NaN, or a ramp that rises toward the edge.
"""
import importlib.util
import os

import numpy as np
import pytest
from astropy.io import fits

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = importlib.util.spec_from_file_location(
    "edge_profile", os.path.join(HERE, "..", "analysis", "procyon3-4_2026-11", "edge_profile.py"))
ep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ep)


def rim_cube(n=6, size=200, rim=3, factor=3.0, seed=1):
    """Return frames of unit noise, NaN past a column, with a noisier rim next to the NaN.

    Parameters
    ----------
    n : int, default 6
        Number of frames.
    size : int, default 200
        Frame side, in pixels.
    rim : int, default 3
        Width of the noisier rim, in pixels.
    factor : float, default 3.0
        Noise of the rim over the noise elsewhere.
    seed : int, default 1
        Seed of the noise.

    Returns
    -------
    ndarray, shape (n, size, size)
        The frames; frame i is NaN from column 120 + 5 i on.
    """
    rng = np.random.default_rng(seed)
    cube = rng.normal(0.0, 1.0, (n, size, size))
    xx = np.arange(size)[None, :]
    for i in range(n):
        edge = 120 + 5 * i
        d = edge - xx                                   # 1 at the last finite column
        cube[i][np.broadcast_to((d >= 1) & (d <= rim), (size, size))] *= factor
        cube[i][:, edge:] = np.nan
    return cube


def test_edge_distance_counts_from_one_at_the_nan_edge():
    """The pixel next to a NaN is at distance 1; a frame without NaN gives None."""
    f = np.zeros((5, 5))
    f[:, 3:] = np.nan
    d = ep.edge_distance(f)
    assert d[2, 2] == 1.0 and d[2, 1] == 2.0 and d[2, 3] == 0.0
    assert ep.edge_distance(np.zeros((4, 4))) is None


def test_highpass_matches_the_klip_tpe_rules():
    """Odd width, array border zeroed, NaN kept, and a flat frame filtered to zero inside."""
    f = np.full((40, 40), 5.0)
    f[:, 30:] = np.nan
    h = ep.highpass(f, 6)                              # width 6 -> 7, half 3
    assert np.all(h[:3, :30] == 0) and np.all(np.isnan(h[:, 30:]))
    assert np.allclose(h[3:-3, 3:30], 0.0)
    assert np.array_equal(ep.highpass(f, 1)[np.isfinite(f)], f[np.isfinite(f)])


def test_profile_finds_the_noisy_rim():
    """The 3 px rim reads about 3 in bins 1-3 and about 1 beyond, raw and high-passed."""
    pooled, used = ep.profile(rim_cube(), [0, 5], rmin=0, rmax=500, border=10, dmax=10, dint=10)
    assert used == 6
    for w in (0, 5):
        rows = ep.summarize(pooled[w])
        assert rows[0][1] == 0                          # no pixel closer than 1 px
        for k in (1, 2, 3):
            assert rows[k][3] == pytest.approx(3.0, rel=0.15)
        for k in (5, 6, 7, 8):
            assert rows[k][3] == pytest.approx(1.0, rel=0.15)


def test_main_writes_and_refuses_to_overwrite(tmp_path):
    """The table and the figure are written once; a second call stops."""
    cube = tmp_path / "c.fits"
    fits.writeto(cube, rim_cube().astype(np.float32))
    out = str(tmp_path / "ep")
    ep.main([str(cube), "--out", out, "--labels", "syn", "--rmin", "0", "--rmax", "500", "--border", "10",
             "--dmax", "10", "--dint", "10"])
    text = open(out + ".csv").read().splitlines()
    assert text[0].startswith("cube,width_px,frames,distance_bin_px")
    assert len(text) == 1 + 2 * 11 and os.path.exists(out + ".png")
    with pytest.raises(SystemExit):
        ep.main([str(cube), "--out", out])
