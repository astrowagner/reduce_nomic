"""Tests of the fixes for failures met in the November 2026 batch (center, NaN replacement, skies, empty cubes)."""
import numpy as np
import pytest

from reduce_nomic import binning, chopsub, geometry, pipeline


def _cube(n=6, r=64, x=63.5, y=63.5, sig=4.0):
    """Cube of n frames 2r wide with a Gaussian star at (x, y)."""
    yy, xx = np.mgrid[0:2 * r, 0:2 * r]
    f = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sig ** 2))
    return np.repeat(f[None].astype(np.float32), n, axis=0)


def test_center_falls_back_to_the_other_estimator(monkeypatch):
    """A failed Gaussian fit uses the IDL centroid instead of stopping the run."""
    def boom(med):
        raise RuntimeError("Gaussian center fit left the fitted box")
    monkeypatch.setattr(geometry, "gauss_center", boom)
    st = {"radius": 64, "center_method": "gauss"}
    diag = {}
    out, (xc, yc) = geometry.center_cube(_cube(), st, np.random.default_rng(1), diag=diag)
    assert diag["center_used"] == "idl"
    assert diag["center_gauss"] is None
    assert np.isfinite(xc) and np.isfinite(yc)
    assert out.shape == (6, 128, 128)


def test_center_fails_only_when_both_fail(monkeypatch):
    """CenterFailed is raised when neither estimator finds the star."""
    def boom(med):
        raise RuntimeError("no star")
    monkeypatch.setattr(geometry, "gauss_center", boom)
    monkeypatch.setattr(geometry, "idl_center", boom)
    with pytest.raises(geometry.CenterFailed):
        geometry.center_cube(_cube(), {"radius": 64, "center_method": "gauss"}, np.random.default_rng(1), diag={})


def test_replace_interior_nans_with_finite_column_zero():
    """A frame whose column 0 is finite is filled instead of raising; an all-NaN frame passes through."""
    f = np.full((20, 20), np.nan)
    f[2:18, 0:15] = 1.0
    f[8, 7] = np.nan                         # interior hole
    out = binning.replace_interior_nans(f)
    assert np.isfinite(out[8, 7])
    assert np.isnan(out[0, 5])               # below the first good row stays NaN
    allnan = np.full((5, 5), np.nan)
    assert np.isnan(binning.replace_interior_nans(allnan)).all()


def test_sky_round_missing_a_chop_position():
    """Mean skies leave out an empty chop position, and the sky step uses the nearest round with it."""
    frames = np.ones((4, 512, 512), np.float32)
    sk1 = chopsub.mean_skies(frames, ["CHOP_A"] * 4)
    assert "CHOP_A" in sk1 and "CHOP_B" not in sk1
    sk1["avginum"] = 10.0
    sk2 = {"CHOP_B": np.full((512, 512), 2.0), "CHOP_A": np.full((512, 512), 3.0), "avginum": 100.0}
    frame = np.full((512, 512), 5.0)
    # a chop A frame takes the chop B sky (flip False): round 1 lacks it, so round 2 is used
    assert np.allclose(chopsub.subtract_sky(frame, "CHOP_A", 12.0, [sk1, sk2], False), 3.0)
    # no round has the needed position: the frame is returned unchanged
    assert np.allclose(chopsub.subtract_sky(frame, "CHOP_A", 12.0, [sk1], False), 5.0)


def test_median_or_nan_on_empty_cube():
    """An empty cube gives an all-NaN median frame of the right shape."""
    m = pipeline._median_or_nan(np.zeros((0, 8, 8), np.float32))
    assert m.shape == (8, 8) and np.isnan(m).all()
    c = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
    assert np.allclose(pipeline._median_or_nan(c), np.median(c, axis=0))
