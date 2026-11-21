"""Tests of reduce_nomic.angles: wrap-aware averages and the detection of wrongly averaged bins."""
import numpy as np

from reduce_nomic import binning
from reduce_nomic.angles import mean_angle, wrap_outliers


def test_mean_angle_matches_numpy_off_the_wrap():
    """Groups that do not straddle ±180° give exactly np.mean / np.nanmean."""
    rng = np.random.default_rng(3)
    for _ in range(200):
        a = rng.uniform(-179, 179, 1) + rng.normal(0, 2, rng.integers(2, 40))
        a = a[np.abs(a) < 180]
        if a.size < 2 or a.max() - a.min() > 180:
            continue
        assert mean_angle(a) == a.mean()
        b = a.copy()
        b[0] = np.nan
        assert np.isnan(mean_angle(b))
        assert mean_angle(b, ignore_nan=True) == np.nanmean(b)


def test_mean_angle_across_the_wrap():
    """Frames at -179.9 and +179.5 average near ±180, not near 0."""
    a = np.array([179.5] * 30 + [-179.9])
    m = mean_angle(a)
    want = 179.5 + 0.6 / 31          # -179.9 is 0.6 deg past 179.5
    assert abs(m - want) < 1e-9
    assert abs(((mean_angle([179.0, -179.0]) + 180.0) % 360.0) - 0.0) < 1e-9   # ±180
    assert np.isnan(mean_angle([np.nan, np.nan], ignore_nan=True))
    assert abs(mean_angle([np.nan, 179.0, -179.0], ignore_nan=True) + 180.0) < 1e-9


def test_prebin_angles_across_the_wrap():
    """binning.prebin gives the wrap-aware mean for a straddling group."""
    cube = np.zeros((4, 3, 3))
    ang = [178.0, 179.0, -179.0, -178.0]
    _, ab, _ = binning.prebin(cube, ang, [1.0] * 4, {"prebin": 4, "prebin_type": "mean"})
    assert abs(((ab[0] + 180.0) % 360.0)) < 1e-9


def test_wrap_outliers_finds_the_2022_05_21_bins():
    """The bins of Vega 2022-05-21 averaged across the wrap are found; clean sequences pass."""
    left = [-174.61, -176.25, -177.1, -178.29, -28.61, 179.38, 178.73, 177.2]
    right = [-174.71, -176.13, -176.93, -177.57, 168.12, 178.27, 176.65, 175.5]
    assert wrap_outliers(left).tolist() == [4]
    assert wrap_outliers(right).tolist() == [4]
    good = [-174.61, -176.25, -177.1, -178.29, -179.6, 179.38, 178.73, 177.2]
    assert wrap_outliers(good).size == 0
    gap = [-160.0, -165.0, -170.0, 170.0, 165.0]        # a 20-degree gap across the wrap
    assert wrap_outliers(gap).size == 0
    assert wrap_outliers(np.linspace(-10, 10, 50)).size == 0
    edge = [168.12, 178.27, 176.65, 175.5, 174.0]          # a wrongly averaged first bin
    assert wrap_outliers(edge).tolist() == [0]
