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


def test_valid_angle_rejects_placeholders():
    """NaN, the 999 placeholder and an exact 0 (telescope dropout) are unusable; others are kept."""
    from reduce_nomic import combine
    assert not combine.valid_angle(float("nan"))
    assert not combine.valid_angle(999.0)
    assert not combine.valid_angle(0.0)
    assert combine.valid_angle(1e-8) and combine.valid_angle(-179.85) and combine.valid_angle(360.0)


def test_repair_angles_recomputes_a_dropout():
    """A frame with LBT_PARA 0 between valid ones is recomputed from its time and the pointing."""
    from reduce_nomic import combine
    states = [combine.FrameState(pos=i, num=56900 + i, inum=f"{56900 + i:06d}", date="2024-04-25",
                                 time_end=t, angle=a, dit=0.08, objname="bet Uma", median=21000.0)
              for i, (t, a) in enumerate([("05:13:44.6", 1.0), ("05:13:45.6", 0.0), ("05:13:46.6", 1.0)])]
    angles, bad, action = combine.repair_angles(states, "recompute", (11.0308, 56.3824), 32.7014, -109.889)
    assert bad == [1] and action == "recomputed"
    assert angles[0] == 1.0 and angles[2] == 1.0 and np.isfinite(angles[1]) and angles[1] != 0.0
