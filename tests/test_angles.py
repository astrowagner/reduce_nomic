"""Tests of the handling of unusable parallactic angles and of the port's per-night decisions."""
import json
import os

import numpy as np
import pytest
from astropy.io import fits

from reduce_nomic import combine
from reduce_nomic.config import load_settings
from reduce_nomic.pipeline import reduce
from test_names import make_archive
from test_pipeline_synthetic import make_night

SYNTH = {"obj": "fakestar", "prebin": 5, "background_limit": 50000.0,
         "pre_corr_thresh_1": 0.5, "pre_corr_thresh_2": 0.5, "corr_thresh_1": 0.5, "corr_thresh_2": 0.5,
         "pre_noise_thresh_1": 3, "pre_noise_thresh_2": 3, "noise_thresh_1": 3, "noise_thresh_2": 3,
         "pre_peak_thresh_1": 0.5, "pre_peak_thresh_2": 0.5, "peak_thresh_1": 0.5, "peak_thresh_2": 0.5}


def state(i: int, angle: float, time_end: str = "05:00:00.000") -> combine.FrameState:
    """Make a minimal frame state.

    Parameters
    ----------
    i : int
        Position and frame number.
    angle : float
        LBT_PARA in degrees.
    time_end : str, default "05:00:00.000"
        TIME-END on 2023-02-05.

    Returns
    -------
    s : FrameState
        With a star found at (100, 100).
    """
    s = combine.FrameState(pos=i, num=i, inum=f"{i:06d}", date="2023-02-05", time_end=time_end, angle=angle,
                           dit=0.08, objname="x", median=0.0)
    s.star_x = s.star_y = 100
    return s


def test_valid_angle_and_sexagesimal():
    """999 and NaN are unusable; sexagesimal pointing values parse in their own unit."""
    assert combine.valid_angle(100.2) and combine.valid_angle(-180.0)
    assert not combine.valid_angle(999.0) and not combine.valid_angle(float("nan"))
    assert abs(combine.sexagesimal("11:01:50.88") - 11.0308) < 1e-6
    assert abs(combine.sexagesimal(" +56:22:57.36 ") - 56.3826) < 1e-6
    assert abs(combine.sexagesimal("-05:13:30") + 5.225) < 1e-9
    assert np.isnan(combine.sexagesimal("unknown"))


def test_header_coords(tmp_path):
    """The pointing is the median over the headers whose values parse."""
    paths = []
    for k, (ra, dec) in enumerate([("11:01:50.0", "+56:22:57"), ("11:01:52.0", "+56:22:59"), ("bad", "bad")]):
        h = fits.Header()
        h["LBT_RA"], h["LBT_DEC"] = ra, dec
        p = str(tmp_path / f"f{k}.fits")
        fits.PrimaryHDU(np.zeros((2, 2)), header=h).writeto(p)
        paths.append(p)
    ra, dec = combine.header_coords(paths)
    assert abs(ra - (11 + 1 / 60 + 51 / 3600)) < 1e-9 and abs(dec - (56 + 22 / 60 + 58 / 3600)) < 1e-9


def test_repair_angles_modes():
    """Unusable angles are recomputed, dropped as starless frames, or kept, as asked."""
    coords = (11.0308, 56.3826)
    st = [state(0, 102.0, "12:44:00.000"), state(1, 999.0, "12:44:01.000"), state(2, 101.9, "12:44:02.000"),
          state(3, float("nan"), "12:44:03.000")]
    angles, bad, action = combine.repair_angles(st, "recompute", coords, 32.7014, -109.889)
    assert bad == [1, 3] and action == "recomputed"
    want = combine.parallactic_angles([st[1], st[3]], *coords)
    assert angles[0] == 102.0 and angles[1] == want[0] and angles[3] == want[1] and 100 < angles[1] < 105
    st = [state(0, 102.0), state(1, 999.0)]
    angles, bad, action = combine.repair_angles(st, "drop", coords, 32.7014, -109.889)
    assert angles is None and action == "dropped" and st[1].nostar and st[1].star_x == -1 and not st[0].nostar
    st = [state(0, 102.0), state(1, 999.0)]
    assert combine.repair_angles(st, "keep", coords, 32.7014, -109.889)[2] == "kept" and not st[1].nostar
    st = [state(0, 102.0), state(1, 999.0)]
    assert combine.repair_angles(st, "recompute", (np.nan, np.nan), 32.7014, -109.889)[2] == "dropped"
    assert combine.repair_angles([state(0, 1.0)], "recompute", coords, 32.7014, -109.889) == (None, [], "none")
    with pytest.raises(ValueError):
        combine.repair_angles([state(0, 999.0)], "maybe", coords, 32.7014, -109.889)


@pytest.mark.parametrize("mode", ["recompute", "keep"])
def test_synthetic_night_with_placeholder_angles(tmp_path, mode):
    """With 999 angles in a synthetic night, recompute keeps every binned angle sane and keep does not."""
    root = str(tmp_path / "archive")
    make_night(root)
    raw = os.path.join(root, "Fake_1", "raw")
    for i in range(240):
        p = os.path.join(raw, f"n_230101_{i:06d}.fits")
        with fits.open(p, mode="update") as hdul:
            hdul[0].header["LBT_RA"] = "11:01:50.88"
            hdul[0].header["LBT_DEC"] = "+56:22:57.36"
            if i in (30, 31, 32):
                hdul[0].header["LBT_PARA"] = 999.0
    st = load_settings("Fake_1", nights_json=str(tmp_path / "none.json"), overrides=dict(SYNTH, archive=root, pa_invalid=mode))
    summary = reduce(st, "left", workers=2, out_dir=str(tmp_path / "out"))
    assert summary["bad_pa"]["bad"] == 3 and summary["bad_pa"]["first"] == 30 and summary["bad_pa"]["last"] == 32
    ang = np.concatenate([fits.getdata(str(tmp_path / "out" / f"Fake_1_left_angles{cn}_bin.fits")) for cn in (1, 2)])
    if mode == "recompute":
        assert summary["bad_pa"]["action"] == "recomputed" and abs(summary["bad_pa"]["ra_hours"] - 11.0308) < 1e-4
        assert np.all(np.abs(ang) <= 180)
    else:
        # the synthetic angles run from -20 to 25 degrees, so a bin beyond 30 is spoiled by a 999
        assert summary["bad_pa"]["action"] == "kept" and np.max(ang) > 30


def test_port_decisions_apply_after_the_idl_block(tmp_path):
    """A night's entry in the port's decisions file overrides its IDL block and shows in the source."""
    nights = make_archive(str(tmp_path))
    port = str(tmp_path / "port.json")
    json.dump({"Procyon_13": {"settings": {"frames": [3, 6], "radius": 99}, "reason": "test"}}, open(port, "w"))
    st = load_settings("Procyon_20230205", nights, {"archive": str(tmp_path)}, port_json=port)
    assert st.frames == [3, 6] and st.radius == 99
    assert st.source == "nights.json:Procyon_13 + port.json:Procyon_13"
    st = load_settings("Procyon_20230205", nights, {"archive": str(tmp_path), "radius": 7}, port_json=port)
    assert st.radius == 7


def test_beta_uma_8_decision_is_recorded():
    """The repository's decisions file gives beta UMa 2023-02-05 its full frame range."""
    st = load_settings("BetaUma_8", None, {"archive": "/nonexistent"})
    assert st.frames == [155443, 195100] and st.source.endswith("nights_port.json:BetaUma_8")
