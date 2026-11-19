"""Tests of the chop identification's crude star image and its destriping."""
import os

import numpy as np

from reduce_nomic import combine, idl
from reduce_nomic.config import load_settings


def _settings(**over):
    """Return the default settings (as a dict) with overrides.

    Parameters
    ----------
    **over
        Settings to change.

    Returns
    -------
    st : dict
        Settings for ``combine.crude_star_image``.
    """
    st = dict(load_settings("Fake_1", nights_json="/nonexistent/no_nights.json", overrides={}).values)
    st.update(over)
    return st


def _star(x0, y0, amp, fwhm=15.0):
    """Return a 512 x 512 Gaussian star image.

    Parameters
    ----------
    x0, y0 : float
        Center in pixels.
    amp : float
        Peak in ADU.
    fwhm : float, default 15.0
        FWHM in pixels.

    Returns
    -------
    img : ndarray, shape (512, 512)
        The star.
    """
    yy, xx = np.mgrid[0:512, 0:512]
    sd = fwhm / 2.3548
    return amp * np.exp(-0.5 * ((xx - x0) ** 2 + (yy - y0) ** 2) / sd ** 2)


def test_destripe_removes_row_offsets_and_keeps_the_star():
    """Check that destriping zeroes each half-row's median and leaves the star's peak in place."""
    rng = np.random.default_rng(5)
    st = _settings()
    img = rng.normal(0, 1, (512, 512)) + _star(140.3, 145.6, 40.0)
    img[:, :256] += rng.normal(0, 5, 512)[:, None]
    img[:, 256:] += rng.normal(0, 5, 512)[:, None]
    out = combine.destripe_rows(img, st)
    assert np.allclose(np.median(out[:, 21:245], axis=1), 0, atol=1e-9)
    assert np.allclose(np.median(out[:, 331:490], axis=1), 0, atol=1e-9)
    assert np.unravel_index(np.argmax(idl.smooth(out, 7)), out.shape) == (146, 140)
    assert abs(out[146, 140] - img[146, 140] + np.median(img[146, 21:245])) < 1e-9


def test_destripe_finds_a_star_a_channel_edge_stripe_hides():
    """Check that a bright stripe just outside the channel-edge mask wins without destriping only."""
    rng = np.random.default_rng(7)
    frame = 9000 + rng.normal(0, 2, (512, 512)) + _star(140.0, 160.0, 25.0)
    frame[114:118, 21:245] += 30.0            # stripe next to the rows 120-135 the masks blank
    bg = np.zeros((512, 512))
    bad = np.ones((512, 512))
    idl_way = combine.crude_star_image(frame, bg, bad, _settings(chopid_destripe=False))
    new_way = combine.crude_star_image(frame, bg, bad, _settings(chopid_destripe=True))
    y0, x0 = np.unravel_index(np.argmax(idl_way), idl_way.shape)
    y1, x1 = np.unravel_index(np.argmax(new_way), new_way.shape)
    assert 112 <= y0 <= 119                    # on the stripe
    assert abs(x1 - 140) <= 1 and abs(y1 - 160) <= 1


def test_destripe_off_reproduces_the_idl_image_and_spares_the_stripe_method():
    """Check that chopid_destripe off (or absent) gives the IDL image, and via_bg ignores it."""
    rng = np.random.default_rng(9)
    frame = 9000 + rng.normal(0, 3, (512, 512)) + _star(380.0, 150.0, 60.0)
    bg = rng.normal(0, 1, (512, 512))
    bad = np.ones((512, 512))
    st_off = _settings(chopid_destripe=False)
    st_absent = {k: v for k, v in st_off.items() if k != "chopid_destripe"}
    a = combine.crude_star_image(frame, bg, bad, st_off)
    b = combine.crude_star_image(frame, bg, bad, st_absent)
    assert np.array_equal(a, b)
    c = combine.crude_star_image(frame, bg, bad, _settings(chopid_destripe=False, set_chop_via_bg=1))
    d = combine.crude_star_image(frame, bg, bad, _settings(chopid_destripe=True, set_chop_via_bg=1))
    assert np.array_equal(c, d)


def test_frame_states_keeps_the_last_chop_image(tmp_path):
    """Check that frame_states hands back the image the last kept frame's position came from."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_pipeline_synthetic import make_night
    from reduce_nomic.frames import list_raw
    root = str(tmp_path / "archive")
    make_night(root, n=40)
    files = list_raw(os.path.join(root, "Fake_1", "raw"))
    st = _settings(background_limit=50000.0)
    init = combine.initial_background(files)
    bad = np.ones((512, 512))
    images = {}
    states = combine.frame_states(files, 10, 25, init["bg"], bad, st, images=images)
    pos, img, bsub = images["last"]
    assert pos == 24 and img.shape == bsub.shape == (512, 512)
    s = states[pos - 10]
    assert np.unravel_index(np.argmax(img), img.shape) == (s.star_y, s.star_x)
    plain = combine.frame_states(files, 10, 25, init["bg"], bad, st)
    assert [(q.star_x, q.star_y) for q in plain] == [(q.star_x, q.star_y) for q in states]


def test_reject_no_star_skips_frames_far_below_the_median():
    """Check that frames with a peak S/N far below the round's median are skipped and flagged."""
    from reduce_nomic.combine import FrameState
    states = [FrameState(pos=i, num=i, inum="", date="", time_end="", angle=0.0, dit=0.1, objname="x",
                         median=0.0, star_x=140, star_y=145, snr=400.0 + i) for i in range(20)]
    states[3].snr = 8.0
    states[4].snr = 60.0
    states[5].star_x = states[5].star_y = -1          # over the background limit: untouched
    states[5].snr = float("nan")
    n = combine.reject_no_star(states, 0.2)
    assert n == 2 and states[3].nostar and states[4].nostar and not states[6].nostar
    assert (states[3].star_x, states[3].star_y) == (-1, -1) and states[6].star_x == 140
    assert combine.reject_no_star(states, 0.0) == 0


def test_frame_states_measures_the_peak_snr(tmp_path):
    """Check that frame_states records a finite, large peak S/N for frames with a star."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_pipeline_synthetic import make_night
    from reduce_nomic.frames import list_raw
    root = str(tmp_path / "archive")
    make_night(root, n=30)
    files = list_raw(os.path.join(root, "Fake_1", "raw"))
    st = _settings(background_limit=50000.0)
    init = combine.initial_background(files)
    states = combine.frame_states(files, 6, 20, init["bg"], np.ones((512, 512)), st)
    snr = np.array([q.snr for q in states])
    assert np.all(np.isfinite(snr)) and snr.min() > 20
    assert combine.reject_no_star(states, 0.2) == 0
