"""Tests of the pre-binning by index and of the memory-aware worker count."""
import warnings

import numpy as np

from reduce_nomic import binning, idl, pipeline


def _cube(m=23, ny=50, nx=40, seed=3):
    """Make a test cube with NaNs, including an all-NaN column of pixels.

    Parameters
    ----------
    m : int, default 23
        Number of frames.
    ny, nx : int, default 50 and 40
        Frame size in pixels.
    seed : int, default 3
        Seed of the random generator.

    Returns
    -------
    cube : ndarray, shape (m, ny, nx)
        Float32 frames of unit Gaussian noise, with 5 % of the values and the pixel (0, 0) of
        every frame set to NaN.
    """
    rng = np.random.default_rng(seed)
    cube = rng.normal(size=(m, ny, nx)).astype(np.float32)
    cube[rng.random(cube.shape) < 0.05] = np.nan
    cube[:, 0, 0] = np.nan
    return cube


def test_median_rows_matches_idl_median():
    """The median taken by blocks of rows equals the IDL median of the whole group."""
    cube = _cube()
    for idx in (np.arange(23), np.arange(0, 23, 2), np.array([1, 4, 5, 9])):
        for block in (7, 128):
            got = binning.median_rows(cube, idx, block=block)
            want = idl.median(cube[idx], axis=0, even=True)
            assert got.dtype == want.dtype
            assert np.array_equal(got, want, equal_nan=True)


def test_prebin_by_index_matches_binning_a_copy():
    """Binning the kept frames by index gives the same bins as binning a copy of them."""
    cube = _cube()
    keep = np.setdiff1d(np.arange(23), [2, 3, 11, 17])      # 19 frames: bins of 5, 5, 5, 4
    a = np.linspace(10.0, 30.0, 23)[keep]
    e = np.full(23, 0.08)[keep]
    for kind in ("median", "mean"):
        for n in (5, 9, 1):                                   # 9 leaves a one-frame last bin
            st = {"prebin": n, "prebin_type": kind}
            with warnings.catch_warnings():               # the all-NaN pixel's mean
                warnings.simplefilter("ignore", RuntimeWarning)
                got = binning.prebin(cube, a, e, st, index=keep)
                want = binning.prebin(cube[keep], a, e, st)
            for g, w in zip(got, want):
                assert np.asarray(g).shape == np.asarray(w).shape
                assert np.array_equal(np.asarray(g), np.asarray(w), equal_nan=True)


def test_mem_workers_caps_and_logs(monkeypatch):
    """A stage keeps its workers when they fit and is capped (and logged) when they do not."""
    msgs = []
    monkeypatch.setattr(pipeline, "log", msgs.append)
    assert pipeline._mem_workers(2, 1.0, "stage") == 2
    assert msgs == []
    assert pipeline._mem_workers(64, 2.0 ** 60, "stage") == 1
    assert len(msgs) == 1 and msgs[0].startswith("stage: 1 worker instead of 64")
