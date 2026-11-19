"""Stages 6 and 9, pre-clean and clean: reject frames by correlation, noise and peak.

These are the ``do_pre_clean`` and ``do_clean`` steps of the IDL predecessor (reduce_nomic.pro
lines 6494-6973 and 7422-7787). Both measure, for each frame, its peak correlation with the
running mean PSF, the background noise away from the star's columns, and its maximum pixel; a
frame is rejected when any measure falls outside its threshold relative to the cube's mean.
With ``peak_scale`` each kept frame is divided by its maximum pixel, as in IDL.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from . import idl


@dataclass
class Selection:
    """Frame measures and the frames kept.

    Parameters
    ----------
    corrs : ndarray, shape (m,)
        Maximum of ``crosscorr`` between each frame's central 128x128 pixels and the mean of the
        frames within +-50 of it, where m is the number of frames.
    noises : ndarray, shape (m,)
        Standard deviation of each frame outside columns radius - 50 .. radius + 49.
    coronoises : ndarray, shape (m,)
        Standard deviation of the central 128x128 pixels within 0.3-0.65 arcsec of the center,
        NaN unless ``pre_coronoise_clean`` is on in the pre-clean.
    peaks : ndarray, shape (m,)
        Maximum pixel of each frame.
    bad : ndarray of int, shape (k,)
        Indices of rejected frames, sorted and unique, where k is their number.
    keep : ndarray of int, shape (m - k,)
        Indices of kept frames.
    """

    corrs: np.ndarray
    noises: np.ndarray
    coronoises: np.ndarray
    peaks: np.ndarray
    bad: np.ndarray
    keep: np.ndarray


def _ring(hs: int) -> np.ndarray:
    """Return the 0.3-0.65 arcsec ring of a central crop, the pre-clean's coronoise region.

    Parameters
    ----------
    hs : int
        Half size of the crop in pixels; the crop is 2 hs by 2 hs with the star at (hs, hs).

    Returns
    -------
    ring : ndarray of bool, shape (2 hs, 2 hs)
        True from 0.3 to 0.65 arcsec of the star, at 0.0179 arcsec per pixel.
    """
    yy, xx = np.mgrid[0:2 * hs, 0:2 * hs]
    d = np.hypot(xx - hs, yy - hs)
    return (d >= 0.3 / 0.0179) & (d <= 0.65 / 0.0179)


def ring_noise(cube: np.ndarray, st: Dict) -> np.ndarray:
    """Return each frame's standard deviation in the 0.3-0.65 arcsec ring around the star.

    The pre-clean's coronoise measure (``pre_coronoise_clean``), computed for every frame and
    without the ``annular_clean`` weighting, for the frame tags handed to KLIP-TPE.

    Parameters
    ----------
    cube : ndarray, shape (m, 2 radius, 2 radius)
        Centered frames, star at (radius, radius) in the 0-based pixel grid, where m is their
        number.
    st : dict
        Setting ``radius``.

    Returns
    -------
    noise : ndarray, shape (m,)
        Sample standard deviation (ddof 1) of the ring's pixels in the central 128 x 128 pixels,
        NaN taken as 0 as in the pre-clean, in the cube's units.
    """
    r, hs = int(st["radius"]), 64
    ring = _ring(hs)
    sub = np.nan_to_num(np.asarray(cube)[:, r - hs:r + hs, r - hs:r + hs], nan=0.0)
    return np.array([float(np.std(f[ring], ddof=1)) for f in sub])


def measure_and_select(cube: np.ndarray, angles: Sequence[float], st: Dict, chop: int, pre: bool) -> Selection:
    """Measure every frame of a cube and pick the bad ones (IDL pre-clean or clean).

    Parameters
    ----------
    cube : ndarray, shape (m, 2 radius, 2 radius)
        Centered frames (pre-clean) or binned, replaced frames (clean), where m is their number.
    angles : sequence of float
        Parallactic angle of each frame in degrees, for the bad-angle windows.
    st : dict
        Night settings: ``radius``, ``annular_clean``, ``manual_bads``, and the thresholds of the
        stage, ``<p>corr_thresh_<chop>``, ``<p>noise_clean``, ``<p>noise_thresh_<chop>``,
        ``<p>peak_clean``, ``<p>peak_thresh_<chop>``, ``<p>clean_angs``, where <p> is ``pre_``
        for pre-clean and empty for clean; for the pre-clean also ``pre_coronoise_clean`` and
        ``pre_coronoise_thresh_<chop>``. The clean stage's ``coronoise_clean`` is not ported.
    chop : int
        1 for cube 1 (chop A), 2 for cube 2 (chop B).
    pre : bool
        True for the pre-clean thresholds, False for the clean thresholds.

    Returns
    -------
    sel : Selection
        The measures and the kept and rejected frames.
    """
    p = "pre_" if pre else ""
    r, hs = int(st["radius"]), 64
    sub = np.nan_to_num(cube[:, r - hs:r + hs, r - hs:r + hs], nan=0.0)
    if st["annular_clean"]:
        yy, xx = np.mgrid[0:2 * hs, 0:2 * hs]
        d = np.hypot(xx - hs, yy - hs)
        sub = sub.copy()
        sub[:, (d < 20) | (d > 50)] /= 1e7
    if not pre and st.get("coronoise_clean"):
        raise NotImplementedError("coronoise_clean (clean stage) is not ported")
    coro = bool(pre and st.get("pre_coronoise_clean"))
    if coro:
        ring = _ring(hs)
    m = cube.shape[0]
    corrs, noises, peaks = np.zeros(m), np.zeros(m), np.zeros(m)
    coronoises = np.full(m, np.nan)
    for j in range(m):
        lmed = sub[max(0, j - 50):min(m - 1, j + 50) + 1].mean(axis=0)
        c = float(np.max(idl.crosscorr(sub[j], lmed)))
        corrs[j] = c if np.isfinite(c) else 0.0
        img = cube[j].copy()
        img[:, r - 50:r + 50] = np.nan
        noises[j] = float(np.nanstd(img, ddof=1))
        peaks[j] = float(np.nanmax(cube[j]))
        if coro:
            coronoises[j] = float(np.std(sub[j][ring], ddof=1))
    bads: List[int] = list(np.nonzero(corrs < st[f"{p}corr_thresh_{chop}"])[0]) + list(st.get("manual_bads") or [])
    if st[f"{p}noise_clean"]:
        bads += list(np.nonzero(noises > st[f"{p}noise_thresh_{chop}"] * noises.mean())[0])
    if coro:
        bads += list(np.nonzero(coronoises > st[f"pre_coronoise_thresh_{chop}"] * np.nanmean(coronoises))[0])
    if st[f"{p}peak_clean"]:
        bads += list(np.nonzero(peaks / peaks.mean() < st[f"{p}peak_thresh_{chop}"])[0])
    if st[f"{p}clean_angs"]:
        ang = np.asarray(angles, dtype=float)
        for centre in (st["badang"], st["badangtwo"]):
            lo, hi = centre - st["badangrange"], centre + st["badangrange"]
            bads += list(np.nonzero((ang > lo) & (ang < hi))[0])
    bad = np.unique(np.asarray(bads, dtype=int))
    keep = np.setdiff1d(np.arange(m), bad)
    return Selection(corrs=corrs, noises=noises, coronoises=coronoises, peaks=peaks, bad=bad, keep=keep)


def peak_scaled(cube: np.ndarray) -> np.ndarray:
    """Divide each frame by its maximum pixel (IDL ``peak_scale``).

    Parameters
    ----------
    cube : ndarray, shape (m, ny, nx)
        Frames, where m is their number.

    Returns
    -------
    out : ndarray, shape (m, ny, nx)
        Each frame divided by its NaN-ignoring maximum.
    """
    return cube / np.nanmax(cube, axis=(1, 2))[:, None, None]
