"""Averages of parallactic angles that stay correct across the ±180° wrap.

A star that transits north of the zenith (declination above the site latitude, such as Vega at the
LBT) has a parallactic angle that passes through ±180° at transit. The IDL predecessor averages
the angles of a chop segment and of a bin arithmetically, so a group holding frames on both sides
of the wrap gets a wrong angle (for example -28.6° for frames at -179.9° and +179.5°). These
averages agree with the arithmetic mean exactly whenever the group does not straddle the wrap.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def mean_angle(angles: Sequence[float], ignore_nan: bool = False) -> float:
    """Average angles in degrees, correctly across the ±180° wrap.

    Parameters
    ----------
    angles : sequence of float
        Angles of one group of frames, degrees.
    ignore_nan : bool, default False
        True ignores NaN angles, like ``np.nanmean``; False returns NaN when any angle is NaN,
        like ``np.mean``.

    Returns
    -------
    mean : float
        The arithmetic mean when the angles span no more than 180° (bit-identical to
        ``np.mean`` or ``np.nanmean``); otherwise the mean of the angles taken relative to the
        first one within ±180°, wrapped to [-180, 180). NaN for an empty group or one with no
        finite angle.
    """
    full = np.asarray(angles, dtype=float).ravel()
    a = full[np.isfinite(full)] if ignore_nan else full
    if a.size == 0 or not np.all(np.isfinite(a)):
        return float("nan")
    if a.max() - a.min() <= 180.0:
        return float(np.nanmean(full) if ignore_nan else full.mean())
    d = (a - a[0] + 180.0) % 360.0 - 180.0
    return float((a[0] + d.mean() + 180.0) % 360.0 - 180.0)


def wrap_outliers(angles: Sequence[float], tol_deg: float = 1.0) -> np.ndarray:
    """Find binned frames whose angle was averaged across the ±180° wrap.

    A bin whose two neighbors lie on opposite sides of the wrap must have an angle on the short
    arc between them; one off that arc by more than ``tol_deg`` was averaged across the wrap
    (directly, or through a chop segment that was). A first or last bin is flagged when its one
    neighbor lies within e of ±180° and its step to that neighbor exceeds e, where
    e = max(5°, 5 × the median step).

    Parameters
    ----------
    angles : sequence of float
        Angles of the binned frames in time order, degrees, in [-180, 180].
    tol_deg : float, default 1.0
        How far outside the arc between its neighbors a bin may lie, degrees.

    Returns
    -------
    bad : ndarray of int, shape (k,)
        Indices of the flagged bins, where k is their number (often 0).
    """
    a = np.asarray(angles, dtype=float).ravel()
    n = a.size
    if n < 3:
        return np.zeros(0, dtype=int)
    step = (np.diff(a) + 180.0) % 360.0 - 180.0
    typical = float(np.median(np.abs(step)))
    bad = []
    for i in range(1, n - 1):
        lo, hi = a[i - 1], a[i + 1]
        if abs(hi - lo) <= 180.0:          # the neighbors do not straddle the wrap
            continue
        arc = (hi - lo + 180.0) % 360.0 - 180.0                 # signed short arc lo -> hi
        x = (a[i] - lo + 180.0) % 360.0 - 180.0                 # bin relative to lo
        t0, t1 = min(0.0, arc), max(0.0, arc)
        if x < t0 - tol_deg or x > t1 + tol_deg:
            bad.append(i)
    edge = max(5.0, 5.0 * typical)
    if abs(step[0]) > edge and abs(a[1]) > 180.0 - edge:
        bad.insert(0, 0)
    if abs(step[-1]) > edge and abs(a[-2]) > 180.0 - edge:
        bad.append(n - 1)
    return np.asarray(bad, dtype=int)
