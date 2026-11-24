"""Sigma-clip the contrasts of one separation over position angle, for the Procyon curves and bands.

The clip is iterative: values more than ``sigma`` standard deviations (ddof 0) from the median of
the values kept are dropped, and the median and standard deviation are taken again from the rest
until no value changes side. This is astropy's ``sigma_clip(v, sigma=sigma, maxiters=None,
cenfunc="median", stdfunc="std")``, written out so that the scripts need numpy alone.
"""
import numpy as np


def sigma_clip_mask(v: np.ndarray, sigma: float) -> np.ndarray:
    """Return which contrasts survive an iterative median-centered sigma clip.

    Parameters
    ----------
    v : ndarray, shape (m,)
        Finite contrasts at one separation, one per position angle, where m is their number.
    sigma : float
        Clipping threshold, in standard deviations (ddof 0) of the values kept.

    Returns
    -------
    keep : ndarray of bool, shape (m,)
        True for the values kept.

    Raises
    ------
    ValueError
        If `sigma` is not positive or `v` holds a non-finite value.
    """
    v = np.asarray(v, float)
    if not sigma > 0:
        raise ValueError(f"sigma must be positive, not {sigma}")
    if not np.isfinite(v).all():
        raise ValueError("the contrasts must be finite")
    keep = np.ones(v.size, bool)
    while keep.any():
        center, scale = np.median(v[keep]), np.std(v[keep])
        new = np.abs(v - center) <= sigma * scale
        if (new == keep).all():
            break
        keep = new
    return keep
