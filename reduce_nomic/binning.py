"""Stages 7 and 8, pre-binning and replacement: bin frames in time and fill interior NaNs.

These are the ``do_pre_binning`` and ``do_replace`` steps of the IDL predecessor
(reduce_nomic.pro lines 6974-7421). Within each chunk, consecutive groups of ``prebin`` frames
are combined; the binned frames of all chunks are then concatenated, and NaNs inside each binned
frame's outermost good rows and columns are filled with ``fixpix``.
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

from . import idl


def prebin(cube: np.ndarray, angles: Sequence[float], exptimes: Sequence[float], st: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin one chunk's frames in groups of ``prebin`` (IDL pre-binning).

    Parameters
    ----------
    cube : ndarray, shape (m, ny, nx)
        Pre-cleaned frames of one chunk, where m is their number.
    angles : sequence of float
        Parallactic angle of each frame in degrees.
    exptimes : sequence of float
        Effective exposure time of each frame in seconds.
    st : dict
        Settings ``prebin`` (group size) and ``prebin_type`` (``median`` or ``mean``).

    Returns
    -------
    binned : ndarray, shape (b, ny, nx)
        One frame per group, where b = ceil(m / prebin): the IDL median with ``/EVEN`` (or the
        NaN-ignoring mean); the last group may be shorter, and a one-frame group is that frame.
    angles_b : ndarray, shape (b,)
        Mean angle of each group.
    exptimes_b : ndarray, shape (b,)
        Summed exposure time of each group.
    """
    n = int(round(st["prebin"]))
    a = np.asarray(angles, dtype=float)
    e = np.asarray(exptimes, dtype=float)
    if n <= 1:
        return cube, a, e
    out, aa, ee = [], [], []
    for s in range(0, cube.shape[0], n):
        g = cube[s:s + n]
        if g.shape[0] == 1:
            out.append(g[0])
        elif st["prebin_type"] == "median":
            out.append(idl.median(g, axis=0, even=True))
        else:
            out.append(np.nanmean(g, axis=0))
        aa.append(a[s:s + n].mean())
        ee.append(e[s:s + n].sum())
    return np.asarray(out), np.asarray(aa), np.asarray(ee)


def replace_interior_nans(frame: np.ndarray) -> np.ndarray:
    """Fill the NaNs inside a frame's outermost good rows and columns (IDL replace stage).

    Parameters
    ----------
    frame : ndarray, shape (ny, nx)
        Binned frame.

    Returns
    -------
    out : ndarray, shape (ny, nx)
        ``fixpix`` applied with every NaN marked bad, except those left of the last leading
        all-NaN column, below the last leading all-NaN row (row 0 when none), from the last
        column holding a finite value to the right, and from the last row holding one upward;
        those stay NaN, as in IDL.
    """
    good = np.isfinite(frame).astype(np.float64)
    col_any = good.sum(axis=0) > 0
    row_any = good.sum(axis=1) > 0
    if col_any[0]:
        raise ValueError("column 0 holds a finite pixel; the IDL replace step fails here")
    leftmost = int(np.argmax(col_any)) - 1
    bottommost = int(np.argmax(row_any)) - 1 if not row_any[0] else 0
    rightmost = int(np.nonzero(col_any)[0][-1])
    topmost = int(np.nonzero(row_any)[0][-1])
    bm = good.copy()
    bm[:, 0:leftmost + 1] = 1.0
    bm[0:bottommost + 1, :] = 1.0
    bm[:, rightmost:] = 1.0
    bm[topmost:, :] = 1.0
    return idl.FixPix(bm).apply(frame)
