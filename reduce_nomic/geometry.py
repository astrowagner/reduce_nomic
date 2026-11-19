"""Stages 3-5, split, align and center: cut each chop position around the star and center it.

These are the ``do_split``, ``do_align`` and ``do_center`` steps of the IDL predecessor
(reduce_nomic.pro lines 5381-6493). They run per subcube ("chunk") of ``frame_limit``
consecutive chop-subtracted frames, as in IDL: chop A frames become cube 1 and chop B frames
cube 2, each cut to 2 radius x 2 radius pixels around the star, aligned by cross-correlation
with a Gaussian, and shifted so the star centroid sits at (radius - 0.5, radius - 0.5).
Positions are in the IDL padded frame (``framew`` x ``framew``, the 512x512 detector at offset
(framew - 512) / 2) and given as ``(x, y)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi

from . import idl


def padded(frame: np.ndarray, framew: int) -> np.ndarray:
    """Embed a 512x512 frame in a NaN frame of ``framew`` pixels (IDL ``pad_frame``).

    Parameters
    ----------
    frame : ndarray, shape (512, 512)
        Frame.
    framew : int
        Padded size in pixels.

    Returns
    -------
    out : ndarray, shape (framew, framew)
        NaN everywhere except the frame, at offset (framew - 512) // 2 in x and y.
    """
    off = (framew - 512) // 2
    out = np.full((framew, framew), np.nan)
    out[off:off + 512, off:off + 512] = frame
    return out


def smooth_edge_truncate(a: np.ndarray, width: int) -> np.ndarray:
    """Boxcar-smooth as IDL ``SMOOTH(a, width, /NAN, /EDGE_TRUNCATE)``.

    Parameters
    ----------
    a : ndarray, shape (ny, nx)
        Image.
    width : int
        Box width; an even width is increased by one.

    Returns
    -------
    out : ndarray, shape (ny, nx)
        NaN-ignoring box means, with the image's edge values repeated beyond its edges.
    """
    w = width + 1 if width % 2 == 0 else width
    good = np.isfinite(a)
    s = ndi.uniform_filter(np.where(good, a, 0.0), size=w, mode="nearest")
    c = ndi.uniform_filter(good.astype(np.float64), size=w, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = s / c
    out[c < 0.5 / (w * w)] = np.nan
    return out


def find_positions(first: np.ndarray, second: np.ndarray, check: np.ndarray, st: Dict) -> Tuple[List[int], List[int]]:
    """Find the star in a chunk's first two frames, as IDL ``auto_positions`` does.

    Parameters
    ----------
    first, second : ndarray, shape (framew, framew)
        The chunk's first two padded frames.
    check : ndarray, shape (framew, framew)
        The chunk's first chop A frame, padded.
    st : dict
        Settings ``auto_positions``, ``pos1``, ``pos2``, ``filter_auto_pos``, ``double_sided``,
        ``framew``, ``test_pos_negative``.

    Returns
    -------
    pos1, pos2 : list of int
        ``[x, y]`` of the chop A and chop B star images. With ``test_pos_negative`` they are
        swapped when the high-passed chop A frame is negative at ``pos1``.
    """
    fw = int(st["framew"])
    if st["auto_positions"]:
        f1 = np.nan_to_num(first, nan=0.0)
        f2 = np.nan_to_num(second, nan=0.0)
        if st["filter_auto_pos"]:
            zero = f1 == 0          # IDL uses the first frame's zero map for both frames
            f1[zero] = np.nan
            f2[zero] = np.nan
            f1 = idl.smooth(f1 - smooth_edge_truncate(f1, 25), 5, nan=True)
            f2 = idl.smooth(f2 - smooth_edge_truncate(f2, 25), 5, nan=True)
            f1[zero] = 0.0
            f2[zero] = 0.0
        if st["double_sided"]:
            off = (fw - 512) // 2
            f1[off + 190:off + 336, :] = 0.0
            f2[off + 190:off + 336, :] = 0.0
        y1, x1 = divmod(int(np.nanargmax(f1)), fw)
        y2, x2 = divmod(int(np.nanargmax(f2)), fw)
        pos1, pos2 = [x1, y1], [x2, y2]
    else:
        pos1, pos2 = list(st["pos1"]), list(st["pos2"])
    c = idl.smooth(check - idl.smooth(check, 25, nan=True), 5, nan=True)
    if st["test_pos_negative"] and c[pos1[1], pos1[0]] < 0:
        pos1, pos2 = pos2, pos1
    return pos1, pos2


def edge_checked(pos: List[int], prev: Optional[List[int]], radius: int) -> List[int]:
    """Revert a coordinate whose cut would leave the padded frame (IDL ``auto_positions_edgecheck``).

    Parameters
    ----------
    pos : list of int
        ``[x, y]`` found for this chunk.
    prev : list of int or None
        The position used for the previous chunk, in the IDL loop order (chop position outer,
        chunk inner); None for the first.
    radius : int
        Half-size of the cut in pixels.

    Returns
    -------
    out : list of int
        ``pos`` with each coordinate outside [radius, 2 radius] replaced by the previous one.
    """
    out = list(pos)
    for k in (0, 1):
        if (out[k] - radius < 0 or out[k] - radius > radius) and prev is not None:
            out[k] = prev[k]
    return out


def cut_chop(frames: np.ndarray, chops: Sequence[str], which: str, pos_raw: Tuple[List[int], List[int]],
             pos_cut: List[int], st: Dict) -> np.ndarray:
    """Cut one chop position's frames around its star (IDL split stage).

    Parameters
    ----------
    frames : ndarray, shape (n, 512, 512)
        The chunk's corrected frames, where n is their number.
    chops : sequence of str
        Chop position of each frame.
    which : str
        ``CHOP_A`` (cube 1) or ``CHOP_B`` (cube 2).
    pos_raw : tuple of list of int
        ``(pos1, pos2)`` as found; the other chop's star image within ``blockrad`` of it is set
        to NaN (the negative image left by the chop subtraction).
    pos_cut : list of int
        ``[x, y]`` to cut around, after the edge check.
    st : dict
        Settings ``framew``, ``radius``, ``blockrad``.

    Returns
    -------
    cube : ndarray, shape (m, 2 radius, 2 radius)
        The cut frames, where m is the number of frames in that chop position; pixels outside
        the detector are NaN.
    """
    fw, r = int(st["framew"]), int(st["radius"])
    off = (fw - 512) // 2
    ch = np.asarray(chops)
    sel = np.nonzero(ch == which)[0]
    neg = pos_raw[1] if which == "CHOP_A" else pos_raw[0]
    yy, xx = np.mgrid[0:fw, 0:fw]
    disk = np.hypot(xx - neg[0], yy - neg[1]) < float(st["blockrad"])
    x0, y0 = pos_cut[0] - r, pos_cut[1] - r
    out = np.full((sel.size, 2 * r, 2 * r), np.nan, dtype=np.float32)
    for k, i in enumerate(sel):
        p = padded(frames[i], fw)
        p[disk] = np.nan
        out[k] = p[y0:y0 + 2 * r, x0:x0 + 2 * r]
    return out


def gaussian_reference(radius: int) -> np.ndarray:
    """Build the align stage's reference image exactly as the IDL predecessor does.

    Parameters
    ----------
    radius : int
        Half-size of the cut in pixels.

    Returns
    -------
    ref : ndarray, shape (2 radius, 2 radius)
        ``psf_gaussian(NPIX=2 radius, FWHM=14)``, shifted by ``fshift`` to the ``cntrd``
        position found near (500, 500), then shifted so its maximum is at (radius, radius). The
        hard-coded 500 makes ``cntrd`` fail for radius 400, and the reference then ends up
        centered at (400.5, 400.5), as in IDL.
    """
    n = 2 * radius
    ref = idl.psf_gaussian(n, 14.0).astype(np.float64)
    xc, yc = idl.cntrd(ref, 500, 500, 14.0)
    ref = idl.fshift(ref, 500.0 - xc, 500.0 - yc)
    iy, ix = divmod(int(np.argmax(ref)), n)
    return idl.fshift(ref, float(radius - ix), float(radius - iy))


def align_cube(cube: np.ndarray, st: Dict, ref: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Align a cut cube on the Gaussian reference (IDL align stage).

    Parameters
    ----------
    cube : ndarray, shape (m, 2 radius, 2 radius)
        Cut frames, where m is their number.
    st : dict
        Settings ``radius``, ``align_filter``, ``align_filter_width``, ``align_smooth``,
        ``align_smooth_width``, ``align_radius_thresh``.
    ref : ndarray, shape (2 radius, 2 radius), optional
        Reference image; default ``gaussian_reference(radius)``.

    Returns
    -------
    aligned : ndarray, shape (m, 2 radius, 2 radius)
        Each frame shifted by ``fshift(-dx, -dy)``, where (dx, dy) come from the subpixel peak of
        ``crosscorr(ref, work, /CUT)`` as in IDL. The work image is the frame clipped at 0 (NaN
        counting as 0), high-passed, smoothed, with rows 0-20 and everything beyond
        ``align_radius_thresh`` of the center set to 0. A float32 input cube is overwritten
        and returned, to save memory.
    shifts : ndarray, shape (m, 2)
        The applied shifts ``(-dx, -dy)`` in pixels.
    """
    r = int(st["radius"])
    if ref is None:
        ref = gaussian_reference(r)
    yy, xx = np.mgrid[0:2 * r, 0:2 * r]
    outside = np.hypot(xx - r, yy - r) > float(st["align_radius_thresh"])
    out = cube if cube.dtype == np.float32 else np.empty(cube.shape, dtype=np.float32)
    shifts = np.zeros((cube.shape[0], 2))
    for i in range(cube.shape[0]):
        frame = np.asarray(cube[i], dtype=np.float64)
        w = np.where(np.isfinite(frame), np.maximum(frame, 0.0), 0.0)
        if st["align_filter"]:
            w = w - idl.smooth(w, int(st["align_filter_width"]), nan=True)
        if st["align_smooth"]:
            w = idl.smooth(w, int(st["align_smooth_width"]), nan=True)
        w[0:21, :] = 0.0
        w[outside] = 0.0
        _, (px, py) = idl.crosscorr(ref, w, cut=True, pmax=True)
        dx = r - px if px >= 0 else -r + abs(px)
        dy = r - py if py >= 0 else -r + abs(py)
        out[i] = idl.fshift(frame, -dx, -dy)
        shifts[i] = (-dx, -dy)
    return out, shifts


def center_cube(cube: np.ndarray, st: Dict, rng: np.random.Generator, n_draw: int = 100) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Center an aligned cube on its median star (IDL center stage).

    Parameters
    ----------
    cube : ndarray, shape (m, 2 radius, 2 radius)
        Aligned frames, where m is their number.
    st : dict
        Settings ``radius``.
    rng : numpy.random.Generator
        Random generator for drawing the frames of the median; IDL draws 100 frames with
        replacement from ``randomu``, so its draw cannot be reproduced exactly.
    n_draw : int, default 100
        Number of frames drawn; 0 uses every frame.

    Returns
    -------
    centered : ndarray, shape (m, 2 radius, 2 radius)
        Every frame shifted by ``fshift(radius - x - 0.5, radius - y - 0.5)``, where (x, y) is
        the ``cntrd`` centroid (FWHM 7) of the high-passed median, near its maximum within 50 px
        of the center. The median is taken over the central 256x256 pixels only, which holds
        every pixel the 50-px zone and the 25-px smoothing use, so the result is unchanged. A
        float32 input cube is overwritten and returned.
    center : tuple of float
        The centroid (x, y) found, in pixels.
    """
    r = int(st["radius"])
    m = cube.shape[0]
    idx = np.arange(m) if n_draw == 0 else np.clip(idl.idl_round(m * rng.random(n_draw)), 0, m - 1)
    h = min(128, r)
    o = r - h
    med = idl.median(cube[idx, o:o + 2 * h, o:o + 2 * h], axis=0, even=True)
    med = med - idl.smooth(med, 25, nan=True)
    yy, xx = np.mgrid[0:2 * h, 0:2 * h]
    med[np.hypot(xx - h, yy - h) > 50] = 0.0
    iy, ix = divmod(int(np.nanargmax(med)), 2 * h)
    xc, yc = idl.cntrd(np.nan_to_num(med, nan=0.0), ix, iy, 7.0)
    if xc < 0:
        raise RuntimeError("cntrd failed in the center stage")
    xc, yc = xc + o, yc + o
    dx, dy = float(r) - xc - 0.5, float(r) - yc - 0.5
    out = cube if cube.dtype == np.float32 else np.empty(cube.shape, dtype=np.float32)
    for i in range(m):
        out[i] = idl.fshift(np.asarray(cube[i], dtype=np.float64), dx, dy)
    return out, (xc, yc)
