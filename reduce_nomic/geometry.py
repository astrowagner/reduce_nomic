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

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi

from . import binning, idl


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


def edge_checked(pos: List[int], prev: Optional[List[int]], radius: int, framew: Optional[int] = None) -> List[int]:
    """Revert a coordinate whose cut would leave the padded frame (IDL ``auto_positions_edgecheck``).

    The cut ``[pos - radius, pos + radius)`` fits in the padded frame when
    ``radius <= pos <= framew - radius``. IDL tests ``pos - radius < 0 or pos - radius > radius``,
    the same test only when ``framew = 3 radius``; for radius 400 in a 1500-px frame it rejects
    every chop B position of the double-sided Vega nights (x near 895), and its reverts then carry a
    single wrong position into every later chunk (``docs/PORTING.md``, "Split-stage positions").

    Parameters
    ----------
    pos : list of int
        ``[x, y]`` found for this chunk.
    prev : list of int or None
        The position used for the previous chunk, in the IDL loop order (chop position outer,
        chunk inner); None for the first.
    radius : int
        Half-size of the cut in pixels.
    framew : int, optional
        Size of the padded frame in pixels. None applies IDL's bounds, [radius, 2 radius].

    Returns
    -------
    out : list of int
        ``pos`` with each coordinate outside the bounds replaced by the previous one. With
        ``framew`` given and no previous position, such a coordinate is moved to the nearest
        bound instead, so that the cut stays inside the frame.
    """
    hi = 2 * radius if framew is None else int(framew) - radius
    out = list(pos)
    for k in (0, 1):
        if out[k] < radius or out[k] > hi:
            if prev is not None:
                out[k] = prev[k]
            elif framew is not None:
                out[k] = min(max(out[k], radius), hi)
    return out


def filter_positions(raw: Sequence[Sequence[Sequence[int]]], tol: float, half: int = 5) -> Tuple[List[List[List[int]]], List[Tuple[int, int]]]:
    """Replace a chunk's star position that disagrees with its chop throw (a port addition).

    The split stage finds both chop positions in each chunk's starting frames. In a chunk that
    starts in cloud or with the AO loop open the brightest point can be elsewhere: on Vega
    2022-05-21 left, chunk 89 gave chop B at (752, 497) while chop A, at (627, 579), stayed with
    its neighbors near (632, 579) and chop B's were near (899, 580). The chop throw (chop B minus
    chop A) stays fixed while the telescope's pointing drifts, so a pair whose throw agrees with
    its neighbors' is kept as found even when both positions moved (chunk 90 there: both 33 px
    off, throw unchanged), and a pair whose throw disagrees has a wrong position. That position is
    the one farther from its neighbors' median; it is rebuilt from the other chop's position and
    the neighbors' median throw, or both are replaced by their neighbors' medians when both are
    more than ``tol`` from them.

    Parameters
    ----------
    raw : sequence of [pos1, pos2]
        Per chunk, the ``[x, y]`` found for chop A and chop B (``find_positions``).
    tol : float
        Largest accepted difference in pixels, in x or in y, between a chunk's throw and its
        neighbors' median throw (and the distance from the neighbors' median position beyond which
        a position counts as wrong); 0 or less returns ``raw`` unchanged.
    half : int, default 5
        Number of neighboring chunks used on each side (the chunk itself excluded).

    Returns
    -------
    out : list of [pos1, pos2]
        ``raw`` with the wrong positions replaced, rounded to whole pixels. A run of fewer than 3
        chunks is returned unchanged.
    replaced : list of tuple of int
        ``(chunk, chop)`` of every replaced position, chop 1 for A and 2 for B.
    """
    out = [[list(p1), list(p2)] for p1, p2 in raw]
    replaced: List[Tuple[int, int]] = []
    if tol <= 0 or len(raw) < 3:
        return out, replaced
    arr = np.asarray([[p1, p2] for p1, p2 in raw], dtype=float)     # shape (nchunk, 2, 2)
    n = arr.shape[0]
    for c in range(n):
        nb = [j for j in range(max(0, c - half), min(n, c + half + 1)) if j != c]
        throw = np.median(arr[nb, 1, :] - arr[nb, 0, :], axis=0)
        if np.all(np.abs((arr[c, 1] - arr[c, 0]) - throw) <= tol):
            continue
        med = [np.median(arr[nb, k, :], axis=0) for k in (0, 1)]
        dev = [float(np.max(np.abs(arr[c, k] - med[k]))) for k in (0, 1)]
        if dev[0] > tol and dev[1] > tol:
            new = {0: med[0], 1: med[1]}
        elif dev[1] >= dev[0]:
            new = {1: arr[c, 0] + throw}
        else:
            new = {0: arr[c, 1] - throw}
        for k, v in new.items():
            out[c][k] = [int(round(v[0])), int(round(v[1]))]
            replaced.append((c, k + 1))
    return out, replaced


def split_cut_positions(raw: Sequence[Sequence[Sequence[int]]], st: Dict) -> Tuple[List[List[List[int]]], List[Tuple[int, int]], Dict[int, List[List[int]]]]:
    """Decide where the split stage cuts each chunk's two chop positions.

    The found positions are first filtered (``filter_positions``), then checked against the
    padded frame chunk by chunk in the IDL loop order, chop A's pass over every chunk before chop
    B's, with each chop's previous position carried through both passes as IDL's ``prevpos1`` and
    ``prevpos2`` are (``edge_checked``).

    Parameters
    ----------
    raw : sequence of [pos1, pos2]
        Per chunk, the ``[x, y]`` found for chop A and chop B (``find_positions``), in padded-frame
        pixels.
    st : dict
        Settings ``radius``, ``framew``, ``auto_positions_edgecheck``, ``edgecheck_idl_bounds`` and
        ``pos_outlier_px``.

    Returns
    -------
    use : list of [pos1, pos2]
        The positions after ``filter_positions``; they also say where the other chop's negative
        image is blocked.
    replaced : list of tuple of int
        ``(chunk, chop)`` replaced by ``filter_positions``.
    cut : dict
        ``{1: [...], 2: [...]}``: per chunk, the ``[x, y]`` cut center for chop A and chop B.
    """
    use, replaced = filter_positions(raw, float(st.get("pos_outlier_px") or 0.0))
    radius = int(st["radius"])
    fw = None if st.get("edgecheck_idl_bounds") else int(st["framew"])
    cut: Dict[int, List[List[int]]] = {1: [], 2: []}
    prev1 = prev2 = None
    for cn in (1, 2):
        for p1, p2 in use:
            if st.get("auto_positions_edgecheck", 1):
                q1, q2 = edge_checked(p1, prev1, radius, fw), edge_checked(p2, prev2, radius, fw)
            else:
                q1, q2 = list(p1), list(p2)
            prev1, prev2 = q1, q2
            cut[cn].append(q1 if cn == 1 else q2)
    return use, replaced, cut


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


def idl_center(med: np.ndarray) -> Tuple[float, float]:
    """Find the star in a median image as the IDL center stage does.

    Parameters
    ----------
    med : ndarray, shape (2 h, 2 h)
        Median of aligned frames around the star, where h is half its size (at least 50).

    Returns
    -------
    xc, yc : float
        The ``cntrd`` centroid (FWHM 7) of the image high-passed by ``smooth(., 25, /nan)``,
        started at its maximum within 50 px of the image center, in the image's pixels.

    Raises
    ------
    RuntimeError
        When ``cntrd`` fails.

    Notes
    -----
    The 7-pixel centroid of the high-passed image can lock onto either of two nearly equal
    local maxima of a broad or structured core; on Procyon_13 SX it moves the chunks' centers
    by up to 3.5 px. ``gauss_center`` is the robust alternative.
    """
    n = med.shape[0]
    h = n // 2
    hp = med - idl.smooth(med, 25, nan=True)
    yy, xx = np.mgrid[0:n, 0:n]
    hp[np.hypot(xx - h, yy - h) > 50] = 0.0
    iy, ix = divmod(int(np.nanargmax(hp)), n)
    xc, yc = idl.cntrd(np.nan_to_num(hp, nan=0.0), ix, iy, 7.0)
    if xc < 0:
        raise RuntimeError("cntrd failed in the center stage")
    return float(xc), float(yc)


def gauss_center(med: np.ndarray, half: int = 20) -> Tuple[float, float]:
    """Find the star in a median image from a Gaussian fit to its core.

    Parameters
    ----------
    med : ndarray, shape (2 h, 2 h)
        Median of aligned frames around the star, where h is half its size (at least 50).
    half : int, default 20
        Half size of the fitted box, pixels; it holds the core of a 15-18 px FWHM NOMIC PSF.

    Returns
    -------
    xc, yc : float
        Center of a circular Gaussian plus constant fitted by least squares to the
        (2 half + 1)^2 pixels around the maximum of the 5-pixel boxcar-smoothed image within
        50 px of the image center (NaN pixels left out; the box is clipped at the edges), in
        the image's pixels, with pixel centers at integers as in ``idl.cntrd``.

    Raises
    ------
    RuntimeError
        When the fit fails or its center lies more than ``half`` from the starting maximum.
    """
    from scipy.optimize import curve_fit
    n = med.shape[0]
    h = n // 2
    s = idl.smooth(np.nan_to_num(med, nan=0.0), 5)
    yy, xx = np.mgrid[0:n, 0:n]
    s[np.hypot(xx - h, yy - h) > 50] = -np.inf
    iy, ix = divmod(int(np.argmax(s)), n)
    y0, y1 = max(iy - half, 0), min(iy + half + 1, n)
    x0, x1 = max(ix - half, 0), min(ix + half + 1, n)
    sub = med[y0:y1, x0:x1]
    ok = np.isfinite(sub)
    if ok.sum() < 20:
        raise RuntimeError("too few finite pixels for the Gaussian center")
    gy, gx = np.mgrid[y0:y1, x0:x1]
    vals = sub[ok]

    def model(xy, amp, xc, yc, sig, off):
        """Circular Gaussian plus constant at the points xy = (x, y)."""
        return amp * np.exp(-((xy[0] - xc) ** 2 + (xy[1] - yc) ** 2) / (2.0 * sig ** 2)) + off

    base = float(np.median(vals))
    p0 = (float(vals.max()) - base, float(ix), float(iy), 7.0, base)
    try:
        p, _ = curve_fit(model, (gx[ok], gy[ok]), vals, p0=p0, maxfev=20000)
    except (RuntimeError, ValueError) as err:
        raise RuntimeError(f"Gaussian center fit failed: {err}") from err
    if not (np.all(np.isfinite(p)) and abs(p[1] - ix) <= half and abs(p[2] - iy) <= half):
        raise RuntimeError("Gaussian center fit left the fitted box")
    return float(p[1]), float(p[2])


class CenterFailed(RuntimeError):
    """Neither center estimator found the star in a chunk's median frame."""


def center_cube(cube: np.ndarray, st: Dict, rng: np.random.Generator, n_draw: int = 100,
                diag: Optional[Dict] = None) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Center an aligned cube on its median star (IDL center stage, or a Gaussian fit to the core).

    Parameters
    ----------
    cube : ndarray, shape (m, 2 radius, 2 radius)
        Aligned frames, where m is their number.
    st : dict
        Settings ``radius`` and ``center_method``: ``gauss`` (default) fits a circular Gaussian
        to the core of the median of every frame (``gauss_center``); ``idl`` takes the IDL
        center stage's centroid (``idl_center``) of the median of ``n_draw`` random frames.
    rng : numpy.random.Generator
        Random generator for the ``idl`` method's draw; IDL draws 100 frames with replacement
        from ``randomu``, so its draw cannot be reproduced exactly.
    n_draw : int, default 100
        Number of frames the ``idl`` method draws; 0 uses every frame.
    diag : dict, optional
        Filled with ``center_idl`` and ``center_gauss``: both estimators' (x, y) on the median
        used, in pixels (None where one fails), and ``center_used``: the estimator whose center
        was applied (``center_method``, or the other one when it failed), for the run's summary.

    Returns
    -------
    centered : ndarray, shape (m, 2 radius, 2 radius)
        Every frame shifted by ``fshift(radius - x - 0.5, radius - y - 0.5)``, where (x, y) is
        the chosen center. The median is taken over the central 256x256 pixels only, which
        hold every pixel both estimators use. A float32 input cube is overwritten and returned.
    center : tuple of float
        The center (x, y) used, in pixels.

    Raises
    ------
    ValueError
        For an unknown ``center_method``.
    CenterFailed
        When both estimators fail. When only the chosen one fails, the other one's center is
        used (a port decision: IDL has only its centroid, and a failed Gaussian fit on one chunk
        should not stop a night).
    """
    r = int(st["radius"])
    m = cube.shape[0]
    method = str(st.get("center_method", "gauss"))
    h = min(128, r)
    o = r - h
    crop = cube[:, o:o + 2 * h, o:o + 2 * h]
    if method == "idl":
        idx = np.arange(m) if n_draw == 0 else np.clip(idl.idl_round(m * rng.random(n_draw)), 0, m - 1)
        med = idl.median(crop[idx], axis=0, even=True)
    elif method == "gauss":
        med = binning.median_rows(crop, np.arange(m), block=32)
    else:
        raise ValueError(f"center_method must be 'gauss' or 'idl', not {method!r}")
    est: Dict[str, Optional[Tuple[float, float]]] = {}
    for name, fn in (("idl", idl_center), ("gauss", gauss_center)):
        try:
            x, y = fn(med)
            est[name] = (x + o, y + o)
        except RuntimeError:
            est[name] = None
    used = method if est[method] is not None else ("idl" if method == "gauss" else "gauss")
    if diag is not None:
        diag["center_idl"], diag["center_gauss"] = est["idl"], est["gauss"]
        diag["center_used"] = used if est[used] is not None else None
    if est[used] is None:
        raise CenterFailed("both center estimators failed in the center stage")
    xc, yc = est[used]
    dx, dy = float(r) - xc - 0.5, float(r) - yc - 0.5
    out = cube if cube.dtype == np.float32 else np.empty(cube.shape, dtype=np.float32)
    for i in range(m):
        out[i] = idl.fshift(np.asarray(cube[i], dtype=np.float64), dx, dy)
    return out, (xc, yc)
