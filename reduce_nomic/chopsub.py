"""Stage 2, chop subtraction: background removal and cosmetic corrections of each chop frame.

This is the ``do_chop_sub`` step of the IDL predecessor (reduce_nomic.pro lines 4436-5380). Each
segment frame from stage 1 has the mean of its neighboring segments subtracted (opposite chop
positions), then the detector's channel seam, a polynomial background and bad pixels are
corrected, the channel edges and the frame border are set to NaN, and, on nights with sky
frames, the mean sky of the matching chop is subtracted. Frames stay 512x512 here; the IDL
padding to ``framew`` is applied by the split stage when cutting around the star.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from . import idl

#: Cosmetic options of the IDL predecessor that no night turns on and the port does not implement.
UNPORTED = ("priorpoly_bgsub", "midpoly_bgsub", "fix_bad_channel", "polyfit_edges",
            "subtract_crosstalk", "flatten_channels", "flatten_horizontal", "flatten_vertical",
            "mask_stripes", "zero_edges", "nan_bad_region", "pca_background")


def check_options(st: Dict) -> None:
    """Refuse settings that turn on a cosmetic option the port does not implement.

    Parameters
    ----------
    st : dict
        Night settings.

    Returns
    -------
    None
        Raises ``NotImplementedError`` naming the options that are on.
    """
    on = [k for k in UNPORTED if st.get(k)]
    if on:
        raise NotImplementedError(f"options not ported: {on}")


def chop_subtract(prev: Optional[np.ndarray], cur: np.ndarray, nxt: Optional[np.ndarray]) -> np.ndarray:
    """Subtract the neighboring chop frames (IDL ``nearest_neighbor_chop_sub``).

    Parameters
    ----------
    prev, nxt : ndarray, shape (512, 512) or None
        Previous and next segment frames; None at the ends of the round.
    cur : ndarray, shape (512, 512)
        The segment frame.

    Returns
    -------
    out : ndarray, shape (512, 512)
        ``cur`` minus the mean of its two neighbors, or minus the one neighbor at an end.
    """
    if prev is None:
        return cur - nxt
    if nxt is None:
        return cur - prev
    return cur - (prev + nxt) / 2.0


def normalize_middle(frame: np.ndarray, xww: int = 8, yww: int = 10) -> np.ndarray:
    """Remove the step between the left and right halves of the detector, row by row.

    Parameters
    ----------
    frame : ndarray, shape (512, 512)
        Frame.
    xww : int, default 8
        Columns 255 - xww .. 255 and 256 .. 256 + xww are compared (IDL ``xww``).
    yww : int, default 10
        Over rows y - yww .. y + yww (IDL ``yww``).

    Returns
    -------
    out : ndarray, shape (512, 512)
        Copy with the right half (x >= 256) of each row shifted by the difference of the two
        column means. Rows are corrected in order and later rows see the corrected earlier
        ones, as in the IDL loop.
    """
    out = np.array(frame, dtype=np.float64)
    ny = out.shape[0]
    sl = out[:, 255 - xww:256].sum(axis=1)
    sr = out[:, 256:257 + xww].sum(axis=1)
    ncol_l, ncol_r = xww + 1, xww + 1
    corr = np.zeros(ny)
    for y in range(ny):
        y0, y1 = max(0, y - yww), min(ny - 1, y + yww)
        m = y1 - y0 + 1
        left = sl[y0:y1 + 1].sum() / (ncol_l * m)
        right = (sr[y0:y1 + 1].sum() + ncol_r * corr[y0:y].sum()) / (ncol_r * m)
        corr[y] = left - right
    out[:, 256:] += corr[:, None]
    return out


def remove_oddeven(frame: np.ndarray, idl_mode: bool = True) -> np.ndarray:
    """Apply the IDL predecessor's ``remove_oddeven_noise`` step.

    Parameters
    ----------
    frame : ndarray, shape (ny, nx)
        Frame.
    idl_mode : bool, default True
        Reproduce IDL: because ``i+j mod 2`` parses as ``i + (j mod 2)``, every pixel except
        those with x = 0 and even y becomes the mean of its 2x2 block [x:x+1, y:y+1]; those take
        the mean of that and themselves. False applies the intended rule: block mean where
        x + y is odd, mean of block and pixel where it is even.

    Returns
    -------
    out : ndarray, shape (ny, nx)
        The corrected frame; the last row and column are unchanged.
    """
    img = np.asarray(frame, dtype=np.float64)
    out = img.copy()
    avg = (img[:-1, :-1] + img[:-1, 1:] + img[1:, :-1] + img[1:, 1:]) / 4.0
    yy, xx = np.mgrid[0:img.shape[0] - 1, 0:img.shape[1] - 1]
    odd = (xx + yy % 2) != 0 if idl_mode else ((xx + yy) % 2) == 1
    out[:-1, :-1] = np.where(odd, avg, (avg + img[:-1, :-1]) / 2.0)
    return out


def edge_masks(frame: np.ndarray, st: Dict) -> np.ndarray:
    """Set the channel edges, the frame border and the other aperture to NaN.

    Parameters
    ----------
    frame : ndarray, shape (512, 512)
        Frame.
    st : dict
        Settings ``nan_edges``, ``badw``, ``nan_exterior``, ``nan_exterior_extra``,
        ``double_sided``, ``side``, ``sx_top``, ``dx_bottom``.

    Returns
    -------
    out : ndarray, shape (512, 512)
        Copy with columns 255 +- badw and rows 127, 255, 383 +- badw (``nan_edges``), the outer
        badw + 1 pixels (``nan_exterior``), rows 0-75 and 200-256 (``nan_exterior_extra``), and
        the other aperture's rows (``double_sided``) set to NaN.
    """
    out = np.array(frame, dtype=np.float64)
    w = int(st["badw"])
    if st["nan_edges"]:
        out[:, 255 - w:256 + w] = np.nan
        for r in (127, 255, 383):
            out[r - w:r + w + 1, :] = np.nan
    if st["nan_exterior"]:
        out[0:w + 1, :] = np.nan
        out[:, 0:w + 1] = np.nan
        out[:, 511 - w:512] = np.nan
        out[511 - w:512, :] = np.nan
        if st["nan_exterior_extra"]:
            out[0:76, :] = np.nan
            out[200:257, :] = np.nan
    if st["double_sided"]:
        if st["side"] == "left":
            out[st["sx_top"]:512, :] = np.nan
        else:
            out[0:st["dx_bottom"] + 1, :] = np.nan
    return out


class Cosmetics:
    """The per-frame corrections after chop subtraction, for one bad-pixel map.

    Parameters
    ----------
    badmap : ndarray, shape (512, 512)
        Bad-pixel map (1 good, 0 bad) of the night and aperture.
    st : dict
        Night settings.
    """

    def __init__(self, badmap: np.ndarray, st: Dict):
        check_options(st)
        self.st = st
        self.fix = idl.FixPix(badmap) if st["replace_bads"] else None
        self.sfit = idl.SurfaceFit((512, 512), int(st["polydegree"])) if st["postpoly_bgsub"] else None

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Correct one chop-subtracted frame.

        Parameters
        ----------
        frame : ndarray, shape (512, 512)
            Chop-subtracted frame.

        Returns
        -------
        out : ndarray, shape (512, 512)
            After ``normalize_middle``, subtraction of ``sfit(frame, polydegree)``, bad-pixel
            replacement, ``remove_oddeven_noise``, a second bad-pixel replacement and the NaN
            masks, in the IDL order.
        """
        st = self.st
        f = np.asarray(frame, dtype=np.float64)
        if st["normalize_middle"]:
            f = normalize_middle(f, int(st["xww"]), int(st["yww"]))
        if self.sfit is not None:
            f = f - self.sfit.surface(f)
        if self.fix is not None:
            f = self.fix.apply(f)
        if st["remove_oddeven_noise"]:
            f = remove_oddeven(f, bool(st.get("idl_oddeven", True)))
        if self.fix is not None:
            f = self.fix.apply(f)
        return edge_masks(f, st)


def mean_skies(frames: np.ndarray, chops: Sequence[str]) -> Dict[str, np.ndarray]:
    """Combine one sky round into its mean sky for each chop position.

    Parameters
    ----------
    frames : ndarray, shape (n, 512, 512)
        The sky round's corrected frames, where n is their number; NaN frames are ignored.
    chops : sequence of str
        ``CHOP_A`` or ``CHOP_B`` for each frame.

    Returns
    -------
    skies : dict
        ``CHOP_A`` and ``CHOP_B``: per-pixel IDL median (upper middle value for an even count,
        NaN ignored) over that chop's frames, as IDL ``median(cube, dim=3)``.
    """
    ch = np.asarray(chops)
    return {c: idl.median(frames[ch == c], axis=0) for c in ("CHOP_A", "CHOP_B")}


def subtract_sky(frame: np.ndarray, chop: str, inum: float, skies: Sequence[Dict], flip: bool) -> np.ndarray:
    """Subtract the mean sky nearest in frame number (IDL ``skysub``).

    Parameters
    ----------
    frame : ndarray, shape (512, 512)
        Corrected science frame.
    chop : str
        The frame's chop position.
    inum : float
        The frame's IDL ``INUM`` as a number.
    skies : sequence of dict
        Per sky round: ``CHOP_A`` and ``CHOP_B`` mean skies and ``avginum``, the mean ``INUM``.
    flip : bool
        IDL ``flip_sky_chop``: subtract the same chop's sky; otherwise chop A takes the chop B
        sky and chop B the chop A sky, as IDL does by default.

    Returns
    -------
    out : ndarray, shape (512, 512)
        The frame minus the selected sky, whose NaNs count as 0.
    """
    use = chop if flip else ("CHOP_B" if chop == "CHOP_A" else "CHOP_A")
    k = int(np.argmin([abs(inum - s["avginum"]) for s in skies]))
    sky = np.nan_to_num(skies[k][use], nan=0.0)
    return frame - sky
