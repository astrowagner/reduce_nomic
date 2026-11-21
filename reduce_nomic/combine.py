"""Stage 1, combine: bad-pixel map, chop state of each frame, and coadds within each chop.

This is the ``do_combine`` step of the IDL predecessor (reduce_nomic.pro lines 3765-4434). Each
raw frame's chop state is found from where the star is after a crude background subtraction;
consecutive frames in the same chop state form a segment, and each segment is averaged into
one frame in counts per second. The functions work on one round (round 0: the science frames;
rounds 1..n_sky: the sky sets) and on blocks of frames, so the driver can run blocks in
parallel.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import idl
from .angles import mean_angle
from .frames import RawFrame, read_frame


def make_badmap(master: np.ndarray, badtest: np.ndarray, sigthresh: float = 2.0, sz: int = 10,
                buff: int = 5, fresh: bool = False) -> np.ndarray:
    """Build a night's bad-pixel map as the IDL predecessor's ``make_badmap`` step does.

    Parameters
    ----------
    master : ndarray, shape (512, 512)
        Master bad-pixel map (1 good, 0 bad), indexed ``[y, x]``.
    badtest : ndarray, shape (512, 512)
        Per-pixel minimum of the first raw frames (IDL ``badtestframe``).
    sigthresh : float, default 2.0
        A pixel is bad when it differs from its neighborhood mean by more than this many standard
        deviations (IDL ``badsigthresh``).
    sz : int, default 10
        The neighborhood spans x - sz .. x + sz - 1 and the same in y (IDL ``badsz``).
    buff : int, default 5
        Pixels within ``buff`` of the tested one (an 11x11 box) are left out of the neighborhood
        (IDL ``badbuff``).
    fresh : bool, default False
        Start from an all-good map instead of the master (IDL ``fresh_badmap``).

    Returns
    -------
    badmap : ndarray of float, shape (512, 512)
        1 for good and 0 for bad pixels. Rows y = 0..65 of the master count as good and columns
        x = 302..306 as bad before the test, as in IDL.
    """
    bm = np.array(master, dtype=np.float64)
    bm[0:66, :] = 1.0
    if fresh:
        bm[:] = 1.0
    bm[:, 302:307] = 0.0
    a = np.asarray(badtest, dtype=np.float64)
    ny, nx = a.shape

    def box_sums(lo_off: int, hi_off: int):
        """Sum the test frame and its square over a box around every pixel, clipped at the edges.

        Parameters
        ----------
        lo_off, hi_off : int
            Box extent relative to each pixel, the same in x and y (inclusive), in pixels.

        Returns
        -------
        n, t1, t2 : ndarray, shape (ny, nx)
            Pixel count, sum and sum of squares of each pixel's box, where (ny, nx) is the
            test frame's shape.
        """
        s1 = np.zeros((ny + 1, nx + 1))
        s2 = np.zeros((ny + 1, nx + 1))
        s1[1:, 1:] = a.cumsum(0).cumsum(1)
        s2[1:, 1:] = (a * a).cumsum(0).cumsum(1)
        yy, xx = np.mgrid[0:ny, 0:nx]
        y0 = np.clip(yy + lo_off, 0, ny - 1)
        y1 = np.clip(yy + hi_off, 0, ny - 1) + 1
        x0 = np.clip(xx + lo_off, 0, nx - 1)
        x1 = np.clip(xx + hi_off, 0, nx - 1) + 1
        n = (y1 - y0) * (x1 - x0)
        t1 = s1[y1, x1] - s1[y0, x1] - s1[y1, x0] + s1[y0, x0]
        t2 = s2[y1, x1] - s2[y0, x1] - s2[y1, x0] + s2[y0, x0]
        return n, t1, t2

    nw, sw, qw = box_sums(-sz, sz - 1)
    ne, se, qe = box_sums(-buff, buff)
    n = nw - ne
    s = sw - se
    q = qw - qe
    mean = s / n
    with np.errstate(invalid="ignore", divide="ignore"):
        std = np.sqrt(np.maximum(q - s * s / n, 0.0) / (n - 1))
        dev = np.abs(a - mean) / std
    bm[dev > sigthresh] = 0.0
    return bm


def destripe_rows(img: np.ndarray, st: Dict) -> np.ndarray:
    """Subtract from each row of each channel half its median over the columns the chop ID uses.

    Removes the horizontal stripes and channel-edge offsets that the rolling background leaves,
    so that the maximum of the chop-ID image falls on the star and not on a stripe. The halves
    are x < 256 and x >= 256 (the detector's channel boundary); the medians skip the border
    columns (x < 21 and x >= 490) and, with ``block_stripe``, the blocked columns 245-330. The
    star covers about a tenth of a half-row, so it barely moves the medians.

    Parameters
    ----------
    img : ndarray, shape (512, 512)
        Rolling-background-subtracted frame, indexed ``[y, x]``.
    st : dict
        Setting ``block_stripe``.

    Returns
    -------
    out : ndarray, shape (512, 512)
        The destriped frame (a new array).

    Notes
    -----
    Not in the IDL predecessor, which has a commented-out ``destripe`` call at this point; it is
    applied when ``chopid_destripe`` is set (the default) and only to the image whose maximum
    gives the star position, not to the stripe-mean image of ``set_chop_via_bg``.
    """
    hl = 245 if st["block_stripe"] else 256
    lr = 331 if st["block_stripe"] else 256
    out = np.array(img, dtype=np.float64, copy=True)
    out[:, :256] -= np.median(img[:, 21:hl], axis=1)[:, None]
    out[:, 256:] -= np.median(img[:, lr:490], axis=1)[:, None]
    return out


def crude_star_image(frame: np.ndarray, bg: np.ndarray, badmap: np.ndarray, st: Dict) -> np.ndarray:
    """Return the background-subtracted, masked image in which the star is looked for.

    Parameters
    ----------
    frame : ndarray, shape (512, 512)
        Raw frame.
    bg : ndarray, shape (512, 512)
        Rolling background: per-pixel minimum of ``f - smooth(f, 20)`` over neighboring frames.
    badmap : ndarray, shape (512, 512)
        Bad-pixel map (1 good, 0 bad).
    st : dict
        Settings: ``bg_sub``, ``block_right``, ``block_stripe``, ``double_sided``, ``block_extra``,
        ``side``, ``sx_top``, ``dx_bottom``, ``block_bad_channel``, ``set_chop_via_bg``,
        ``chopid_destripe`` (``destripe_rows`` before the masks) and ``chopid_median`` (a
        median filter of that width, after destriping, against hot pixels); absent means off.

    Returns
    -------
    img : ndarray, shape (512, 512)
        The image IDL takes the maximum of (``bgsubframe``). With ``set_chop_via_bg`` it is the
        unmasked background-subtracted frame with only rows 127-129 kept.
    """
    bframe = frame - idl.smooth(frame, 20)
    b = bframe - bg if st["bg_sub"] else bframe
    first = b.copy()
    b = destripe_rows(b, st) if st.get("chopid_destripe") else b.copy()
    if int(st.get("chopid_median") or 0) > 1:
        from scipy.ndimage import median_filter
        b = median_filter(b, size=int(st["chopid_median"]), mode="nearest")
    if st["block_right"]:
        b[:, 250:512] = 0.0
    b[:, 0:21] = 0.0
    b[0:21, :] = 0.0
    b[490:512, :] = 0.0
    b[:, 490:512] = 0.0
    b[120:136, :] = 0.0
    b[240:271, :] = 0.0
    b[379:392, :] = 0.0
    if st["block_stripe"]:
        b[:, 245:331] = 0.0
    if st["double_sided"]:
        if st["side"] == "left":
            b[190:512, :] = 0.0
            if st["block_extra"]:
                b[180:512, :] = 0.0
            b[st["sx_top"]:512, :] = 0.0
        else:
            b[0:306, :] = 0.0
            if st["block_extra"]:
                b[0:316, :] = 0.0
            b[0:st["dx_bottom"] + 1, :] = 0.0
    b = idl.smooth(b, 7)
    if st["block_bad_channel"] and not st["double_sided"]:
        b[0:257, 256:512] = 0.0
    b = b * badmap
    if st["set_chop_via_bg"]:
        b = first
        b[0:127, :] = 0.0
        b[130:512, :] = 0.0
    return b


@dataclass
class FrameState:
    """What the combine step learns about one raw frame.

    Parameters
    ----------
    pos : int
        Position of the frame in the round's file list.
    num : int
        Frame number.
    inum : str
        The IDL frame label (6 characters before ``.fits``).
    date : str
        DATE-OBS.
    time_end : str
        TIME-END, used by ``fix_pa``.
    angle : float
        LBT_PARA in degrees.
    dit : float
        EXPTIME in seconds.
    objname : object
        OBJNAME, or None when absent.
    median : float
        IDL median of the frame (upper middle value for an even count).
    star_x, star_y : int
        Maximum pixel of the crude star image, or -1 when the frame is over the background limit.
    stripe : float
        Mean of rows 126-130, columns 50-100 of the crude image (used with ``set_chop_via_bg``).
    snr : float
        The crude image's maximum over its robust noise (``chopid_snr``); NaN over the
        background limit or with ``set_chop_via_bg``.
    nostar : bool
        True when ``reject_no_star`` found no star (S/N far below the round's), in which case
        star_x and star_y are set to -1 and the frame is skipped like one over the background
        limit.
    """

    pos: int
    num: int
    inum: str
    date: str
    time_end: str
    angle: float
    dit: float
    objname: object
    median: float
    star_x: int = -1
    star_y: int = -1
    stripe: float = float("nan")
    snr: float = float("nan")
    nostar: bool = False


def initial_background(files: Sequence[RawFrame]) -> Dict[str, np.ndarray]:
    """Compute the starting background and the bad-pixel test frame from frames 0-10.

    Parameters
    ----------
    files : sequence of RawFrame
        The round's file list.

    Returns
    -------
    out : dict
        ``bg``: per-pixel minimum of ``f - smooth(f, 20)``; ``badtest``: per-pixel minimum of
        the raw frames (IDL ``badtestframe``). Both shape (512, 512).
    """
    raws = [read_frame(f.path)[0] for f in files[:11]]
    return {"bg": np.min([r - idl.smooth(r, 20) for r in raws], axis=0),
            "badtest": np.min(raws, axis=0)}


def chopid_snr(img: np.ndarray, j: int) -> float:
    """Return the chop-ID image's value at pixel j over its robust noise.

    Parameters
    ----------
    img : ndarray, shape (512, 512)
        Crude star image (``crude_star_image``); masked pixels are zero or within rounding of
        zero.
    j : int
        Flat index of the maximum.

    Returns
    -------
    snr : float
        ``(img[j] - median) / (1.4826 MAD)`` over the unmasked pixels; NaN when fewer than 100
        pixels are unmasked or the noise is zero.
    """
    scale = float(np.max(np.abs(img)))
    v = img[np.abs(img) > 1e-9 * scale] if scale > 0 else img[:0]
    if v.size < 100:
        return float("nan")
    med = float(np.median(v))
    sd = 1.4826 * float(np.median(np.abs(v - med)))
    return (float(img.flat[j]) - med) / sd if sd > 0 else float("nan")


def reject_no_star(states: Sequence["FrameState"], frac: float) -> int:
    """Skip the frames whose chop-ID maximum is far weaker than the round's star.

    A frame with no star in its aperture's searched region (the star outside it, or lost)
    still has a maximum, on a hot pixel, a channel edge or noise, and would get a chop state
    from it. Such frames have an S/N far below the frames with the star.

    Parameters
    ----------
    states : sequence of FrameState
        The round's frame states (changed in place).
    frac : float
        A frame is skipped when its ``snr`` is below this fraction of the median ``snr`` of the
        round's frames under the background limit; 0 or less skips none.

    Returns
    -------
    n : int
        Number of frames skipped: their ``nostar`` is set and their ``star_x``, ``star_y`` set
        to -1, so ``build_segments`` passes over them as over frames above the background
        limit.
    """
    if frac <= 0:
        return 0
    snr = np.array([q.snr for q in states if q.star_x >= 0], dtype=float)
    snr = snr[np.isfinite(snr)]
    if snr.size == 0:
        return 0
    cut = frac * float(np.median(snr))
    n = 0
    for q in states:
        if q.star_x >= 0 and np.isfinite(q.snr) and q.snr < cut:
            q.star_x = q.star_y = -1
            q.nostar = True
            n += 1
    return n


def frame_states(files: Sequence[RawFrame], lo: int, hi: int, init_bg: np.ndarray,
                 badmap: np.ndarray, st: Dict, images: Optional[Dict] = None) -> List[FrameState]:
    """Find the crude star position of frames ``lo..hi-1`` of a round.

    Parameters
    ----------
    files : sequence of RawFrame
        The round's whole file list (positions index it).
    lo, hi : int
        Range of positions to process.
    init_bg : ndarray, shape (512, 512)
        Starting background (``initial_background``).
    badmap : ndarray, shape (512, 512)
        Bad-pixel map.
    st : dict
        Settings used by ``crude_star_image`` plus ``rolling_background``, ``nbg``,
        ``background_limit``.
    images : dict, optional
        When given, receives ``images["last"] = (pos, img, bsub)`` for the last frame of the
        block under the background limit: its position, the crude star image its position was
        taken from, and its rolling-background-subtracted frame (``f - smooth(f, 20)`` minus the
        background, before any destriping or mask), both shape (512, 512). For the display.

    Returns
    -------
    states : list of FrameState
        One per position. The rolling background of position i is the minimum over positions
        i - nbg .. i + nbg (without i) when nbg < i < n - 1 - nbg, where n is the file count;
        earlier positions use ``init_bg`` and later ones the background of position n - 2 - nbg,
        as the IDL loop leaves it.
    """
    n = len(files)
    nbg = int(st["nbg"])
    cache: Dict[int, np.ndarray] = {}

    def bframe(i: int) -> np.ndarray:
        """Return frame i minus its 20-px boxcar, read once and cached.

        Parameters
        ----------
        i : int
            Position in the file list.

        Returns
        -------
        hp : ndarray, shape (512, 512)
            The high-passed frame; cached frames more than 2 nbg + 2 positions back are dropped.
        """
        if i not in cache:
            r = read_frame(files[i].path)[0]
            cache[i] = r - idl.smooth(r, 20)
            for k in [k for k in cache if k < i - 2 * nbg - 2]:
                del cache[k]
        return cache[i]

    def rolling(i: int) -> np.ndarray:
        """Return the rolling background of position i.

        Parameters
        ----------
        i : int
            Position in the file list, with nbg positions on each side.

        Returns
        -------
        bg : ndarray, shape (512, 512)
            Per-pixel minimum of ``bframe`` over the nbg positions on each side of i.
        """
        return np.min([bframe(i - k) for k in range(1, nbg + 1)] + [bframe(i + k) for k in range(1, nbg + 1)], axis=0)

    last_bg = None
    last_kept = None
    out = []
    for i in range(lo, hi):
        data, head = read_frame(files[i].path)
        med = idl.median(data)
        s = FrameState(pos=i, num=files[i].num, inum=files[i].inum, date=str(head["DATE-OBS"]),
                       time_end=str(head["TIME-END"]), angle=float(head["LBT_PARA"] if head["LBT_PARA"] is not None else np.nan),
                       dit=float(head["EXPTIME"]), objname=head["OBJNAME"], median=float(med))
        if med < st["background_limit"]:
            if st["rolling_background"] and nbg < i < n - 1 - nbg:
                bg = rolling(i)
            elif st["rolling_background"] and i >= n - 1 - nbg and n - 2 - nbg > nbg:
                if last_bg is None:
                    last_bg = rolling(n - 2 - nbg)
                bg = last_bg
            else:
                bg = init_bg
            img = crude_star_image(data, bg, badmap, st)
            j = int(np.argmax(img))
            s.star_y, s.star_x = divmod(j, img.shape[1])
            s.stripe = float(img[126:131, 50:101].mean())
            if not st["set_chop_via_bg"]:
                s.snr = chopid_snr(img, j)
            last_kept = (i, data, bg, img)
        out.append(s)
    if images is not None and last_kept is not None:
        i, data, bg, img = last_kept
        highpass = data - idl.smooth(data, 20)
        images["last"] = (i, img, highpass - bg if st["bg_sub"] else highpass)
    return out


@dataclass
class Segment:
    """Consecutive frames of one chop state, to be averaged into one chop frame.

    Parameters
    ----------
    positions : list of int
        Positions of the frames in the round's file list.
    chop : str
        ``CHOP_A`` or ``CHOP_B``.
    nod : str
        ``NOD_A`` (the IDL predecessor sets every frame to nod A).
    angle : float
        Mean parallactic angle in degrees, correct across the ±180° wrap (``angles.mean_angle``;
        NaN for a sky segment set to NaN).
    coadd : int
        Number of frames (IDL ``COADD``).
    eff_expt : float
        IDL ``EFF_EXPT``: the triggering frame's EXPTIME times the frame count.
    inum : str
        IDL ``INUM``: the label of the frame that triggered the write (the next segment's first).
    date : str
        DATE-OBS of that frame.
    nan_frame : bool
        True for a sky segment of two or more frames, which ``limit_sky_to_one`` sets to NaN.
    """

    positions: List[int]
    chop: str
    nod: str
    angle: float
    coadd: int
    eff_expt: float
    inum: str
    date: str
    nan_frame: bool = False


def chop_label(s: FrameState, st: Dict, prevmean: List[float], first_pos: int) -> str:
    """Return a frame's chop state from its crude star position.

    Parameters
    ----------
    s : FrameState
        The frame (under the background limit).
    st : dict
        Settings ``chop_direction`` and ``set_chop_via_bg``.
    prevmean : list of float
        One-element holder of the previous frame's stripe mean, updated in place.
    first_pos : int
        Position of the round's first file; the stripe mean starts from 0 there.

    Returns
    -------
    chop : str
        ``CHOP_A`` or ``CHOP_B``.
    """
    if st["chop_direction"] == "LEFT-RIGHT":
        chop = "CHOP_A" if s.star_x < 256 else "CHOP_B"
    else:
        chop = "CHOP_A" if s.star_y < 256 else "CHOP_B"
    if st["set_chop_via_bg"]:
        if s.pos == first_pos:
            prevmean[0] = 0.0
        chop = "CHOP_A" if s.stripe - prevmean[0] > 0 else "CHOP_B"
        prevmean[0] = s.stripe
    return chop


def build_segments(states: Sequence[FrameState], st: Dict, sky_round: bool,
                   angles: Optional[Sequence[float]] = None) -> List[Segment]:
    """Group frames into segments as the IDL coadd loop does.

    Parameters
    ----------
    states : sequence of FrameState
        Every frame of the round, in file-list order.
    st : dict
        Settings ``obj``, ``skip_target_check``, ``background_limit``, ``chop_direction``,
        ``set_chop_via_bg``, ``min_n``, ``skip_first``, ``skip_last``, ``limit_sky_to_one``.
    sky_round : bool
        True for rounds 1..n_sky, which force ``set_chop_via_bg`` and ``skip_target_check``.
    angles : sequence of float, optional
        Parallactic angles to use instead of LBT_PARA (``fix_pa``), one per state.

    Returns
    -------
    segments : list of Segment
        In order. A segment is closed when a kept frame arrives in another chop state; the last
        segment of the round is never closed, as in IDL.
    """
    st = dict(st)
    if sky_round:
        st["set_chop_via_bg"] = 1
        st["skip_target_check"] = 1
    obj = "".join(str(st["obj"]).split())
    segs: List[Segment] = []
    cur: List[int] = []
    cur_angles: List[float] = []
    last_chop = last_nod = None
    prevmean = [0.0]
    first_pos = states[0].pos if states else 0
    for idx, s in enumerate(states):
        if s.star_x < 0:
            continue
        chop = chop_label(s, st, prevmean, first_pos)
        nod = "NOD_A"
        if st["skip_target_check"]:
            match = True
        else:
            match = isinstance(s.objname, str) and "".join(s.objname.split()) == obj
        if not match:
            continue
        angle = s.angle if angles is None else angles[idx]
        if not cur:
            cur, cur_angles = [s.pos], [angle]
        elif chop == last_chop and nod == last_nod:
            cur.append(s.pos)
            cur_angles.append(angle)
        else:
            k = len(cur) + 1
            if k > st["min_n"]:
                nfr = k - 1
                if k > st["min_n"] and k > 2 and st["skip_first"]:
                    nfr = k - 2
                if k > st["min_n"] and k > 2 and st["skip_first"] and st["skip_last"]:
                    nfr = k - 3
                nan_frame = bool(k > 2 and sky_round and st["limit_sky_to_one"])
                ang = mean_angle(cur_angles, ignore_nan=True) if len(cur) > 1 else float(cur_angles[0])
                segs.append(Segment(positions=list(cur), chop=last_chop, nod=last_nod,
                                    angle=float("nan") if nan_frame else ang, coadd=k - 1,
                                    eff_expt=s.dit * nfr, inum=s.inum, date=s.date, nan_frame=nan_frame))
            cur, cur_angles = [s.pos], [angle]
        last_chop, last_nod = chop, nod
    return segs


def segment_frame(seg: Segment, files: Sequence[RawFrame], st: Dict) -> np.ndarray:
    """Average a segment's frames into one chop frame in counts per second.

    Parameters
    ----------
    seg : Segment
        The segment.
    files : sequence of RawFrame
        The round's file list.
    st : dict
        Settings ``initial_combination``, ``skip_first``, ``skip_last``.

    Returns
    -------
    frame : ndarray, shape (512, 512)
        Each frame divided by its own EXPTIME, then their NaN-ignoring mean (or median), or the
        single frame; all NaN for a segment set to NaN.
    """
    frames = []
    for p in seg.positions:
        data, head = read_frame(files[p].path)
        frames.append(data / float(head["EXPTIME"]))
    cube = np.asarray(frames)
    if seg.nan_frame:
        return np.full(cube.shape[1:], np.nan)
    if len(frames) == 1:
        return cube[0]
    if st["skip_first"]:
        cube[0] = np.nan
    if st["skip_last"]:
        cube[-1] = np.nan
    if st["initial_combination"] == "median":
        return idl.median(cube, axis=0, even=True)
    return np.nanmean(cube, axis=0)


def segment_order(segs: Sequence[Segment]) -> List[int]:
    """Return the order in which the chop-subtraction step reads the segments.

    Parameters
    ----------
    segs : sequence of Segment
        Segments of one round.

    Returns
    -------
    order : list of int
        Indices sorted by (DATE-OBS, INUM) of each segment's triggering frame, which is how the
        IDL chop files sort by name.
    """
    return sorted(range(len(segs)), key=lambda i: (segs[i].date, segs[i].inum))


def valid_angle(angle: float) -> bool:
    """Tell whether a header parallactic angle is usable.

    Parameters
    ----------
    angle : float
        LBT_PARA in degrees; NaN when the keyword was absent.

    Returns
    -------
    ok : bool
        False for NaN and for the TCS placeholder 999 (any value beyond 360 degrees).
    """
    return bool(np.isfinite(angle)) and abs(angle) <= 360.0


def sexagesimal(text: object) -> float:
    """Parse a sexagesimal header value such as ``11:01:50.8`` or ``+56:22:57``.

    Parameters
    ----------
    text : object
        The value; surrounding spaces and quotes are ignored.

    Returns
    -------
    value : float
        In the value's own unit (hours or degrees); NaN when it does not parse.
    """
    t = str(text).strip().strip("'").strip()
    try:
        sign = -1.0 if t.startswith("-") else 1.0
        parts = [float(x) for x in t.lstrip("+-").split(":")]
        return sign * sum(v / 60.0 ** k for k, v in enumerate(parts))
    except ValueError:
        return float("nan")


def header_coords(paths: Sequence[str]) -> Tuple[float, float]:
    """Read the target's pointing from the headers of some frames.

    Parameters
    ----------
    paths : sequence of str
        Raw frames to read (LBT_RA in hours, LBT_DEC in degrees, sexagesimal).

    Returns
    -------
    ra_hours, dec_deg : float
        Medians over the frames whose values parse; NaN when none does.
    """
    from astropy.io import fits

    ra, dec = [], []
    for p in paths:
        h = fits.getheader(p)
        r, d = sexagesimal(h.get("LBT_RA", "")), sexagesimal(h.get("LBT_DEC", ""))
        if np.isfinite(r) and np.isfinite(d):
            ra.append(r)
            dec.append(d)
    if not ra:
        return float("nan"), float("nan")
    return float(np.median(ra)), float(np.median(dec))


def repair_angles(states: Sequence["FrameState"], mode: str, coords: Tuple[float, float],
                  lat_deg: float, lng_deg: float) -> Tuple[Optional[List[float]], List[int], str]:
    """Deal with frames whose header parallactic angle is unusable (``valid_angle``).

    LBT_PARA is sometimes the placeholder 999, for example at the start or end of a sequence or
    while the telescope control system has no angle. Averaged into a segment or a bin, one such
    value spoils its angle. The IDL predecessor used the values as they were.

    Parameters
    ----------
    states : sequence of FrameState
        The round's frames (changed in place with ``drop``).
    mode : str
        ``recompute``: compute those frames' angles from DATE-OBS and TIME-END and the pointing
        ``coords`` (``parallactic_angles``), falling back to ``drop`` when ``coords`` is NaN;
        ``drop``: skip those frames, as frames without a star; ``keep``: use the values as they
        are, as IDL.
    coords : tuple of float
        ``(ra_hours, dec_deg)`` of the target.
    lat_deg, lng_deg : float
        Observatory latitude and longitude (east positive), degrees.

    Returns
    -------
    angles : list of float or None
        One angle per state for ``build_segments``, with the bad ones recomputed; None when no
        angle changed.
    bad : list of int
        Indices of the frames with an unusable angle.
    action : str
        What was done: ``none`` (no bad angle), ``recomputed``, ``dropped`` or ``kept``.
    """
    bad = [i for i, q in enumerate(states) if not valid_angle(q.angle)]
    if not bad:
        return None, bad, "none"
    if mode == "keep":
        return None, bad, "kept"
    if mode == "recompute" and np.isfinite(coords[0]) and np.isfinite(coords[1]):
        new = parallactic_angles([states[i] for i in bad], coords[0], coords[1], lat_deg, lng_deg)
        if all(np.isfinite(new)):
            angles = [q.angle for q in states]
            for i, a in zip(bad, new):
                angles[i] = a
            return angles, bad, "recomputed"
    if mode not in ("recompute", "drop"):
        raise ValueError(f"pa_invalid must be recompute, drop or keep, not {mode!r}")
    for i in bad:
        states[i].nostar = True
        states[i].star_x = states[i].star_y = -1
    return None, bad, "dropped"


def parallactic_angles(states: Sequence[FrameState], ra_hours: float, dec_deg: float,
                       lat_deg: float = 32.7014, lng_deg: float = -109.889) -> List[float]:
    """Recompute parallactic angles from the observing time (IDL ``fix_pa``).

    Parameters
    ----------
    states : sequence of FrameState
        Frames, with DATE-OBS and TIME-END (UTC).
    ra_hours : float
        Right ascension of the target in hours (IDL ``ra``).
    dec_deg : float
        Declination in degrees (IDL ``dec``).
    lat_deg : float, default 32.7014
        Observatory latitude in degrees (IDL ``lat``).
    lng_deg : float, default -109.889
        Observatory longitude in degrees, east positive (IDL ``lng``).

    Returns
    -------
    angles : list of float
        ``atan2(sin HA, cos(dec) tan(lat) - sin(dec) cos HA)`` in degrees, as IDL ``parangle``,
        with HA = local mean sidereal time minus RA. IDL's ``ct2lst`` takes the local hour and
        the time zone and so recovers the same UTC instant; the sidereal-time formulas differ
        by well under a second.
    """
    from astropy import units as u
    from astropy.time import Time
    from astropy.utils import iers

    iers.conf.auto_download = False

    out = []
    good = [i for i, s in enumerate(states) if s.date not in ("None", "") and s.time_end not in ("None", "")]
    t = Time([f"{states[i].date}T{states[i].time_end}" for i in good], scale="utc")
    lst = t.sidereal_time("mean", longitude=lng_deg * u.deg).hour
    ha = np.radians((lst - ra_hours) * 15.0)
    d, la = np.radians(dec_deg), np.radians(lat_deg)
    pa = np.degrees(np.arctan2(np.sin(ha), np.cos(d) * np.tan(la) - np.sin(d) * np.cos(ha)))
    out = [float("nan")] * len(states)
    for k, i in enumerate(good):
        out[i] = float(pa[k])
    return out
