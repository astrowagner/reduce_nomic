"""Driver: reduce one night and aperture from raw frames to the cleaned, binned cubes.

The stages and their order follow the IDL predecessor, reduce_nomic.pro: combine (per round:
science frames, then each sky set), chop subtraction (sky rounds first), then, for the science
round, per chunk of ``frame_limit`` frames, split, align, center, pre-clean and pre-binning,
and finally replacement and clean over all binned frames. Blocks of frames, segments and chunks
run in parallel worker processes; large intermediate arrays live in memory-mapped files in a
work folder, never in the archive's IDL folders.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits

from . import binning, chopsub, combine, geometry, idl, names, select
from .config import Settings
from .frames import RawFrame, list_raw, select_science_frames


def log(msg: str) -> None:
    """Print a time-stamped progress line.

    Parameters
    ----------
    msg : str
        Message.

    Returns
    -------
    None
        The line goes to stdout, flushed.
    """
    print(time.strftime("%H:%M:%S"), msg, flush=True)


#: File lists of the rounds, by round label, shared with forked workers.
_FILES: Dict[str, List[RawFrame]] = {}


CHUNK_TASK_FIXED = 0.5 * 2 ** 30    # memory of a chunk task beyond its frames, bytes (tools/bench_chunk.py)
MEM_FRACTION = 0.75                 # share of the available memory the workers of a stage may use


def _mem_workers(workers: int, per_task: float, label: str) -> int:
    """Cap a stage's worker count so that its tasks fit in the memory available.

    Parameters
    ----------
    workers : int
        Requested worker processes.
    per_task : float
        Estimated peak memory of one task, in bytes.
    label : str
        Stage name, for the log line.

    Returns
    -------
    n : int
        ``workers``, or the largest count whose tasks fit in ``MEM_FRACTION`` of
        ``MemAvailable`` (``/proc/meminfo``, read now), and at least 1; ``workers`` when
        ``/proc/meminfo`` cannot be read. A reduced count is logged.
    """
    try:
        with open("/proc/meminfo") as fh:
            avail = next(int(line.split()[1]) * 1024 for line in fh if line.startswith("MemAvailable:"))
    except (OSError, StopIteration, ValueError, IndexError):
        return workers
    n = max(1, min(workers, int(MEM_FRACTION * avail // max(per_task, 1.0))))
    if n < workers:
        log(f"{label}: {n} worker{'s' if n > 1 else ''} instead of {workers}, so that {per_task / 2 ** 30:.1f} GB per task fits in "
            f"{100 * MEM_FRACTION:.0f} % of the {avail / 2 ** 30:.0f} GB of memory available")
    return n


def _run(fn, tasks: Sequence, workers: int, label: str = "", on_result=None, disp=None,
         sizes: Optional[Sequence[int]] = None, unit: str = "tasks") -> List:
    """Map ``fn`` over ``tasks`` in forked worker processes, keeping the order, with progress.

    Parameters
    ----------
    fn : callable
        Function of one task.
    tasks : sequence
        Task arguments.
    workers : int
        Number of worker processes; 1 runs in this process.
    label : str, default ""
        Stage name for the progress lines.
    on_result : callable, optional
        Called as ``on_result(result)`` in this process as each result arrives, in order.
    disp : Display, optional
        Live display whose progress strip is updated; while the next result is awaited its
        window events are processed every 0.2 s (``Display.pump``).
    sizes : sequence of int, optional
        Items (frames or segments) each task covers, so that progress counts items, not
        tasks; default one item per task.
    unit : str, default "tasks"
        Name of the items, for the progress lines.

    Returns
    -------
    results : list
        ``fn(task)`` for each task, in order. The pool is created here, after ``_FILES`` is
        filled, so the workers inherit it without pickling. A progress line (items done of
        all, percent, elapsed and estimated remaining time) is printed at most every 10 s and
        at the end.
    """
    n = len(tasks)
    size = [1] * n if sizes is None else [int(v) for v in sizes]
    cum = np.concatenate([[0], np.cumsum(size)]).astype(int)
    total = int(cum[-1])
    t0 = time.time()
    last = [0.0]
    out: List = []

    def progress(done_tasks: int) -> None:
        """Report progress after ``done_tasks`` results.

        Parameters
        ----------
        done_tasks : int
            Results received so far.

        Returns
        -------
        None
            The display's progress strip is updated; a log line is printed at most every 10 s
            and at the end. Both count items (``unit``), not tasks.
        """
        done = int(cum[done_tasks])
        el = time.time() - t0
        eta = el / done * (total - done) if done else None
        if disp is not None:
            disp.show_progress(label, done, total, el, eta, unit)
        if label and (done_tasks == n or time.time() - last[0] >= 10.0):
            last[0] = time.time()
            left = f", about {eta / 60:.1f} min left" if eta is not None and done < total else ""
            log(f"{label}: {done:,}/{total:,} {unit} ({100.0 * done / max(total, 1):.0f} %), "
                f"{el / 60:.1f} min elapsed{left}")

    def handle(r) -> None:
        """Keep one result, pass it to ``on_result`` and report progress.

        Parameters
        ----------
        r : object
            The next result, in task order.

        Returns
        -------
        None
            ``r`` is appended to the results.
        """
        out.append(r)
        if on_result is not None:
            on_result(r)
        progress(len(out))

    if workers <= 1 or n <= 1:
        for t in tasks:
            handle(fn(t))
        return out
    with warnings.catch_warnings():
        # The only other thread here is the feeder thread of the display queue
        # (display.Display); the workers never use its locks, so forking past it is safe.
        warnings.filterwarnings("ignore", message=r"This process .* is multi-threaded",
                                category=DeprecationWarning)
        pool = mp.get_context("fork").Pool(min(workers, n))
    with pool:
        it = pool.imap(fn, tasks, chunksize=1)
        for _ in range(n):
            while True:
                try:
                    r = it.next(timeout=0.2) if disp is not None else it.next()
                    break
                except mp.TimeoutError:
                    disp.pump()
            handle(r)
    return out


def _states_task(args) -> List[combine.FrameState]:
    """Run ``combine.frame_states`` for one block of positions (worker).

    Parameters
    ----------
    args : tuple
        ``(tag, lo, hi, init_bg, badmap, st)``, where tag names the round's file list in
        ``_FILES``.

    Returns
    -------
    states : list of FrameState
        The block's frame states.
    thumb : tuple or None
        For the display, from the block's last frame under the background limit: its crude star
        image (the image its position was taken from) and its rolling-background-subtracted
        frame, both shape (512, 512) float32, its star position in pixels, and its frame number;
        None without ``display`` or when no frame is under the limit.
    """
    tag, lo, hi, init_bg, badmap, st = args
    images = {} if st.get("display") else None
    states = combine.frame_states(_FILES[tag], lo, hi, init_bg, badmap, st, images=images)
    thumb = None
    if images and "last" in images:
        pos, img, bsub = images["last"]
        s = states[pos - lo]
        thumb = (img.astype(np.float32), bsub.astype(np.float32), (float(s.star_x), float(s.star_y)), f"{s.num}")
    return states, thumb


def _segments_task(args) -> int:
    """Average a block of segments into the segment memmap (worker).

    Parameters
    ----------
    args : tuple
        ``(rows, segs, tag, st, path, n)``: memmap rows to fill, their segments, the round's
        label in ``_FILES``, settings, memmap path and total number of rows.

    Returns
    -------
    count : int
        Number of rows written.
    """
    rows, segs, tag, st, path, n = args
    mm = np.memmap(path, dtype=np.float32, mode="r+", shape=(n, 512, 512))
    for r, seg in zip(rows, segs):
        mm[r] = combine.segment_frame(seg, _FILES[tag], st)
    mm.flush()
    return len(rows)


def _chopsub_task(args) -> List[Tuple[int, float, float]]:
    """Chop-subtract and correct a block of segment frames (worker).

    Parameters
    ----------
    args : tuple
        ``(lo, hi, seg_path, out_path, n, badmap, st, sky_info)``: rows lo..hi-1 of the ordered
        segment memmap, output memmap, row count, bad-pixel map, settings, and for sky
        subtraction ``(chops, inums, skies)`` or None.

    Returns
    -------
    noise : list of tuple
        ``(row, noise_before_sky, noise_after_sky)`` for sky-subtracted rows (IDL ``checksky``
        measure), empty otherwise.
    thumb : tuple or None
        For the display: the block's last corrected frame, its 35-px high-pass and its row
        label; None without ``display``.
    """
    lo, hi, seg_path, out_path, n, badmap, st, sky_info = args
    seg = np.memmap(seg_path, dtype=np.float32, mode="r", shape=(n, 512, 512))
    out = np.memmap(out_path, dtype=np.float32, mode="r+", shape=(n, 512, 512))
    cos = chopsub.Cosmetics(badmap, st)
    noise = []
    for i in range(lo, hi):
        prev = np.asarray(seg[i - 1], np.float64) if i > 0 else None
        nxt = np.asarray(seg[i + 1], np.float64) if i < n - 1 else None
        f = cos.apply(chopsub.chop_subtract(prev, np.asarray(seg[i], np.float64), nxt))
        if sky_info is not None:
            chops, inums, skies = sky_info
            before = f[0:256, 384:512] - idl.smooth(f, 35, nan=True)[0:256, 384:512]
            f = chopsub.subtract_sky(f, chops[i], inums[i], skies, bool(st["flip_sky_chop"]))
            after = f[0:256, 384:512] - idl.smooth(f, 35, nan=True)[0:256, 384:512]
            noise.append((i, float(np.nanmean(np.abs(before))), float(np.nanmean(np.abs(after)))))
        out[i] = f
    out.flush()
    thumb = None
    if st.get("display") and hi > lo:
        thumb = (f.astype(np.float32), (f - idl.smooth(f, 35, nan=True)).astype(np.float32), f"segment {hi - 1}")
    return noise, thumb


def _chunk_task(args) -> Dict:
    """Split, align, center, pre-clean and pre-bin one chop position of one chunk (worker).

    Parameters
    ----------
    args : tuple
        ``(c, lo, hi, cn, pos_raw, pos_cut, path, n, chops, angles, coadds, exptimes, st)``:
        chunk index, its rows, chop position (1 or 2), positions found and used for the cut,
        corrected-frame memmap, row count, and per-row chop, angle, COADD and EFF_EXPT.

    Returns
    -------
    res : dict
        ``binned``, ``angles``, ``exptimes`` (binned), ``coadds`` (kept frames, unbinned, as IDL
        saves them), and diagnostics: frame counts, bad frames, alignment shifts, the center
        used and both estimators' centers (``center_idl``, ``center_gauss``, ``center_used``; see
        ``geometry.center_cube``), ``dropped`` (None, or why the chunk's frames were dropped: both
        center estimators failed), the pre-clean's ``corrs``, ``noises`` and ``peaks``, and for
        the display ``thumb``, the median of the centered frames' central 200x200 pixels.

    Notes
    -----
    Only the frames of this chop position are read from the memmap. The cut cube (float32) is
    aligned, centered and peak-scaled in place, and the kept frames are binned by index
    (``binning.prebin``), so a task's peak memory is about m x ((2 radius)^2 + 512^2) x 4 bytes
    plus 0.5 GB, where m is the number of frames of this chop in the chunk (``CHUNK_TASK_FIXED``).
    """
    c, lo, hi, cn, pos_raw, pos_cut, path, n, chops, angles, coadds, exptimes, st = args
    which = "CHOP_A" if cn == 1 else "CHOP_B"
    ch = np.asarray(chops[lo:hi])
    sel_rows = np.nonzero(ch == which)[0]
    mm = np.memmap(path, dtype=np.float32, mode="r", shape=(n, 512, 512))
    frames = np.asarray(mm[lo + sel_rows])      # only this chop's frames are read
    del mm
    cube = geometry.cut_chop(frames, np.full(sel_rows.size, which), which, pos_raw, pos_cut, st)
    del frames
    ang = np.asarray(angles[lo:hi])[sel_rows]
    coa = np.asarray(coadds[lo:hi])[sel_rows]
    exp = np.asarray(exptimes[lo:hi])[sel_rows]
    aligned, shifts = geometry.align_cube(cube, st)
    rng = np.random.default_rng([int(st["seed"]), c, cn])
    diag: Dict = {}
    try:
        centered, center = geometry.center_cube(aligned, st, rng, diag=diag)
    except geometry.CenterFailed as err:        # no star in this chunk's median: drop its frames
        return {"chunk": c, "cn": cn, "n_in": int(aligned.shape[0]), "bad": list(range(aligned.shape[0])),
                "shifts": shifts.tolist(), "center": None, "center_idl": None, "center_gauss": None,
                "center_used": None, "dropped": str(err), "corrs": [], "noises": [], "peaks": [],
                "thumb": None, "binned": None, "angles": [], "exptimes": [], "coadds": []}
    sel = select.measure_and_select(centered, ang, st, cn, pre=True)
    res = {"chunk": c, "cn": cn, "n_in": int(centered.shape[0]), "bad": sel.bad.tolist(),
           "shifts": shifts.tolist(), "center": list(center), "center_idl": diag.get("center_idl"),
           "center_gauss": diag.get("center_gauss"), "center_used": diag.get("center_used"),
           "dropped": None, "corrs": sel.corrs.tolist(),
           "noises": sel.noises.tolist(), "peaks": sel.peaks.tolist(), "thumb": None}
    if st.get("display") and centered.shape[0]:
        r = int(st["radius"])
        step = max(1, centered.shape[0] // 50)
        with warnings.catch_warnings():     # NaN channel-edge rows: all-NaN columns are expected
            warnings.simplefilter("ignore", RuntimeWarning)
            res["thumb"] = np.nanmedian(centered[::step, r - 100:r + 100, r - 100:r + 100], axis=0).astype(np.float32)
    if sel.keep.size == 0:
        res.update(binned=None, angles=[], exptimes=[], coadds=[])
        return res
    if st["pre_peak_scale"]:                     # select.peak_scaled, in place to save memory
        centered /= np.nanmax(centered, axis=(1, 2))[:, None, None]
    b, ab, eb = binning.prebin(centered, ang[sel.keep], exp[sel.keep], st, index=sel.keep)
    res.update(binned=b.astype(np.float32), angles=ab.tolist(), exptimes=eb.tolist(), coadds=coa[sel.keep].tolist())
    return res


def _start_median(mm: np.ndarray, start: int, hi: int, chops: Sequence[str], label: str, k: int) -> np.ndarray:
    """Take the median of a chunk's first k frames of one chop, for the split stage's positions.

    Parameters
    ----------
    mm : ndarray, shape (n, 512, 512)
        Corrected chop-combined frames (the chop subtraction's memmap), where n is their number.
    start : int
        First row searched.
    hi : int
        Row after the chunk's last.
    chops : sequence of str
        Chop of every row of ``mm``.
    label : str
        Chop of the frames taken (``CHOP_A`` or ``CHOP_B``).
    k : int
        Number of frames (``split_median_frames``); with 1 the first frame of that chop from
        ``start`` on is returned itself, as the IDL predecessor uses single frames.

    Returns
    -------
    img : ndarray, shape (512, 512)
        Float64 NaN-ignoring median of up to k frames of chop ``label`` in rows ``start`` to
        ``hi - 1`` (NaN where all are NaN); the frame at ``start`` when none has that chop.
    """
    rows = [i for i in range(start, hi) if chops[i] == label][:k]
    if len(rows) <= 1:
        return np.asarray(mm[rows[0] if rows else start], dtype=np.float64)
    with warnings.catch_warnings():         # pixels masked in every frame
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(np.asarray(mm[rows], dtype=np.float64), axis=0)


def _blocks(n: int, size: int) -> List[Tuple[int, int]]:
    """Split ``range(n)`` into consecutive blocks.

    Parameters
    ----------
    n : int
        Number of items.
    size : int
        Block size.

    Returns
    -------
    blocks : list of tuple of int
        ``(lo, hi)`` pairs covering 0..n-1.
    """
    return [(lo, min(lo + size, n)) for lo in range(0, n, size)]


def _chop_feed(disp, st: Dict, sky_round: bool, tag: str):
    """Start a round on the display and return the callback that feeds it the chop states.

    The callback labels each frame under the background limit with ``combine.chop_label``, in
    file order and with the settings ``combine.build_segments`` uses (a sky round forces
    ``set_chop_via_bg``), so the display shows the chop states the segments get.

    Parameters
    ----------
    disp : Display
        Live display.
    st : dict
        Settings (``chop_direction``, ``set_chop_via_bg``).
    sky_round : bool
        True for rounds 1..n_sky.
    tag : str
        Round label.

    Returns
    -------
    show : callable
        ``show(result)`` for each ``_states_task`` result, in order: adds the block's frames
        (frame number, star position, chop state, the value that decided it: the star's x or y
        in pixels, or with ``set_chop_via_bg`` the stripe mean minus the previous kept frame's,
        and the chop-ID peak S/N) and shows the block's last chop-ID image with its chop state:
        the crude star image, or with ``set_chop_via_bg`` the rolling-background-subtracted
        frame.
    """
    eff = dict(st)
    if sky_round:
        eff["set_chop_via_bg"] = 1
    via_bg = bool(eff["set_chop_via_bg"])
    lr = st["chop_direction"] == "LEFT-RIGHT"
    if via_bg:
        disp.start_round(tag, "Stripe Mean Change (ADU)", 0.0, st["chop_direction"], via_bg=True,
                         desc="rolling background subtracted; stripe rows 126-130 decide")
    else:
        desc = "rolling background, " + ("destriped, " if st.get("chopid_destripe") else "") + "masked, 7-px boxcar"
        disp.start_round(tag, f"{'X' if lr else 'Y'} of Detected Star (Pixels)", 256.0, st["chop_direction"],
                         desc=desc)
    prevmean = [0.0]
    first: List[Optional[int]] = [None]

    def show(r) -> None:
        """Feed one block's chop states and raw-frame thumbnail to the display.

        Parameters
        ----------
        r : tuple
            ``(states, thumb)`` from ``_states_task``.

        Returns
        -------
        None
            ``prevmean`` and ``first`` carry the labeling state to the next block.
        """
        sts, thumb = r
        if first[0] is None and sts:
            first[0] = sts[0].pos
        nums, xs, ys, chops, vals, snrs = [], [], [], [], [], []
        for q in sts:
            if q.star_x < 0:
                continue
            if via_bg:
                val = q.stripe - (0.0 if q.pos == first[0] else prevmean[0])
            else:
                val = float(q.star_x if lr else q.star_y)
            chops.append(combine.chop_label(q, eff, prevmean, first[0]))
            nums.append(q.num)
            xs.append(q.star_x)
            ys.append(q.star_y)
            vals.append(val)
            snrs.append(q.snr)
        disp.add_states(nums, xs, ys, chops, vals, skipped=len(sts) - len(nums), snrs=snrs)
        if thumb is not None and chops:
            img, bsub, star, label = thumb
            disp.show_chop_image(bsub if via_bg else img, star, chops[-1], label, via_bg)

    return show


def combine_round(files: Sequence[RawFrame], st: Dict, sky_round: bool, master: np.ndarray, work: str,
                  tag: str, workers: int) -> Dict:
    """Run the combine stage on one round.

    Parameters
    ----------
    files : sequence of RawFrame
        The round's frames, sorted.
    st : dict
        Settings.
    sky_round : bool
        True for a sky round.
    master : ndarray, shape (512, 512)
        Master bad-pixel map.
    work : str
        Work folder.
    tag : str
        Round label for file names, e.g. ``round0``.
    workers : int
        Worker processes.

    Returns
    -------
    rnd : dict
        ``badmap``, ``segments`` (in chop-subtraction order), ``seg_path`` (memmap of their
        frames, in that order), ``n`` (segment count), ``states``, ``nostar`` and
        ``nostar_cut``: the frames skipped for having no star in the searched region
        (``combine.reject_no_star``, science round only) and the S/N cut used, and ``pa``: the
        frames whose LBT_PARA is unusable (999 or absent) and what was done with them
        (``pa_invalid``, ``combine.repair_angles``; science round only, not with ``fix_pa``):
        ``bad`` (count), ``action``, and when any, ``first`` and ``last`` frame numbers and the
        pointing ``ra_hours`` and ``dec_deg`` used to recompute them.
    """
    log(f"{tag}: {len(files)} frames, initial background and bad-pixel map")
    init = combine.initial_background(files)
    badmap = combine.make_badmap(master, init["badtest"], st["badsigthresh"], int(st["badsz"]), int(st["badbuff"]), bool(st["fresh_badmap"]))
    _FILES[tag] = list(files)
    disp = st.get("_disp")
    st = {k: v for k, v in st.items() if k != "_disp"}
    # smaller blocks with the display, so the chop identification updates often; each block
    # rereads up to 2 nbg neighbor frames for its rolling background, so without it larger ones
    size = 200 if disp is not None else 500
    tasks = [(tag, lo, hi, init["bg"], badmap, st) for lo, hi in _blocks(len(files), size)]
    if disp is not None:
        show = _chop_feed(disp, st, sky_round, tag)
    else:
        show = None
    states = [s for part, _ in _run(_states_task, tasks, workers, f"{tag} chop states", show, disp,
                                    sizes=[hi - lo for _, lo, hi, *_ in tasks], unit="frames") for s in part]
    frac = float(st.get("chopid_nostar_frac") or 0.0)
    nrej, cut = 0, float("nan")
    if not sky_round and not st["set_chop_via_bg"] and frac > 0:
        snr = np.array([q.snr for q in states if q.star_x >= 0], dtype=float)
        cut = frac * float(np.nanmedian(snr)) if np.isfinite(snr).any() else float("nan")
        nrej = combine.reject_no_star(states, frac)
        log(f"{tag}: {nrej:,} frames without a star in the searched region skipped "
            f"(chop-ID peak S/N below {frac:g} x the median, {cut:.0f})")
        if disp is not None:
            disp.mark_no_star([q.num for q in states if q.nostar], cut)
    angles = None
    pa = {"bad": 0, "action": "none"}
    if st["fix_pa"]:
        angles = combine.parallactic_angles(states, float(st["ra"]), float(st["dec"]), float(st["lat"]), float(st["lng"]))
    elif not sky_round and any(not combine.valid_angle(q.angle) for q in states):
        mode = st.get("pa_invalid") or "recompute"
        good = [i for i, q in enumerate(states) if combine.valid_angle(q.angle)]
        pick = [good[k] for k in np.linspace(0, len(good) - 1, min(15, len(good))).astype(int)] if good else []
        coords = combine.header_coords([files[states[i].pos].path for i in pick]) if mode == "recompute" else (np.nan, np.nan)
        angles, bad, action = combine.repair_angles(states, mode, coords, float(st["lat"]), float(st["lng"]))
        pa = {"bad": len(bad), "action": action, "first": states[bad[0]].num, "last": states[bad[-1]].num,
              "ra_hours": None if not np.isfinite(coords[0]) else round(coords[0], 5),
              "dec_deg": None if not np.isfinite(coords[1]) else round(coords[1], 5)}
        log(f"{tag}: {len(bad):,} frames with an unusable LBT_PARA (999 or none), frames {pa['first']}-{pa['last']}: {action}"
            + (f" from the pointing RA {coords[0]:.5f} h, Dec {coords[1]:+.5f} deg" if action == "recomputed" else ""))
    segs = combine.build_segments(states, st, sky_round, angles)
    order = combine.segment_order(segs)
    segs = [segs[i] for i in order]
    n = len(segs)
    log(f"{tag}: {sum(s.star_x >= 0 for s in states):,} frames used for chop states, {n:,} segments")
    if disp is not None:
        disp.show_segments([len(s.positions) for s in segs])
    path = os.path.join(work, f"{tag}_segments.f32")
    np.memmap(path, dtype=np.float32, mode="w+", shape=(max(n, 1), 512, 512)).flush()
    blocks = _blocks(n, 200)
    _run(_segments_task, [(list(range(lo, hi)), segs[lo:hi], tag, st, path, n) for lo, hi in blocks], workers,
         f"{tag} segment coadds", None, disp, sizes=[hi - lo for lo, hi in blocks], unit="segments")
    return {"badmap": badmap, "segments": segs, "seg_path": path, "n": n, "states": states,
            "nostar": nrej, "nostar_cut": cut, "pa": pa}


def chopsub_round(rnd: Dict, badmap: np.ndarray, st: Dict, work: str, tag: str, workers: int,
                  skies: Optional[List[Dict]] = None) -> str:
    """Run the chop-subtraction stage on one round.

    Parameters
    ----------
    rnd : dict
        Output of ``combine_round``.
    badmap : ndarray, shape (512, 512)
        Bad-pixel map for the corrections: the last combine round's, as in IDL.
    st : dict
        Settings.
    work : str
        Work folder.
    tag : str
        Round label.
    workers : int
        Worker processes.
    skies : list of dict, optional
        Mean skies for sky subtraction (science round with ``skysub``).

    Returns
    -------
    path : str
        Memmap of the corrected frames, (n, 512, 512) float32, in segment order.
    """
    n = rnd["n"]
    path = os.path.join(work, f"{tag}_chopsub.f32")
    np.memmap(path, dtype=np.float32, mode="w+", shape=(max(n, 1), 512, 512)).flush()
    sky_info = None
    if skies:
        sky_info = ([s.chop for s in rnd["segments"]], [float(s.inum) for s in rnd["segments"]], skies)
    disp = st.get("_disp")
    st = {k: v for k, v in st.items() if k != "_disp"}
    tasks = [(lo, hi, rnd["seg_path"], path, n, badmap, st, sky_info) for lo, hi in _blocks(n, 250)]

    def show(r):
        """Show one block's last chop-subtracted frame on the display.

        Parameters
        ----------
        r : tuple
            ``(noise, thumb)`` from ``_chopsub_task``.

        Returns
        -------
        None
            The thumbnail, when there is one, goes to ``Display.show_chopsub``.
        """
        if disp is not None and r[1] is not None:
            disp.show_chopsub(*r[1])

    noise = [x for part, _ in _run(_chopsub_task, tasks, workers, f"{tag} chop subtraction", show, disp,
                                   sizes=[t[1] - t[0] for t in tasks], unit="segments") for x in part]
    worse = [i for i, b, a in noise if b > 0 and (b - a) / b < -0.1]
    if st.get("checksky") and worse:
        log(f"{tag}: sky subtraction raised the noise by more than 10 % on {len(worse)} frames "
            "(IDL would pause here; consider flip_sky_chop)")
    log(f"{tag}: chop subtraction done")
    return path


def write_fits(path: str, data, header: Optional[Dict] = None) -> None:
    """Write an array to a FITS file, refusing to overwrite.

    Parameters
    ----------
    path : str
        Output path; must not exist.
    data : ndarray
        Array, numpy order (frames, y, x).
    header : dict, optional
        Header cards to add.

    Returns
    -------
    None
        The file is written.
    """
    hdu = fits.PrimaryHDU(np.asarray(data, dtype=np.float32))
    for k, v in (header or {}).items():
        hdu.header[k] = v
    hdu.writeto(path, overwrite=False)


def write_frame_tags(stem: str, angles, exptimes, corrs, noises, coronoise, peaks, idl_keep) -> None:
    """Write the frame tags of the binned frames, the metrics KLIP-TPE selects frames with.

    Parameters
    ----------
    stem : str
        Path without extension; ``.fits`` (a binary table) and ``.csv`` are written, never over
        existing files.
    angles, exptimes : array_like, shape (m,)
        Parallactic angle in degrees and summed exposure time in seconds of each of the m binned
        frames.
    corrs : array_like, shape (m,)
        Correlation of each frame with the mean of the frames within 50 of it, in the central
        128 x 128 pixels (the clean stage's measure, 0 to 1).
    noises : array_like, shape (m,)
        Standard deviation of each frame outside the central 100 columns, in the cube's units
        (counts per second before peak scaling).
    coronoise : array_like, shape (m,)
        Standard deviation of each frame from 0.3 to 0.65 arcsec of the star, in the same units
        (``select.ring_noise``).
    peaks : array_like, shape (m,)
        Maximum pixel of each frame, in the same units.
    idl_keep : array_like of bool, shape (m,)
        True for the frames the IDL predecessor's clean stage keeps with the night's thresholds.

    Returns
    -------
    None
        Both files have the columns ``frame``, ``angle_deg``, ``exptime_s``, ``corrs``,
        ``noises`` and ``coronoise`` (the noises over their median, the names and units of
        KLIP-TPE's frame tags, which its ``corr_thresh``, ``noise_max`` and ``coronoise_max``
        cut), ``noise_raw``, ``coronoise_raw``, ``peak_raw`` (the measures before normalizing)
        and ``idl_clean_keep``.
    """
    from astropy.table import Table
    noises = np.asarray(noises, dtype=float)
    coronoise = np.asarray(coronoise, dtype=float)

    def rel(v):
        """Return v over its median, or v unchanged when the median is not positive."""
        m = np.nanmedian(v) if v.size else np.nan
        return v / m if np.isfinite(m) and m > 0 else v

    t = Table({"frame": np.arange(len(noises)), "angle_deg": np.asarray(angles, dtype=float),
               "exptime_s": np.asarray(exptimes, dtype=float), "corrs": np.asarray(corrs, dtype=float),
               "noises": rel(noises), "coronoise": rel(coronoise), "noise_raw": noises,
               "coronoise_raw": coronoise, "peak_raw": np.asarray(peaks, dtype=float),
               "idl_clean_keep": np.asarray(idl_keep, dtype=bool)})
    t.meta["COMMENT"] = ["Frame tags of the binned frames; KLIP-TPE cuts corrs, noises and coronoise."]
    t.write(stem + ".fits", format="fits", overwrite=False)
    with open(stem + ".csv", "x") as fh:
        fh.write(",".join(t.colnames) + "\n")
        for row in t:
            fh.write(",".join(str(int(v)) if isinstance(v, (bool, np.bool_)) else
                              (f"{v:.6g}" if isinstance(v, (float, np.floating)) else str(v)) for v in row) + "\n")


def _median_or_nan(cube: np.ndarray, even: bool = False) -> np.ndarray:
    """Per-pixel IDL median of a cube, or an all-NaN frame when the cube has no frames.

    Parameters
    ----------
    cube : ndarray, shape (m, ny, nx)
        Frames; m may be 0.
    even : bool, default False
        Passed to ``idl.median``.

    Returns
    -------
    med : ndarray, shape (ny, nx)
        The median, or NaN everywhere for an empty cube (so a night whose clean keeps no frame
        still writes every product).
    """
    if cube.shape[0] == 0:
        return np.full(cube.shape[1:], np.nan, dtype=np.float32)
    return idl.median(cube, axis=0, even=even) if even else idl.median(cube, axis=0)


def planned_outputs(pre: str) -> List[str]:
    """List the product file names one run writes.

    Parameters
    ----------
    pre : str
        Product prefix ``<night>_<side>``.

    Returns
    -------
    names : list of str
        The badmap, the summary and, for chops 1 and 2, the cubes, PSFs, angles, coadds and frame
        tags listed in ``reduce``; a run refuses to start when any of them exists.
    """
    return [f"{pre}_badmap.fits", f"{pre}_summary.json"] + [
        f"{pre}_{name}" for cn in (1, 2) for name in (
            f"cube{cn}_align_cen_clean_bin.fits", f"cube{cn}_align_cen_clean_median_PSF.fits",
            f"cube{cn}_align_cen_median_PSF.fits", f"angles{cn}_clean_bin.fits",
            f"angles{cn}_clean_bin.csv", f"coadds{cn}_clean_bin.csv",
            f"cube{cn}_align_cen_bin.fits", f"angles{cn}_bin.fits", f"frametags{cn}_bin.fits",
            f"frametags{cn}_bin.csv", f"coadds{cn}_bin.csv")]


def sky_folders(st: Dict) -> List[str]:
    """List a night's sky folders in round order, as the IDL predecessor names them.

    Parameters
    ----------
    st : dict
        Settings: ``n_sky``, ``archive`` and ``sky_dirs`` (explicit folders relative to the
        night folder; an old folder name after ``../`` is renamed, ``names.current_rel``).

    Returns
    -------
    folders : list of str
        ``n_sky`` folders relative to the night folder: ``sky_dirs`` when given, otherwise
        ``skys1``, ``skys2``, ``skys3``, ``skys3``, ... (round k reads ``skys<min(k, 3)>``, as IDL).
    """
    n = int(st["n_sky"])
    dirs = st.get("sky_dirs") or [f"skys{min(k, 3)}" for k in range(1, n + 1)]
    return [names.current_rel(st["archive"], d) for d in dirs][:n]


def reduce(settings: Settings, side: str, workers: int = 1, out_dir: Optional[str] = None,
           work_dir: Optional[str] = None) -> Dict:
    """Reduce one night and aperture.

    Parameters
    ----------
    settings : Settings
        The night's settings (``config.load_settings``).
    side : str
        ``left`` or ``right``.
    workers : int, default 1
        Worker processes.
    out_dir : str, optional
        Output folder; default ``<archive>/<night>/<out_subdir>``. Existing outputs are never
        overwritten.
    work_dir : str, optional
        Folder for the memory-mapped intermediates; default ``<out_dir>/work_<side>``.

    Returns
    -------
    summary : dict
        Counts and file names; also written as ``<night>_<side>_summary.json``.

    Notes
    -----
    A run holds an exclusive lock on ``<work_dir>/lock`` (released when it ends, however it ends)
    and refuses to start when another run holds it or when any of its outputs already exist.
    Outputs, in ``out_dir``, for n = 1 (chop A) and 2 (chop B), the frames 2 radius x 2 radius
    with the star at (radius - 0.5, radius - 0.5):

    * for KLIP-TPE, which makes the final frame selection itself: every binned frame after the
      pre-clean, ``<night>_<side>_cube<n>_align_cen_bin.fits`` (replaced and peak-scaled as the
      clean stage would), ``<night>_<side>_angles<n>_bin.fits`` (parallactic angles in degrees),
      ``<night>_<side>_coadds<n>_bin.csv``, and the frame tags ``<night>_<side>_frametags<n>_bin``
      ``.fits`` and ``.csv`` (``write_frame_tags``);
    * with the IDL predecessor's names, after its clean stage, for comparison:
      ``<night>_<side>_cube<n>_align_cen_clean_bin.fits``, ``<night>_<side>_angles<n>_clean_bin.fits``
      and ``.csv`` (angles and summed exposure times), ``<night>_<side>_coadds<n>_clean_bin.csv``,
      ``<night>_<side>_cube<n>_align_cen_clean_median_PSF.fits``;
    * ``<night>_<side>_cube<n>_align_cen_median_PSF.fits`` (before replacement and clean) and
      ``<night>_<side>_badmap.fits``.
    """
    st = dict(settings.values)
    st["side"] = side
    night = st["night"]
    archive = st["archive"]
    out_dir = out_dir or os.path.join(archive, night, st["out_subdir"])
    work = work_dir or os.path.join(out_dir, f"work_{side}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(work, exist_ok=True)
    pre = f"{night}_{side}"
    t0 = time.time()
    chopsub.check_options(st)
    planned = planned_outputs(pre)
    present = [f for f in planned if os.path.exists(os.path.join(out_dir, f))]
    if present:
        raise FileExistsError(f"{len(present)} outputs already exist in {out_dir} (e.g. {present[0]}); "
                              "move them or give another --out")
    import fcntl
    lock = open(os.path.join(work, "lock"), "a")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise RuntimeError(f"another run is using {work}; give another --work or wait for it to finish")
    disp = None
    if st.get("display"):
        from .display import Display
        disp = Display(f"{night} {side}", os.path.join(out_dir, f"{pre}_display.png"), log=log)
        log(f"display: {disp.note}")
    st["_disp"] = disp
    start = st["start_frame_left"] if side == "left" else st["start_frame_right"]
    end = st["end_frame_left"] if side == "left" else st["end_frame_right"]
    sci = select_science_frames(archive, night, start, end, frame_range=st.get("frames"),
                                old_night=st.get("night_old") or night)
    sky_dirs = sky_folders(st)
    sky_lists = [list_raw(os.path.join(archive, night, d)) for d in sky_dirs]
    master = np.asarray(fits.getdata(os.path.join(archive, st["badmap_master"])), dtype=np.float64)
    rounds = [combine_round(sci, st, False, master, work, f"{side}_round0", workers)]
    for k, files in enumerate(sky_lists, 1):
        rounds.append(combine_round(files, st, True, master, work, f"{side}_round{k}", workers))
    badmap = rounds[-1]["badmap"]
    skies = []
    for k in range(len(rounds) - 1, 0, -1):
        path = chopsub_round(rounds[k], badmap, st, work, f"{side}_round{k}", workers)
        nk = rounds[k]["n"]
        mmk = np.memmap(path, dtype=np.float32, mode="r", shape=(max(nk, 1), 512, 512))
        sk = chopsub.mean_skies(np.asarray(mmk[:nk]), [s.chop for s in rounds[k]["segments"]])
        missing = [c for c in ("CHOP_A", "CHOP_B") if c not in sk]
        if missing:
            log(f"{side}_round{k}: no {' or '.join(missing)} frames in this sky round; frames that "
                "need that sky take it from the nearest round that has it (none: no sky subtracted)")
        sk["avginum"] = float(np.mean([float(s.inum) for s in rounds[k]["segments"]])) if nk else float("nan")
        skies.insert(0, sk)
    sci_path = chopsub_round(rounds[0], badmap, st, work, f"{side}_round0", workers,
                             skies if st["skysub"] else None)
    segs = rounds[0]["segments"]
    n = rounds[0]["n"]
    chops = [s.chop for s in segs]
    angles = [s.angle for s in segs]
    coadds = [s.coadd for s in segs]
    exptimes = [s.eff_expt for s in segs]
    mm = np.memmap(sci_path, dtype=np.float32, mode="r", shape=(max(n, 1), 512, 512))
    chunks = _blocks(n, int(st["frame_limit"]))
    if chunks:
        last_lo, last_hi = chunks[-1]
        tot = np.nansum(np.asarray(mm[last_lo:last_hi]), axis=(1, 2))
        if np.any(tot == 0):
            log(f"warning: {int(np.sum(tot == 0))} all-NaN frames in the last chunk; the IDL "
                "predecessor drops them from the cube but not from the angle list")
    fw = int(st["framew"])
    k_med = max(1, int(st.get("split_median_frames", 1)))
    raw_pos = []
    for lo, hi in chunks:
        j2 = min(lo + 1, hi - 1)
        first = geometry.padded(_start_median(mm, lo, hi, chops, chops[lo], k_med), fw)
        second = geometry.padded(_start_median(mm, j2, hi, chops, chops[j2], k_med), fw)
        check = geometry.padded(_start_median(mm, lo, hi, chops, "CHOP_A", k_med), fw)
        raw_pos.append(geometry.find_positions(first, second, check, st))
    tasks, prev1, prev2 = [], None, None
    for cn in (1, 2):
        for c, (lo, hi) in enumerate(chunks):
            p1, p2 = raw_pos[c]
            q1 = geometry.edge_checked(p1, prev1, int(st["radius"]))
            q2 = geometry.edge_checked(p2, prev2, int(st["radius"]))
            prev1, prev2 = q1, q2
            tasks.append((c, lo, hi, cn, (p1, p2), q1 if cn == 1 else q2, sci_path, n, chops, angles,
                          coadds, exptimes, st))
    log(f"science: {n} frames in {len(chunks)} chunks; split, align, center, pre-clean, pre-bin")

    def show(r):
        """Show one chunk's centered median and selection on the display.

        Parameters
        ----------
        r : dict
            Result of ``_chunk_task``.

        Returns
        -------
        None
            The chunk, when it has a thumbnail, goes to ``Display.show_chunk`` with the
            pre-clean thresholds of its chop.
        """
        if disp is not None and r.get("thumb") is not None:
            cn = r["cn"]
            disp.show_chunk(r["thumb"], r, {"corr": st[f"pre_corr_thresh_{cn}"],
                                            "noise": st[f"pre_noise_thresh_{cn}"],
                                            "peak": st[f"pre_peak_thresh_{cn}"]})

    st_tasks = dict(st)
    st_tasks.pop("_disp", None)
    tasks = [t[:-1] + (st_tasks,) for t in tasks]
    sizes = [sum(1 for c in chops[t[1]:t[2]] if c == ("CHOP_A" if t[3] == 1 else "CHOP_B")) for t in tasks]
    label = f"{side} split, align, center, pre-clean"
    per_task = max(sizes, default=0) * ((2 * int(st["radius"])) ** 2 + 512 * 512) * 4 + CHUNK_TASK_FIXED
    results = _run(_chunk_task, tasks, _mem_workers(workers, per_task, label), label, show, disp,
                   sizes=sizes, unit="frames")
    cmeth = str(st.get("center_method", "gauss"))
    fell = [(r["chunk"], r["cn"]) for r in results if r.get("center_used") not in (None, cmeth)]
    gone = [(r["chunk"], r["cn"], r["n_in"]) for r in results if r.get("dropped")]
    if fell:
        log(f"center: the {cmeth} estimator failed on {len(fell)} chunk/chop pairs, which used the other "
            f"estimator's center: {fell}")
    if gone:
        log(f"center: both estimators failed on {len(gone)} chunk/chop pairs; their "
            f"{sum(g[2] for g in gone)} frames were dropped: {[(c, cn) for c, cn, _ in gone]}")
    write_fits(os.path.join(out_dir, f"{pre}_badmap.fits"), badmap)
    summary = {"night": night, "night_old": st.get("night_old") or night, "side": side,
               "settings_source": settings.source, "n_raw": len(sci),
               "sky_dirs": sky_dirs, "n_segments": n, "chunks": len(chunks),
               "nostar_frames": int(rounds[0]["nostar"]), "bad_pa": rounds[0]["pa"],
               "nostar_cut": None if not np.isfinite(rounds[0]["nostar_cut"]) else round(float(rounds[0]["nostar_cut"]), 2),
               "positions": raw_pos, "cubes": {}}
    for cn in (1, 2):
        res = [r for r in results if r["cn"] == cn]
        parts = [r for r in res if r["binned"] is not None]
        big = np.concatenate([r["binned"] for r in parts]) if parts else np.zeros((0, 2 * st["radius"], 2 * st["radius"]), np.float32)
        big[big == 0] = np.nan
        ang = np.concatenate([r["angles"] for r in parts]) if parts else np.zeros(0)
        exp = np.concatenate([r["exptimes"] for r in parts]) if parts else np.zeros(0)
        coa = np.concatenate([r["coadds"] for r in parts]) if parts else np.zeros(0)
        write_fits(os.path.join(out_dir, f"{pre}_cube{cn}_align_cen_median_PSF.fits"), _median_or_nan(big, even=True))
        n_col0 = int(sum(bool(np.isfinite(f[:, 0]).any()) for f in big))
        rep = np.stack([binning.replace_interior_nans(f) for f in big]).astype(np.float32) if big.shape[0] else big
        sel = select.measure_and_select(rep, ang, st, cn, pre=False)
        every = select.peak_scaled(rep) if st["peak_scale"] else rep
        head = {"RADIUS": int(st["radius"]), "XCEN": st["radius"] - 0.5, "YCEN": st["radius"] - 0.5,
                "PXSCALE": st["pxscale"], "CHOP": "A" if cn == 1 else "B", "SIDE": side}
        write_fits(os.path.join(out_dir, f"{pre}_cube{cn}_align_cen_bin.fits"), every,
                   dict(head, FRAMESEL="none: KLIP-TPE selects with frametags"))
        fits.PrimaryHDU(np.asarray(ang, dtype=np.float64)).writeto(
            os.path.join(out_dir, f"{pre}_angles{cn}_bin.fits"), overwrite=False)
        keep_mask = np.zeros(rep.shape[0], dtype=bool)
        keep_mask[sel.keep] = True
        write_frame_tags(os.path.join(out_dir, f"{pre}_frametags{cn}_bin"), ang, exp, sel.corrs, sel.noises,
                         select.ring_noise(rep, st) if rep.shape[0] else np.zeros(0), sel.peaks, keep_mask)
        with open(os.path.join(out_dir, f"{pre}_coadds{cn}_bin.csv"), "x") as fh:
            fh.write("coadd\n" + "".join(f"{int(x)}\n" for x in coa))
        fin = every[sel.keep]
        ang_f, exp_f = ang[sel.keep], exp[sel.keep]
        coa_f = np.delete(coa, sel.bad[sel.bad < coa.size])
        write_fits(os.path.join(out_dir, f"{pre}_cube{cn}_align_cen_clean_bin.fits"), fin,
                   dict(head, FRAMESEL="IDL clean thresholds"))
        write_fits(os.path.join(out_dir, f"{pre}_cube{cn}_align_cen_clean_median_PSF.fits"), _median_or_nan(fin))
        hdu = fits.PrimaryHDU(np.asarray(ang_f, dtype=np.float64))
        hdu.writeto(os.path.join(out_dir, f"{pre}_angles{cn}_clean_bin.fits"), overwrite=False)
        with open(os.path.join(out_dir, f"{pre}_angles{cn}_clean_bin.csv"), "x") as fh:
            fh.write("frame,angle_deg,exptime_s\n")
            for i, (a, e) in enumerate(zip(ang_f, exp_f)):
                fh.write(f"{i},{a:.6f},{e:.6f}\n")
        with open(os.path.join(out_dir, f"{pre}_coadds{cn}_clean_bin.csv"), "x") as fh:
            fh.write("coadd\n" + "".join(f"{int(x)}\n" for x in coa_f))
        summary["cubes"][cn] = {"frames_in": int(sum(r["n_in"] for r in res)),
                                "pre_clean_bad": int(sum(len(r["bad"]) for r in res)),
                                "binned": int(big.shape[0]), "clean_bad": sel.bad.tolist(),
                                "final": int(fin.shape[0]), "exptime_s": float(np.sum(exp_f)),
                                "exptime_all_s": float(np.sum(exp)),
                                "angle_range": [float(np.min(ang_f)), float(np.max(ang_f))] if ang_f.size else None,
                                "centers": [r["center"] for r in res],
                                "centers_idl": [r.get("center_idl") for r in res],
                                "centers_gauss": [r.get("center_gauss") for r in res],
                                "center_fallbacks": int(sum(1 for r in res if r.get("center_used") not in (None, st.get("center_method", "gauss")))),
                                "chunks_dropped": [r["chunk"] for r in res if r.get("dropped")],
                                "frames_dropped": int(sum(r["n_in"] for r in res if r.get("dropped"))),
                                "frames_finite_column0": n_col0}
    summary["elapsed_s"] = round(time.time() - t0, 1)
    with open(os.path.join(out_dir, f"{pre}_summary.json"), "x") as fh:
        json.dump(summary, fh, indent=1)
    log(f"done in {summary['elapsed_s']} s: " + ", ".join(
        f"cube{cn} {v['binned']} binned frames with frame tags (the IDL clean would keep {v['final']})"
        for cn, v in summary["cubes"].items()))
    if disp is not None:
        disp.close()
    return summary
