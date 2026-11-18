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
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits

from . import binning, chopsub, combine, geometry, idl, select
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


def _run(fn, tasks: Sequence, workers: int) -> List:
    """Map ``fn`` over ``tasks`` in forked worker processes, keeping the order.

    Parameters
    ----------
    fn : callable
        Function of one task.
    tasks : sequence
        Task arguments.
    workers : int
        Number of worker processes; 1 runs in this process.

    Returns
    -------
    results : list
        ``fn(task)`` for each task, in order. The pool is created here, after ``_FILES`` is
        filled, so the workers inherit it without pickling.
    """
    if workers <= 1 or len(tasks) <= 1:
        return [fn(t) for t in tasks]
    with mp.get_context("fork").Pool(min(workers, len(tasks))) as pool:
        return pool.map(fn, tasks, chunksize=1)


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
    """
    tag, lo, hi, init_bg, badmap, st = args
    return combine.frame_states(_FILES[tag], lo, hi, init_bg, badmap, st)


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
    return noise


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
        saves them), and diagnostics: frame counts, bad frames, alignment shifts, center.
    """
    c, lo, hi, cn, pos_raw, pos_cut, path, n, chops, angles, coadds, exptimes, st = args
    which = "CHOP_A" if cn == 1 else "CHOP_B"
    mm = np.memmap(path, dtype=np.float32, mode="r", shape=(n, 512, 512))
    frames = np.asarray(mm[lo:hi])
    ch = np.asarray(chops[lo:hi])
    sel_rows = np.nonzero(ch == which)[0]
    cube = geometry.cut_chop(frames, ch, which, pos_raw, pos_cut, st)
    ang = np.asarray(angles[lo:hi])[sel_rows]
    coa = np.asarray(coadds[lo:hi])[sel_rows]
    exp = np.asarray(exptimes[lo:hi])[sel_rows]
    aligned, shifts = geometry.align_cube(cube, st)
    del cube
    rng = np.random.default_rng([int(st["seed"]), c, cn])
    centered, center = geometry.center_cube(aligned, st, rng)
    del aligned
    sel = select.measure_and_select(centered, ang, st, cn, pre=True)
    res = {"chunk": c, "cn": cn, "n_in": int(centered.shape[0]), "bad": sel.bad.tolist(),
           "shifts": shifts.tolist(), "center": list(center), "corrs": sel.corrs.tolist(),
           "noises": sel.noises.tolist(), "peaks": sel.peaks.tolist()}
    if sel.keep.size == 0:
        res.update(binned=None, angles=[], exptimes=[], coadds=[])
        return res
    kept = select.peak_scaled(centered) if st["pre_peak_scale"] else centered
    kept = kept[sel.keep]
    b, ab, eb = binning.prebin(kept, ang[sel.keep], exp[sel.keep], st)
    res.update(binned=b.astype(np.float32), angles=ab.tolist(), exptimes=eb.tolist(), coadds=coa[sel.keep].tolist())
    return res


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
        frames, in that order), ``n`` (segment count) and ``states``.
    """
    log(f"{tag}: {len(files)} frames, initial background and bad-pixel map")
    init = combine.initial_background(files)
    badmap = combine.make_badmap(master, init["badtest"], st["badsigthresh"], int(st["badsz"]), int(st["badbuff"]), bool(st["fresh_badmap"]))
    _FILES[tag] = list(files)
    tasks = [(tag, lo, hi, init["bg"], badmap, st) for lo, hi in _blocks(len(files), 500)]
    states = [s for part in _run(_states_task, tasks, workers) for s in part]
    angles = None
    if st["fix_pa"]:
        angles = combine.parallactic_angles(states, float(st["ra"]), float(st["dec"]), float(st["lat"]), float(st["lng"]))
    segs = combine.build_segments(states, st, sky_round, angles)
    order = combine.segment_order(segs)
    segs = [segs[i] for i in order]
    n = len(segs)
    log(f"{tag}: {sum(s.star_x >= 0 for s in states)} frames under the background limit, {n} segments")
    path = os.path.join(work, f"{tag}_segments.f32")
    np.memmap(path, dtype=np.float32, mode="w+", shape=(max(n, 1), 512, 512)).flush()
    blocks = _blocks(n, 200)
    _run(_segments_task, [(list(range(lo, hi)), segs[lo:hi], tag, st, path, n) for lo, hi in blocks], workers)
    return {"badmap": badmap, "segments": segs, "seg_path": path, "n": n, "states": states}


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
    tasks = [(lo, hi, rnd["seg_path"], path, n, badmap, st, sky_info) for lo, hi in _blocks(n, 250)]
    noise = [x for part in _run(_chopsub_task, tasks, workers) for x in part]
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
    Outputs, in ``out_dir``, with the IDL predecessor's names:
    ``<night>_<side>_cube<n>_align_cen_clean_bin.fits`` (frames, 2 radius x 2 radius, star at
    (radius - 0.5, radius - 0.5)), ``<night>_<side>_angles<n>_clean_bin.fits`` (parallactic
    angles in degrees) and ``.csv`` (angles and summed exposure times),
    ``<night>_<side>_coadds<n>_clean_bin.csv``, ``<night>_<side>_cube<n>_align_cen_clean_median_PSF.fits``,
    ``<night>_<side>_cube<n>_align_cen_median_PSF.fits`` (before the clean), and
    ``<night>_<side>_badmap.fits``, for n = 1 (chop A) and 2 (chop B).
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
    start = st["start_frame_left"] if side == "left" else st["start_frame_right"]
    end = st["end_frame_left"] if side == "left" else st["end_frame_right"]
    sci = select_science_frames(archive, night, start, end, frame_range=st.get("frames"))
    sky_dirs = st.get("sky_dirs") or [f"skys{min(k, 3)}" for k in range(1, int(st["n_sky"]) + 1)]
    sky_lists = [list_raw(os.path.join(archive, night, d)) for d in sky_dirs[:int(st["n_sky"])]]
    master = np.asarray(fits.getdata(os.path.join(archive, st["badmap_master"])), dtype=np.float64)
    rounds = [combine_round(sci, st, False, master, work, f"{side}_round0", workers)]
    for k, files in enumerate(sky_lists, 1):
        rounds.append(combine_round(files, st, True, master, work, f"{side}_round{k}", workers))
    badmap = rounds[-1]["badmap"]
    write_fits(os.path.join(out_dir, f"{pre}_badmap.fits"), badmap)
    skies = []
    for k in range(len(rounds) - 1, 0, -1):
        path = chopsub_round(rounds[k], badmap, st, work, f"{side}_round{k}", workers)
        nk = rounds[k]["n"]
        mmk = np.memmap(path, dtype=np.float32, mode="r", shape=(max(nk, 1), 512, 512))
        sk = chopsub.mean_skies(np.asarray(mmk[:nk]), [s.chop for s in rounds[k]["segments"]])
        sk["avginum"] = float(np.mean([float(s.inum) for s in rounds[k]["segments"]]))
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
    raw_pos = []
    for lo, hi in chunks:
        ch = np.asarray(chops[lo:hi])
        first = geometry.padded(np.asarray(mm[lo], np.float64), fw)
        second = geometry.padded(np.asarray(mm[min(lo + 1, hi - 1)], np.float64), fw)
        ia = int(np.nonzero(ch == "CHOP_A")[0][0]) + lo
        check = geometry.padded(np.asarray(mm[ia], np.float64), fw)
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
    results = _run(_chunk_task, tasks, workers)
    summary = {"night": night, "side": side, "settings_source": settings.source, "n_raw": len(sci),
               "sky_dirs": sky_dirs[:int(st["n_sky"])], "n_segments": n, "chunks": len(chunks),
               "positions": raw_pos, "cubes": {}}
    for cn in (1, 2):
        res = [r for r in results if r["cn"] == cn]
        parts = [r for r in res if r["binned"] is not None]
        big = np.concatenate([r["binned"] for r in parts]) if parts else np.zeros((0, 2 * st["radius"], 2 * st["radius"]), np.float32)
        big[big == 0] = np.nan
        ang = np.concatenate([r["angles"] for r in parts]) if parts else np.zeros(0)
        exp = np.concatenate([r["exptimes"] for r in parts]) if parts else np.zeros(0)
        coa = np.concatenate([r["coadds"] for r in parts]) if parts else np.zeros(0)
        write_fits(os.path.join(out_dir, f"{pre}_cube{cn}_align_cen_median_PSF.fits"), idl.median(big, axis=0, even=True))
        rep = np.stack([binning.replace_interior_nans(f) for f in big]).astype(np.float32) if big.shape[0] else big
        sel = select.measure_and_select(rep, ang, st, cn, pre=False)
        fin = select.peak_scaled(rep) if st["peak_scale"] else rep
        fin = fin[sel.keep]
        ang_f, exp_f = ang[sel.keep], exp[sel.keep]
        coa_f = np.delete(coa, sel.bad[sel.bad < coa.size])
        write_fits(os.path.join(out_dir, f"{pre}_cube{cn}_align_cen_clean_bin.fits"), fin,
                   {"RADIUS": int(st["radius"]), "XCEN": st["radius"] - 0.5, "YCEN": st["radius"] - 0.5,
                    "PXSCALE": st["pxscale"], "CHOP": "A" if cn == 1 else "B", "SIDE": side})
        write_fits(os.path.join(out_dir, f"{pre}_cube{cn}_align_cen_clean_median_PSF.fits"), idl.median(fin, axis=0))
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
                                "angle_range": [float(np.min(ang_f)), float(np.max(ang_f))] if ang_f.size else None,
                                "centers": [r["center"] for r in res]}
    summary["elapsed_s"] = round(time.time() - t0, 1)
    with open(os.path.join(out_dir, f"{pre}_summary.json"), "x") as fh:
        json.dump(summary, fh, indent=1)
    log(f"done in {summary['elapsed_s']} s: " + ", ".join(f"cube{cn} {v['final']} frames" for cn, v in summary["cubes"].items()))
    return summary
