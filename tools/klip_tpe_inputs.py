"""Prepare one or more nights' port products as KLIP-TPE generic inputs, and the command that runs them.

usage: python tools/klip_tpe_inputs.py ARCHIVE NIGHT [NIGHT ...] --tag TAG [--out-folder FOLDER]
       [--labels LABEL ...] [--sides left right] [--half 120]
       [--command-archive PATH] [--min-frames 10] [--display show|none] [--ref-min-dpa DEG]
       [--psf-half 30] [--product all|clean] [--ref NIGHT ...] [--ann-edges 11 34 67]
       [--n-iter 400 400] [--n-init 60 60] [--n-remeasure 3] [--contrast0 1e-3] [--workers auto]
       [--use-contrast C ...] [--keep-nan] [--min-coverage F] [--link BASE ...] [--batch N]

KLIP-TPE's generic adapter (``klip-tpe generic``) takes one registered cube, its derotation angles
and a PSF template per partition, and optionally a reference cube (RDI). For each side and chop of
each ``NIGHT`` with at least ``--min-frames`` frames this writes, under
``<ARCHIVE>/<FOLDER>/klip_tpe/inputs_<TAG>/`` (``FOLDER`` is the night, or ``--out-folder`` when
several nights are joined, such as the combined folder ``Procyon_20211219-20211220``):

* after checking that the star is at the center (``--max-offset``), the binned frames
  (``--product all``: every binned frame, ``<pre>_cube<n>_align_cen_bin.fits``,
  for KLIP-TPE's own frame selection; ``clean``: the IDL clean selection) cut to (2 half)² pixels
  with the star at the array center, so that KLIP-TPE's default center applies, without the frames
  that have a NaN pixel within the cut (``finite_half``): a cut that reaches past the detector's
  edge carries NaN padding there, and up to klip-tpe ``aa0702b`` the high-pass spreads one NaN over
  the whole frame, which empties every bin the frame falls in (``--keep-nan`` keeps those frames,
  for klip-tpe ``bf99474`` and later, which take NaN as no data); and without the frames whose star
  is more than ``--max-frame-offset`` from the center (``frame_offsets``; a chunk whose centering
  failed);
* their parallactic angles (KLIP-TPE derotates counterclockwise by angle + truenorth, as the IDL
  predecessor does with ``rot(frame, -angle - truenorth)``), unwrapped in time order so that a
  night whose angles cross ±180° (a transit north of the zenith) has no jump: KLIP-TPE's reference
  selection uses the plain difference |angle_i - angle_j|, and a derotation by angle + 360° is the
  same rotation. Binned frames whose angle was averaged across the wrap by a port version before
  the wrap-aware mean (``reduce_nomic.angles.wrap_outliers``) are left out rather than given a
  guessed angle, and listed in the manifest;
* the median PSF of the IDL-clean frames cut to (2 psf_half)² pixels as the injection template:
  KLIP-TPE normalizes it to unit total flux, so contrasts are flux ratios to the star's flux
  within that box (the frames are peak-scaled, so every frame shares the unit);
* with ``--ref``: the same side and chop of the given nights' IDL-clean frames, cut the same way
  and stacked, as the reference library, without cubes of fewer than ``--min-frames`` frames,
  frames whose angle was averaged across the wrap, frames with NaN within the cut (unless
  ``--keep-nan``) or with the star off the center, and frames within ``--ref-min-dpa`` of any
  science frame's angle. A sky-fixed companion
  sits at detector angle φ - q, so in a reference frame whose angle q_r is close to a science
  frame's q_s it sits where it does in the science frame, and RDI would subtract it.

and ``<ARCHIVE>/<FOLDER>/klip_tpe/run_<TAG>.sh``, the ``klip-tpe generic`` command over all the
partitions, writing to ``<ARCHIVE>/<FOLDER>/klip_tpe/run_<TAG>/``, with ``run_<TAG>_smoke.sh``, a
10-evaluation test of the first annulus writing to ``smoke_<TAG>/``, and ``resume_<TAG>.sh``, which
resumes ``run_<TAG>/`` from its checkpoint (``klip-tpe resume --instrument generic`` with the run's
data arguments: klip-tpe's run record restores the data arguments only of runs made with
``--root``, and the checkpoint holds the rest of the search's configuration). All three need the
environment variable ``KT`` set to the ``klip-tpe`` executable to use (a pinned clone's);
``WORKERS`` sets the worker budget (default ``--workers``, 16 for the smoke test). With ``--link``
the three run ``tools/klip_tpe_linked.py``, copied into ``inputs_<TAG>/``, with the Python beside
``KT``, so that the bases given are linked across the partitions. Arguments given to a script go to
the end of its klip-tpe command line, such as ``--gpu-share auto --gpu-device GPU-...`` (a card by
the UUID ``nvidia-smi -L`` shows).

Parameters
----------
ARCHIVE : str
    NOMIC archive root.
NIGHT : str, one or more
    Night folders, e.g. ``Vega_20220521``; each one's ``reduced_py`` holds the port's products.
    With several, every side and chop of every night is a partition of one joint run.
--tag : str
    Name of this input set and run, e.g. ``adi`` or ``rdi``.
--out-folder : str, default NIGHT
    Folder of the archive that receives ``klip_tpe/``; needed with several nights, e.g. the
    combined folder ``Procyon_20211219-20211220``.
--labels : str list, optional
    One label per night, put before each partition's name (``<label>_<side>_<chop>``, such as
    ``P3_left_A``), so that the nights' partitions differ. With one night the default is none
    (``left_A``); with several, each night folder's last ``_``-separated part (its UT date).
--sides : str list, default left right
    Sides to include (a side without products is skipped).
--half : int, default 120
    Half-width of the cut frames, pixels (240 px = 4.3 arcsec). The 2022 nights need 80 (160 px =
    2.9 arcsec): their star sat 78-110 px from the detector's edge in y, so a wider cut leaves
    most frames with NaN (``finite_half_px`` in the manifest says how far each partition's frames
    are finite). 80 px still holds the outer annulus edge, 67 px, plus half of KLIP-TPE's widest
    high-pass, 25 px.
--psf-half : int, default 30
    Half-width of the PSF template, pixels (60 px = 1.07 arcsec, about 2 λ/D at 11.1 µm).
--product : {"all", "clean"}, default "all"
    Which binned cube to use.
--ref : str list, optional
    Nights whose same-side, same-chop IDL-clean frames form the reference library; none of them
    may be a science night.
--ref-min-dpa : float, required with ``--ref``
    Smallest angle difference, degrees, between a reference frame and every science frame (0
    keeps every reference frame). A companion at separation r moves by r Δq between frames, so
    one λ/D at the inner edge r_in needs Δq = (λ/D) / r_in (78° at 0.2″, 39° at 0.4″ at 11.1 µm).
--command-archive : str, default ARCHIVE
    Archive root to write into the command and the manifest, when the archive is mounted
    elsewhere where this runs (the device sandbox sees /mnt/RAID64TB/NOMIC as ~/mnt/NOMIC).
--min-frames : int, default 10
    Fewest frames a partition needs, binned and again after the frames with NaN within the cut or
    with the star off the center are left out; a side and chop with fewer is left out, and the
    manifest lists it under
    ``skipped`` with its frame count. A reference night's cube with fewer binned frames is left
    out of the library too.
--max-offset : float, default 3.0
    Largest distance, pixels, of the star's peak from the center, found in the median of the cut
    frames (and in the cut PSF) within 40 px of the center (``star_offset``); a science partition
    or reference cube beyond it, or whose peak there is below half the median's maximum anywhere
    (no star near the center), is left out (a failed split or centering, such as Vega 2022-05-20's
    right chop B), with the peak in the manifest.
--max-frame-offset : float, default 5.0
    Largest distance, pixels, of a single frame's brightest pixel within 40 px of the center from
    the center (``frame_offsets``); science and reference frames beyond it are left out
    (``dropped_off_center``, ``off_center``). Well-centered Vega frames lie within 2.5 px; the
    frames of 2022-05-21 left chop B chunk 88, whose Gaussian center fit failed, lie 40 px off.
--display : {"show", "none"}, default "show"
    ``show`` writes ``--show`` into both scripts (KLIP-TPE's live window), ``none`` writes
    ``--no-display``.
--ann-edges, --n-iter, --n-init, --n-remeasure, --contrast0, --workers
    Written into the command (KLIP-TPE options of the same names; annulus edges in pixels).
--use-contrast : float list, optional
    Injection contrast per annulus, in the annuli's order, held fixed instead of calibrated
    (KLIP-TPE's ``--use-contrast``). A 0 calibrates that annulus, and so does every annulus past
    the end of the list. Written into both scripts and recorded in the manifest
    (``use_contrast``); without it every annulus calibrates.
--keep-nan : flag
    Keep the science and reference frames that have NaN within the cut (past the detector's
    edge) instead of leaving them out, for klip-tpe ``bf99474`` and later, which take NaN as no
    data and place injections and noise apertures only where enough frames have data. Frames
    with the star off the center are still left out. The manifest records the flag
    (``keep_nan``) and, per partition, the frames with NaN (``frames_with_nan``) and the NaN
    fraction of the cut (``nan_fraction``).
--min-coverage : float, optional
    Fewest of the frames, as a fraction from 0 to 1, that must have data where KLIP-TPE places a
    source or a noise aperture (its ``--min-coverage``, default 0.5; 0 masks only where no
    frame has data). Written into both scripts and recorded in the manifest
    (``min_coverage``); needs klip-tpe ``bf99474`` or later.
--link : str list, optional
    Reduction parameters that take one value for every partition (each chop), such as ``bin filter
    angsep anglemax``; without it klip-tpe links its default, ``bin`` and ``filter``. The three
    scripts then run the command through ``klip_tpe_linked.py`` (its docstring says how), and the
    manifest records the bases, the wrapper's path and its SHA-256 (``link``, None without
    ``--link``).
--batch : int, optional
    Trials in flight at once (KLIP-TPE's ``--batch``, which keeps that many trials going and asks
    each new proposal from the history as the oldest trial comes in); without it one trial runs at a
    time. A trial has only a few jobs, one per injected draw and chop (Procyon_4's adi3ann4: 3 to 6),
    so one at a time leaves most of a large machine and its GPU idle. TPE then proposes from a
    posterior up to that many trials old, which changes the search's trajectory. Written into both
    scripts and recorded in the manifest (``batch``, 1 without it).

Returns
-------
inputs_<TAG>/ : files
    With ``<pre>`` = ``<NIGHT>_<side>``:
    ``<pre>_cube<n>_<product>_c<2 half>.fits`` (float32, shape (n_frames, 2 half, 2 half)),
    ``<pre>_angles<n>_<product>.fits`` (degrees, unwrapped), ``<pre>_cube<n>_psf<2 psf_half>.fits``,
    with ``--ref`` ``<pre>_cube<n>_ref_c<2 half>.fits``, and ``manifest.json`` (sources with their
    SHA-256, cuts, frame counts, first and last angle, rotation span, whether the angles were
    unwrapped, the binned frames left out for a wrongly averaged angle (``dropped_wrap_averaged``)
    or for NaN within the cut (``dropped_nan``, indices in the binned cube), how far the frames are
    finite (``finite_half_px``: the smallest, 5th-percentile and median half-width of the largest
    finite centered square, at most ``--half``), the frames left out with the star off the center
    (``dropped_off_center``: index in the binned cube and offset in pixels), the reference frames
    with NaN (``nan_within_cut`` per reference night, left out unless ``nan_kept``) or with the
    star off the center (``off_center``), the skipped partitions, the fixed injection contrasts
    (``use_contrast``, None when every annulus calibrates), ``keep_nan``, ``min_coverage``, ``link``
    and ``batch``, the science nights (``nights``, with ``labels``; ``night`` is FOLDER), and per
    partition its night and the frames kept with NaN within the cut (``frames_with_nan``) and the
    cut's NaN fraction (``nan_fraction``)); with ``--link`` also ``klip_tpe_linked.py``.
run_<TAG>.sh, run_<TAG>_smoke.sh, resume_<TAG>.sh : file
    The run, its smoke test and its resume, for Kevin to check and run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reduce_nomic.angles import wrap_outliers  # noqa: E402


def sha256_of(path: str, block: int = 1 << 24) -> str:
    """SHA-256 of a file, read in blocks.

    Parameters
    ----------
    path : str
        File to hash.
    block : int, default 16777216
        Read size, bytes.

    Returns
    -------
    digest : str
        Hexadecimal SHA-256.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(block)
            if not b:
                return h.hexdigest()
            h.update(b)


def cut_frames(path: str, half: int) -> np.ndarray:
    """Cut every frame of a port cube (or a single image) around the star.

    Parameters
    ----------
    path : str
        Port FITS product with the star at (XCEN, YCEN) (default (nx/2 - 0.5, ny/2 - 0.5)), shape
        (n, ny, nx) or (ny, nx).
    half : int
        Half-width of the cut, pixels.

    Returns
    -------
    cut : ndarray, float32, shape (n, 2 half, 2 half) or (2 half, 2 half)
        The cut, with the star at ((2 half - 1) / 2, (2 half - 1) / 2), the array center.
    """
    with fits.open(path, memmap=True) as hd:
        d = hd[0].data
        h = hd[0].header
        ny, nx = d.shape[-2:]
        xc = float(h.get("XCEN", nx / 2 - 0.5))
        yc = float(h.get("YCEN", ny / 2 - 0.5))
        x0, y0 = int(round(xc + 0.5)) - half, int(round(yc + 0.5)) - half
        if x0 < 0 or y0 < 0 or x0 + 2 * half > nx or y0 + 2 * half > ny:
            raise ValueError(f"{os.path.basename(path)}: a {2 * half} px cut around ({xc}, {yc}) leaves the frame")
        return np.array(d[..., y0:y0 + 2 * half, x0:x0 + 2 * half], dtype=np.float32)


def star_offset(cube: np.ndarray, n_max: int = 60, radius: float = 40.0) -> Tuple[float, Tuple[int, int], float]:
    """Where the star is in cut frames: the peak of their median near the center.

    The search is limited to a disk around the center, because a bright artifact far out (an
    edge, the other aperture's image) can outshine the star in the median of peak-scaled frames.

    Parameters
    ----------
    cube : ndarray, shape (n, ny, nx) or (ny, nx)
        Cut frames (or one image) that should have the star at ((nx - 1) / 2, (ny - 1) / 2).
    n_max : int, default 60
        Most frames, evenly spaced, whose median is used.
    radius : float, default 40.0
        Radius of the disk searched, pixels (the whole image when that is smaller).

    Returns
    -------
    offset : float
        Distance of the brightest pixel within the disk from the center, pixels (inf when the
        median has no finite pixel there).
    peak : (int, int)
        (x, y) of that pixel.
    ratio : float
        Its value over the median image's maximum anywhere (1 when the star is the brightest
        thing; a small value when the disk holds no star).
    """
    c = np.asarray(cube)
    img = c if c.ndim == 2 else np.nanmedian(c[:: max(1, c.shape[0] // n_max)], axis=0)
    ny, nx = img.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(xx - (nx - 1) / 2.0, yy - (ny - 1) / 2.0)
    disk = np.where((r <= radius) & np.isfinite(img), img, np.nan)
    if not np.any(np.isfinite(disk)):
        return float("inf"), (-1, -1), 0.0
    y, x = np.unravel_index(np.nanargmax(disk), img.shape)
    top = float(np.nanmax(img))
    return float(r[y, x]), (int(x), int(y)), float(img[y, x] / top) if top > 0 else 0.0


def finite_half(cube: np.ndarray) -> np.ndarray:
    """How far from the center each cut frame is finite: the half-width of its largest finite centered square.

    A cut that reaches past the detector's edge carries NaN padding there (the 2022 nights, whose
    star sat 78-110 px from the edge in y). KLIP-TPE's high-pass spreads a NaN pixel over the
    whole frame, so a frame is usable only within a square that holds no NaN.

    Parameters
    ----------
    cube : ndarray, shape (n, 2 half, 2 half)
        Cut frames with the star at the array center, ((2 half - 1) / 2, (2 half - 1) / 2), where
        half is the cut's half-width in pixels.

    Returns
    -------
    h : ndarray, int, shape (n,)
        For each frame, the largest half-width, pixels, of a centered (2 h)² square with no NaN
        or infinite pixel; ``half`` for a frame that is finite everywhere.
    """
    c = np.asarray(cube)
    n, ny, nx = c.shape
    half = min(ny, nx) // 2
    yy, xx = np.mgrid[0:ny, 0:nx]
    # Chebyshev distance of each pixel from the center; a centered (2h)^2 square holds the pixels
    # with distance < h, so a bad pixel at distance d (= k + 0.5) allows h <= d - 0.5.
    dist = np.maximum(np.abs(xx - (nx - 1) / 2.0), np.abs(yy - (ny - 1) / 2.0)).astype(np.float32)
    out = np.full(n, half, dtype=int)
    for k in range(n):
        bad = ~np.isfinite(c[k])
        if bad.any():
            out[k] = min(half, int(round(float(dist[bad].min()) - 0.5)))
    return out


def frame_offsets(cube: np.ndarray, radius: float = 40.0) -> np.ndarray:
    """Where the star is in each cut frame: the distance of its brightest pixel near the center from the center.

    A chunk whose centering failed (the port's fallback estimator found the wrong spot, as in
    2022-05-21 left chop B chunk 88, put 43 px off) gives binned frames with the star far from the
    center. Such a frame would leave a bright residual in the reduction, where it could pass for a
    companion, and the median-image check (``star_offset``) cannot see a few of them.

    Parameters
    ----------
    cube : ndarray, shape (n, ny, nx)
        Cut frames that should have the star at ((nx - 1) / 2, (ny - 1) / 2), where n is the
        number of frames and ny, nx their size in pixels.
    radius : float, default 40.0
        Radius of the disk around the center searched in each frame, pixels.

    Returns
    -------
    offset : ndarray, float64, shape (n,)
        For each frame, the distance of the brightest finite pixel within the disk from the center,
        pixels (at least 0.71 px for the half-pixel center of an even-sized frame; inf for a frame
        with no finite pixel there).
    """
    c = np.asarray(cube)
    n, ny, nx = c.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(xx - (nx - 1) / 2.0, yy - (ny - 1) / 2.0)
    disk = r <= radius
    rd = r[disk]
    out = np.full(n, np.inf)
    for k in range(n):
        v = np.asarray(c[k][disk], dtype=np.float64)
        good = np.isfinite(v)
        if good.any():
            out[k] = float(rd[np.argmax(np.where(good, v, -np.inf))])
    return out


def unwrap_degrees(angles: np.ndarray) -> np.ndarray:
    """Remove ±360° jumps from a time-ordered sequence of angles.

    Parameters
    ----------
    angles : ndarray, shape (n,)
        Parallactic angles in time order, degrees.

    Returns
    -------
    unwrapped : ndarray, float64, shape (n,)
        The same angles with multiples of 360° added where consecutive values differ by more
        than 180°, so that consecutive values differ by the actual rotation; the first value is
        unchanged.
    """
    a = np.asarray(angles, dtype=np.float64).ravel()
    return np.rad2deg(np.unwrap(np.deg2rad(a))) if a.size else a


def main(argv: Sequence[str]) -> int:
    """Write the inputs and the command for one night, or for several nights joined.

    Parameters
    ----------
    argv : sequence of str
        Command-line arguments without the program name (see the module docstring).

    Returns
    -------
    status : int
        0 when at least one partition was written, 1 otherwise.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("archive")
    ap.add_argument("nights", nargs="+", metavar="night")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-folder", default=None)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--sides", nargs="+", default=["left", "right"])
    ap.add_argument("--half", type=int, default=120)
    ap.add_argument("--psf-half", type=int, default=30)
    ap.add_argument("--product", choices=["all", "clean"], default="all")
    ap.add_argument("--ref", nargs="*", default=[])
    ap.add_argument("--ann-edges", nargs="+", default=["11", "34", "67"])
    ap.add_argument("--n-iter", nargs="+", default=["400", "400"])
    ap.add_argument("--n-init", nargs="+", default=["60", "60"])
    ap.add_argument("--n-remeasure", default="3")
    ap.add_argument("--contrast0", default="1e-3")
    ap.add_argument("--workers", default="auto")
    ap.add_argument("--command-archive", default=None)
    ap.add_argument("--min-frames", type=int, default=10)
    ap.add_argument("--display", choices=["show", "none"], default="show")
    ap.add_argument("--ref-min-dpa", type=float, default=None)
    ap.add_argument("--max-offset", type=float, default=3.0)
    ap.add_argument("--max-frame-offset", type=float, default=5.0)
    ap.add_argument("--use-contrast", nargs="+", default=None)
    ap.add_argument("--keep-nan", action="store_true")
    ap.add_argument("--min-coverage", type=float, default=None)
    ap.add_argument("--link", nargs="+", default=None)
    ap.add_argument("--batch", type=int, default=None)
    a = ap.parse_args(argv)
    if a.batch is not None and a.batch < 1:
        ap.error("--batch is a number of trials, at least 1")
    if a.min_coverage is not None and not (np.isfinite(a.min_coverage) and 0.0 <= a.min_coverage <= 1.0):
        ap.error("--min-coverage is a fraction of the frames, from 0 to 1")
    link: Optional[List[str]] = None
    if a.link is not None:
        bad = [b for b in a.link if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", b)]
        if bad:
            ap.error(f"--link takes parameter names, not {' '.join(bad)}")
        link = list(dict.fromkeys(a.link))
    wrapper_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "klip_tpe_linked.py")
    if a.ref and a.ref_min_dpa is None:
        ap.error("--ref needs --ref-min-dpa (degrees; 0 keeps every reference frame)")
    use_contrast: Optional[List[float]] = None
    if a.use_contrast is not None:
        try:
            use_contrast = [float(v) for v in a.use_contrast]
        except ValueError:
            ap.error(f"--use-contrast takes numbers, not {' '.join(a.use_contrast)}")
        if not all(np.isfinite(v) and v >= 0 for v in use_contrast):
            ap.error("--use-contrast values must be finite and at least 0 (0 calibrates that annulus)")
    extra_opts = f" --use-contrast {' '.join(a.use_contrast)}" if a.use_contrast else ""
    if a.min_coverage is not None:
        extra_opts += f" --min-coverage {a.min_coverage:g}"
    if a.batch is not None and a.batch > 1:
        extra_opts += f" --batch {a.batch}"
    display = "--show" if a.display == "show" else "--no-display"
    nights = list(a.nights)
    if len(set(nights)) != len(nights):
        ap.error("a night is given twice")
    if len(nights) > 1 and not a.out_folder:
        ap.error("several nights need --out-folder, the folder of the archive that receives klip_tpe/")
    if set(a.ref) & set(nights):
        ap.error(f"--ref night(s) {sorted(set(a.ref) & set(nights))} are science nights too")
    if a.labels is not None:
        if len(a.labels) != len(nights):
            ap.error(f"--labels takes one label per night ({len(nights)})")
        bad = [x for x in a.labels if not re.fullmatch(r"[A-Za-z0-9]+", x)]
        if bad:
            ap.error(f"--labels takes letters and digits only, not {' '.join(bad)}")
        labels = list(a.labels)
    elif len(nights) > 1:
        labels = [n.rsplit("_", 1)[-1] for n in nights]
    else:
        labels = [""]
    if len(set(labels)) != len(labels):
        ap.error(f"the nights' labels {labels} are not all different; give --labels")
    folder = a.out_folder or nights[0]
    title = folder if len(nights) == 1 else f"{folder} ({' + '.join(nights)})"
    night_dir = os.path.join(a.archive, folder)          # the folder that receives klip_tpe/
    out = os.path.join(night_dir, "klip_tpe", f"inputs_{a.tag}")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; choose another --tag")
    tag = "_clean" if a.product == "clean" else ""
    manifest: Dict = {"night": folder, "nights": nights, "labels": labels if len(nights) > 1 or a.labels else None,
                      "tag": a.tag, "product": a.product, "half": a.half,
                      "psf_half": a.psf_half, "ref_nights": a.ref, "min_frames": a.min_frames,
                      "use_contrast": use_contrast, "keep_nan": bool(a.keep_nan),
                      "min_coverage": a.min_coverage, "link": None, "batch": int(a.batch or 1),
                      "partitions": [], "skipped": []}
    cubes: List[str] = []
    angles: List[str] = []
    psfs: List[str] = []
    refs: List[str] = []
    names: List[str] = []
    for night, label in zip(nights, labels):
        red = os.path.join(a.archive, night, "reduced_py")
        for side in a.sides:
            pre = f"{night}_{side}"
            for cn in (1, 2):
                pname = f"{label}_{side}_{'AB'[cn - 1]}" if label else f"{side}_{'AB'[cn - 1]}"
                src = os.path.join(red, f"{pre}_cube{cn}_align_cen{tag}_bin.fits")
                asrc = os.path.join(red, f"{pre}_angles{cn}{tag}_bin.fits")
                psrc = os.path.join(red, f"{pre}_cube{cn}_align_cen_clean_median_PSF.fits")
                if not all(os.path.exists(p) for p in (src, asrc, psrc)):
                    print(f"{pre} cube{cn}: products missing; skipped")
                    continue
                with fits.open(src, memmap=True) as hd:
                    n_src = 1 if hd[0].data.ndim == 2 else int(hd[0].data.shape[0])
                if n_src < a.min_frames:
                    print(f"{pre} cube{cn}: {n_src} frames, fewer than --min-frames {a.min_frames}; skipped")
                    manifest["skipped"].append({"name": pname, "frames": n_src,
                                                "reason": f"fewer than {a.min_frames} frames"})
                    continue
                cube = cut_frames(src, a.half)
                if cube.ndim == 2:
                    cube = cube[None]
                ang_raw = np.asarray(fits.getdata(asrc), dtype=np.float64).ravel()
                if len(ang_raw) != cube.shape[0]:
                    raise SystemExit(f"{asrc}: {len(ang_raw)} angles for {cube.shape[0]} frames")
                if not np.all(np.isfinite(ang_raw)):
                    raise SystemExit(f"{asrc}: {int(np.sum(~np.isfinite(ang_raw)))} angles are not finite")
                src_idx = np.arange(ang_raw.size)               # index of each kept frame in the binned cube
                bad = wrap_outliers(ang_raw)
                dropped = [{"index": int(i), "angle_deg": float(ang_raw[i])} for i in bad]
                if bad.size:
                    keep = np.setdiff1d(np.arange(ang_raw.size), bad)
                    cube, ang_raw, src_idx = cube[keep], ang_raw[keep], src_idx[keep]
                    print(f"{pre} cube{cn}: left out binned frame(s) {bad.tolist()} with angle(s) "
                          f"{[round(d['angle_deg'], 2) for d in dropped]} averaged across ±180")
                fh = finite_half(cube)
                fh_stats = {"min": int(fh.min()), "p05": int(np.floor(np.percentile(fh, 5))), "median": int(np.median(fh))}
                has_nan = fh < a.half
                n_with_nan = int(has_nan.sum())
                if a.keep_nan:
                    if n_with_nan:
                        print(f"{pre} cube{cn}: kept {n_with_nan} of {has_nan.size} frame(s) with NaN within the "
                              f"{2 * a.half} px cut (--keep-nan); finite within ±{fh_stats['min']} px in every frame")
                    has_nan = np.zeros_like(has_nan)
                dropped_nan = [int(i) for i in src_idx[has_nan]]
                if has_nan.any():
                    cube, ang_raw, src_idx = cube[~has_nan], ang_raw[~has_nan], src_idx[~has_nan]
                    print(f"{pre} cube{cn}: left out {len(dropped_nan)} of {has_nan.size} frame(s) with NaN within the "
                          f"{2 * a.half} px cut (past the detector's edge); finite within ±{fh_stats['min']} px in every "
                          f"frame, ±{fh_stats['p05']} px in 95 %")
                if cube.shape[0] < a.min_frames:
                    print(f"{pre} cube{cn}: {cube.shape[0]} frames finite within the cut, fewer than --min-frames "
                          f"{a.min_frames}; skipped")
                    manifest["skipped"].append({"name": pname, "frames": int(cube.shape[0]),
                                                "reason": f"fewer than {a.min_frames} frames finite within the "
                                                          f"{2 * a.half} px cut ({len(dropped_nan)} with NaN left out)",
                                                "finite_half_px": fh_stats})
                    continue
                psf = cut_frames(psrc, a.psf_half)
                if not np.all(np.isfinite(psf)):
                    print(f"{pre} cube{cn}: the PSF template has {int(np.sum(~np.isfinite(psf)))} NaN pixel(s) within "
                          f"its {2 * a.psf_half} px cut; skipped")
                    manifest["skipped"].append({"name": pname, "frames": int(cube.shape[0]),
                                                "reason": f"PSF template not finite within its {2 * a.psf_half} px cut"})
                    continue
                off_c, pk_c, ratio_c = star_offset(cube)
                off_p, pk_p, ratio_p = star_offset(psf)
                if off_c > a.max_offset or off_p > a.max_offset or ratio_c < 0.5 or ratio_p < 0.5:
                    print(f"{pre} cube{cn}: star not at the center (median peak near the center at {pk_c} of "
                          f"{cube.shape[-1]} px, {ratio_c:.2f} of the maximum; PSF peak at {pk_p} of {psf.shape[-1]} px, "
                          f"{ratio_p:.2f}); skipped")
                    manifest["skipped"].append({"name": pname, "frames": int(cube.shape[0]),
                                                "reason": f"star not at the center: median peak at {pk_c} ({ratio_c:.2f} of the "
                                                          f"maximum), PSF peak at {pk_p} ({ratio_p:.2f}); limits {a.max_offset} px, 0.5"})
                    continue
                # The partition's median is centered; now single frames whose chunk was centered on the
                # wrong spot, which the median does not show.
                foff = frame_offsets(cube)
                is_off = foff > a.max_frame_offset
                dropped_off = [{"index": int(i), "offset_px": round(float(o), 1)} for i, o in zip(src_idx[is_off], foff[is_off])]
                if is_off.any():
                    cube, ang_raw, src_idx = cube[~is_off], ang_raw[~is_off], src_idx[~is_off]
                    print(f"{pre} cube{cn}: left out {len(dropped_off)} frame(s) whose star is more than "
                          f"{a.max_frame_offset} px from the center (index, px): "
                          f"{[(d['index'], d['offset_px']) for d in dropped_off]}")
                if cube.shape[0] < a.min_frames:
                    print(f"{pre} cube{cn}: {cube.shape[0]} frames with the star at the center, fewer than --min-frames "
                          f"{a.min_frames}; skipped")
                    manifest["skipped"].append({"name": pname, "frames": int(cube.shape[0]),
                                                "reason": f"fewer than {a.min_frames} frames with the star within "
                                                          f"{a.max_frame_offset} px of the center ({len(dropped_off)} left out)"})
                    continue
                ang = unwrap_degrees(ang_raw)
                unwrapped = bool(np.any(np.abs(ang - ang_raw) > 1e-9))
                rc, used, ref_info = None, [], []
                if a.ref:
                    stack = []
                    for rn in a.ref:
                        rpre = os.path.join(a.archive, rn, "reduced_py", f"{rn}_{side}")
                        rsrc, rasrc = f"{rpre}_cube{cn}_align_cen_clean_bin.fits", f"{rpre}_angles{cn}_clean_bin.fits"
                        if not (os.path.exists(rsrc) and os.path.exists(rasrc)):
                            continue
                        rframes = cut_frames(rsrc, a.half)
                        if rframes.ndim == 2:
                            rframes = rframes[None]
                        rang = np.asarray(fits.getdata(rasrc), dtype=np.float64).ravel()
                        if len(rang) != rframes.shape[0] or not np.all(np.isfinite(rang)):
                            raise SystemExit(f"{rasrc}: {len(rang)} angles for {rframes.shape[0]} frames, or angles not finite")
                        if rang.size < a.min_frames:                  # a reference cube this short is a failed reduction
                            ref_info.append({"night": rn, "frames": int(rang.size), "used": 0,
                                             "reason": f"fewer than {a.min_frames} frames"})
                            continue
                        off_r, pk_r, ratio_r = star_offset(rframes)
                        if off_r > a.max_offset or ratio_r < 0.5:     # star not where the cube says it is
                            ref_info.append({"night": rn, "frames": int(rang.size), "used": 0,
                                             "reason": f"star not at the center: median peak at {pk_r} ({ratio_r:.2f} of the maximum)"})
                            continue
                        ok = np.ones(rang.size, dtype=bool)
                        rbad = wrap_outliers(rang)
                        ok[rbad] = False
                        rfin = finite_half(rframes) >= a.half       # a reference frame with NaN in the cut would empty its bin too
                        if not a.keep_nan:
                            ok &= rfin
                        rcen = frame_offsets(rframes) <= a.max_frame_offset
                        ok &= rcen
                        dpa = np.abs((rang[:, None] - ang_raw[None, :] + 180.0) % 360.0 - 180.0).min(axis=1)
                        near = ok & (dpa < a.ref_min_dpa)
                        ok &= dpa >= a.ref_min_dpa
                        ref_info.append({"night": rn, "frames": int(rang.size), "wrap_averaged": int(rbad.size),
                                         "nan_within_cut": int((~rfin).sum()), "nan_kept": bool(a.keep_nan),
                                         "off_center": int((~rcen).sum()),
                                         "within_min_dpa": int(near.sum()), "used": int(ok.sum()),
                                         "sources": {os.path.basename(x): sha256_of(x) for x in (rsrc, rasrc)}})
                        if ok.any():
                            stack.append(rframes[ok])
                            used.append(rn)
                    if not stack:
                        print(f"{pre} cube{cn}: no reference frames from {a.ref} at least {a.ref_min_dpa} deg from the "
                              f"science angles; skipped")
                        continue
                    rc = np.concatenate(stack)
                c_out = os.path.join(out, f"{pre}_cube{cn}_{a.product}_c{2 * a.half}.fits")
                a_out = os.path.join(out, f"{pre}_angles{cn}_{a.product}.fits")
                p_out = os.path.join(out, f"{pre}_cube{cn}_psf{2 * a.psf_half}.fits")
                os.makedirs(out, exist_ok=True)              # made with the first partition written
                fits.PrimaryHDU(cube).writeto(c_out)
                fits.PrimaryHDU(ang).writeto(a_out)
                fits.PrimaryHDU(psf).writeto(p_out)
                part = {"name": pname, "night": night, "cube": c_out, "angles": a_out, "psf": p_out,
                        "frames": int(cube.shape[0]), "angle_first": float(ang[0]), "angle_last": float(ang[-1]),
                        "angle_min": float(ang.min()), "angle_max": float(ang.max()),
                        "rotation_deg": float(ang.max() - ang.min()), "unwrapped": unwrapped,
                        "dropped_wrap_averaged": dropped, "dropped_nan": dropped_nan, "finite_half_px": fh_stats,
                        "frames_with_nan": int(np.sum(finite_half(cube) < a.half)),
                        "nan_fraction": float(np.mean(~np.isfinite(cube))),
                        "dropped_off_center": dropped_off,
                        "sources": {os.path.basename(p): sha256_of(p) for p in (src, asrc, psrc)}}
                if rc is not None:
                    r_out = os.path.join(out, f"{pre}_cube{cn}_ref_c{2 * a.half}.fits")
                    fits.PrimaryHDU(rc).writeto(r_out)
                    part.update(ref=r_out, ref_frames=int(rc.shape[0]), ref_nights_used=used, ref_min_dpa=a.ref_min_dpa,
                                ref_detail=ref_info)
                    refs.append(r_out)
                manifest["partitions"].append(part)
                cubes.append(c_out); angles.append(a_out); psfs.append(p_out); names.append(part["name"])
                print(f"{pre} cube{cn}: {cube.shape[0]} frames, angles {ang[0]:.1f} -> {ang[-1]:.1f} deg "
                      f"(rotation {part['rotation_deg']:.1f} deg{', unwrapped across ±180' if unwrapped else ''})"
                      + (f", {part['ref_frames']} reference frames" if a.ref else ""))
    if not cubes:
        return 1
    root = a.command_archive or a.archive
    for part in manifest["partitions"]:
        for k in ("cube", "angles", "psf", "ref"):
            if k in part:
                part[k] = os.path.join(root, os.path.relpath(part[k], a.archive))
    cmd_root = a.command_archive or a.archive
    def there(path: str) -> str:
        """The path as seen from ``--command-archive``."""
        return os.path.join(cmd_root, os.path.relpath(path, a.archive))
    q = shlex.quote
    if link:
        wrapper = os.path.join(out, "klip_tpe_linked.py")
        shutil.copyfile(wrapper_src, wrapper)
        manifest["link"] = {"bases": link, "wrapper": there(wrapper), "wrapper_sha256": sha256_of(wrapper)}
        # the Python of the venv whose klip-tpe KT names runs the wrapper, so it imports that klip_tpe
        launch = ['PY="$(dirname "${KT:?}")/python"',
                  '[ -x "${PY:?}" ] || { echo "${PY} is not there: KT must be a virtual environment\'s klip-tpe"; exit 1; }',
                  f'"${{PY:?}}" {q(there(wrapper))} --link {" ".join(link)} --']
    else:
        launch = ['"${KT:?}"']
    with open(os.path.join(out, "manifest.json"), "x") as fh:
        json.dump(manifest, fh, indent=1)
    cubes, angles, psfs, refs = ([there(x) for x in lst] for lst in (cubes, angles, psfs, refs))
    run_dir = there(os.path.join(night_dir, "klip_tpe", f"run_{a.tag}"))
    data_args = ["  --cube " + " ".join(q(c) for c in cubes) + " \\",
                 "  --angles " + " ".join(q(x) for x in angles) + " \\",
                 "  --psf " + " ".join(q(x) for x in psfs) + " \\"]
    if refs:
        data_args.append("  --ref-cube " + " ".join(q(x) for x in refs) + " \\")
    data_args += ["  --names " + " ".join(names) + " \\",
                  "  --pxscale 0.0179 --lam 11.1e-6 --diam 8.4 \\"]
    preamble = ["#!/usr/bin/env bash",
                f"# KLIP-TPE generic run for {title} ({a.tag}); made by reduce_nomic tools/klip_tpe_inputs.py",
                "set -euo pipefail",
                'KT="${KT:?set KT to the klip-tpe executable of the pinned clone}"',
                f'RUN={q(run_dir)}']
    lines = preamble + ['[ ! -e "${RUN:?}" ] || { echo "${RUN} exists"; exit 1; }'] + launch[:-1] + [
        launch[-1] + " generic \\"] + data_args[:-2]
    head = list(lines)
    resume = preamble + [
        '[ -f "${RUN:?}/checkpoint.json" ] || { echo "${RUN} has no checkpoint.json"; exit 1; }',
        '[ ! -e "${RUN:?}/final_results.json" ] || { echo "${RUN} is already finished"; exit 1; }'] + launch[:-1] + [
        launch[-1] + " resume --instrument generic \\"] + data_args + [
        f'  --workers "${{WORKERS:-{a.workers}}}" {display} \\',
        '  --run-dir "${RUN:?}" "$@"']
    resume[1] = (f"# Resume the KLIP-TPE generic run for {title} ({a.tag}) from its checkpoint, which holds the "
                 f"search's configuration; made by reduce_nomic tools/klip_tpe_inputs.py")
    lines += ["  --names " + " ".join(names) + " \\",
              "  --pxscale 0.0179 --lam 11.1e-6 --diam 8.4 \\",
              f"  --ann-edges {' '.join(a.ann_edges)} --n-iter {' '.join(a.n_iter)} --n-init {' '.join(a.n_init)} \\",
              f"  --n-remeasure {a.n_remeasure} --n-top 6 --n-valid 8 --contrast0 {a.contrast0}{extra_opts} \\",
              f'  --workers "${{WORKERS:-{a.workers}}}" --seed 1 --verify {display} \\',
              '  --run-dir "${RUN:?}" "$@"']
    smoke_dir = there(os.path.join(night_dir, "klip_tpe", f"smoke_{a.tag}"))
    smoke = [ln.replace(f"RUN={q(run_dir)}", f"RUN={q(smoke_dir)}") for ln in head]
    smoke[1] = f"# KLIP-TPE smoke test for {title} ({a.tag}): 10 evaluations on the first annulus; its numbers mean nothing"
    smoke += ["  --names " + " ".join(names) + " \\",
              "  --pxscale 0.0179 --lam 11.1e-6 --diam 8.4 \\",
              f"  --ann-edges {' '.join(a.ann_edges[:2])} --n-iter 10 --n-init 5 --n-top 1 --n-valid 1 --contrast0 {a.contrast0}{extra_opts} \\",
              f'  --workers "${{WORKERS:-16}}" --seed 1 {display} \\',
              '  --run-dir "${RUN:?}" "$@"']
    for name, body in ((f"run_{a.tag}.sh", lines), (f"run_{a.tag}_smoke.sh", smoke), (f"resume_{a.tag}.sh", resume)):
        sh = os.path.join(night_dir, "klip_tpe", name)
        with open(sh, "x") as fh:
            fh.write("\n".join(body) + "\n")
        os.chmod(sh, 0o755)
    print(f"wrote {out}/, run_{a.tag}.sh, run_{a.tag}_smoke.sh and resume_{a.tag}.sh in "
          f"{os.path.join(night_dir, 'klip_tpe')}" + (f" (linking {', '.join(link)})" if link else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
