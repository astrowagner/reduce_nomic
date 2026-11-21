"""Prepare one night's port products as KLIP-TPE generic inputs, and the command that runs it.

usage: python tools/klip_tpe_inputs.py ARCHIVE NIGHT --tag TAG [--sides left right] [--half 120]
       [--command-archive PATH] [--min-frames 10] [--display show|none] [--ref-min-dpa DEG]
       [--psf-half 30] [--product all|clean] [--ref NIGHT ...] [--ann-edges 11 34 67]
       [--n-iter 400 400] [--n-init 60 60] [--n-remeasure 3] [--contrast0 1e-3] [--workers auto]

KLIP-TPE's generic adapter (``klip-tpe generic``) takes one registered cube, its derotation angles
and a PSF template per partition, and optionally a reference cube (RDI). For each side and chop of
``NIGHT`` with at least ``--min-frames`` frames this writes, under
``<ARCHIVE>/<NIGHT>/klip_tpe/inputs_<TAG>/``:

* after checking that the star is at the center (``--max-offset``), the binned frames
  (``--product all``: every binned frame, ``<pre>_cube<n>_align_cen_bin.fits``,
  for KLIP-TPE's own frame selection; ``clean``: the IDL clean selection) cut to (2 half)² pixels
  with the star at the array center, so that KLIP-TPE's default center applies;
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
  frames whose angle was averaged across the wrap and frames within ``--ref-min-dpa`` of any
  science frame's angle. A sky-fixed companion
  sits at detector angle φ - q, so in a reference frame whose angle q_r is close to a science
  frame's q_s it sits where it does in the science frame, and RDI would subtract it.

and ``<ARCHIVE>/<NIGHT>/klip_tpe/run_<TAG>.sh``, the ``klip-tpe generic`` command over all the
partitions, writing to ``<ARCHIVE>/<NIGHT>/klip_tpe/run_<TAG>/``, with ``run_<TAG>_smoke.sh``, a
10-evaluation test of the first annulus writing to ``smoke_<TAG>/``. Both need the environment
variable ``KT`` set to the ``klip-tpe`` executable to use (a pinned clone's); ``WORKERS`` sets the
worker budget (default ``--workers``, 16 for the smoke test).

Parameters
----------
ARCHIVE : str
    NOMIC archive root.
NIGHT : str
    Night folder, e.g. ``Vega_20220521``; its ``reduced_py`` holds the port's products.
--tag : str
    Name of this input set and run, e.g. ``adi`` or ``rdi``.
--sides : str list, default left right
    Sides to include (a side without products is skipped).
--half : int, default 120
    Half-width of the cut frames, pixels (240 px = 4.3 arcsec).
--psf-half : int, default 30
    Half-width of the PSF template, pixels (60 px = 1.07 arcsec, about 2 λ/D at 11.1 µm).
--product : {"all", "clean"}, default "all"
    Which binned cube to use.
--ref : str list, optional
    Nights whose same-side, same-chop IDL-clean frames form the reference library.
--ref-min-dpa : float, required with ``--ref``
    Smallest angle difference, degrees, between a reference frame and every science frame (0
    keeps every reference frame). A companion at separation r moves by r Δq between frames, so
    one λ/D at the inner edge r_in needs Δq = (λ/D) / r_in (78° at 0.2″, 39° at 0.4″ at 11.1 µm).
--command-archive : str, default ARCHIVE
    Archive root to write into the command and the manifest, when the archive is mounted
    elsewhere where this runs (the device sandbox sees /mnt/RAID64TB/NOMIC as ~/mnt/NOMIC).
--min-frames : int, default 10
    Fewest frames a partition needs; a side and chop with fewer is left out, and the manifest
    lists it under ``skipped`` with its frame count. A reference night's cube with fewer is left
    out of the library too.
--max-offset : float, default 3.0
    Largest distance, pixels, of the star's peak in the median of the cut frames (and in the cut
    PSF) from the center; a science partition or reference cube beyond it is left out (a failed
    split or centering, such as Vega 2022-05-20's chop B cubes), with the peak in the manifest.
--display : {"show", "none"}, default "show"
    ``show`` writes ``--show`` into both scripts (KLIP-TPE's live window), ``none`` writes
    ``--no-display``.
--ann-edges, --n-iter, --n-init, --n-remeasure, --contrast0, --workers
    Written into the command (KLIP-TPE options of the same names; annulus edges in pixels).

Returns
-------
inputs_<TAG>/ : files
    ``<pre>_cube<n>_<product>_c<2 half>.fits`` (float32, shape (n_frames, 2 half, 2 half)),
    ``<pre>_angles<n>_<product>.fits`` (degrees, unwrapped), ``<pre>_cube<n>_psf<2 psf_half>.fits``,
    with ``--ref`` ``<pre>_cube<n>_ref_c<2 half>.fits``, and ``manifest.json`` (sources with their
    SHA-256, cuts, frame counts, first and last angle, rotation span, whether the angles were
    unwrapped, the binned frames left out for a wrongly averaged angle, and the skipped
    partitions).
run_<TAG>.sh, run_<TAG>_smoke.sh : file
    The run and its smoke test, for Kevin to check and run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
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


def star_offset(cube: np.ndarray, n_max: int = 60) -> Tuple[float, Tuple[int, int]]:
    """Distance of the star's peak from the center of cut frames.

    Parameters
    ----------
    cube : ndarray, shape (n, ny, nx) or (ny, nx)
        Cut frames (or one image) that should have the star at ((nx - 1) / 2, (ny - 1) / 2).
    n_max : int, default 60
        Most frames, evenly spaced, whose median is used.

    Returns
    -------
    offset : float
        Distance of the brightest pixel of the median from the center, pixels (inf when the median
        has no finite pixel).
    peak : (int, int)
        (x, y) of that pixel.
    """
    c = np.asarray(cube)
    img = c if c.ndim == 2 else np.nanmedian(c[:: max(1, c.shape[0] // n_max)], axis=0)
    if not np.any(np.isfinite(img)):
        return float("inf"), (-1, -1)
    y, x = np.unravel_index(np.nanargmax(img), img.shape)
    ny, nx = img.shape
    return float(np.hypot(x - (nx - 1) / 2.0, y - (ny - 1) / 2.0)), (int(x), int(y))


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
    """Write the inputs and the command for one night.

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
    ap.add_argument("night")
    ap.add_argument("--tag", required=True)
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
    a = ap.parse_args(argv)
    if a.ref and a.ref_min_dpa is None:
        ap.error("--ref needs --ref-min-dpa (degrees; 0 keeps every reference frame)")
    display = "--show" if a.display == "show" else "--no-display"
    night_dir = os.path.join(a.archive, a.night)
    red = os.path.join(night_dir, "reduced_py")
    out = os.path.join(night_dir, "klip_tpe", f"inputs_{a.tag}")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; choose another --tag")
    tag = "_clean" if a.product == "clean" else ""
    manifest: Dict = {"night": a.night, "tag": a.tag, "product": a.product, "half": a.half,
                      "psf_half": a.psf_half, "ref_nights": a.ref, "min_frames": a.min_frames,
                      "partitions": [], "skipped": []}
    cubes: List[str] = []
    angles: List[str] = []
    psfs: List[str] = []
    refs: List[str] = []
    names: List[str] = []
    for side in a.sides:
        pre = f"{a.night}_{side}"
        for cn in (1, 2):
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
                manifest["skipped"].append({"name": f"{side}_{'AB'[cn - 1]}", "frames": n_src,
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
            bad = wrap_outliers(ang_raw)
            dropped = [{"index": int(i), "angle_deg": float(ang_raw[i])} for i in bad]
            if bad.size:
                keep = np.setdiff1d(np.arange(ang_raw.size), bad)
                cube, ang_raw = cube[keep], ang_raw[keep]
                print(f"{pre} cube{cn}: left out binned frame(s) {bad.tolist()} with angle(s) "
                      f"{[round(d['angle_deg'], 2) for d in dropped]} averaged across ±180")
            ang = unwrap_degrees(ang_raw)
            unwrapped = bool(np.any(np.abs(ang - ang_raw) > 1e-9))
            psf = cut_frames(psrc, a.psf_half)
            off_c, pk_c = star_offset(cube)
            off_p, pk_p = star_offset(psf)
            if off_c > a.max_offset or off_p > a.max_offset:
                print(f"{pre} cube{cn}: star not at the center (median peak at {pk_c} of {cube.shape[-1]} px, "
                      f"PSF peak at {pk_p} of {psf.shape[-1]} px); skipped")
                manifest["skipped"].append({"name": f"{side}_{'AB'[cn - 1]}", "frames": int(cube.shape[0]),
                                            "reason": f"star not at the center: median peak at {pk_c}, PSF peak at {pk_p}, "
                                                      f"more than {a.max_offset} px from the center"})
                continue
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
                    off_r, pk_r = star_offset(rframes)
                    if off_r > a.max_offset:                      # star not where the cube says it is
                        ref_info.append({"night": rn, "frames": int(rang.size), "used": 0,
                                         "reason": f"star not at the center: median peak at {pk_r}"})
                        continue
                    ok = np.ones(rang.size, dtype=bool)
                    rbad = wrap_outliers(rang)
                    ok[rbad] = False
                    dpa = np.abs((rang[:, None] - ang_raw[None, :] + 180.0) % 360.0 - 180.0).min(axis=1)
                    near = ok & (dpa < a.ref_min_dpa)
                    ok &= dpa >= a.ref_min_dpa
                    ref_info.append({"night": rn, "frames": int(rang.size), "wrap_averaged": int(rbad.size),
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
            part = {"name": f"{side}_{'AB'[cn - 1]}", "cube": c_out, "angles": a_out, "psf": p_out,
                    "frames": int(cube.shape[0]), "angle_first": float(ang[0]), "angle_last": float(ang[-1]),
                    "angle_min": float(ang.min()), "angle_max": float(ang.max()),
                    "rotation_deg": float(ang.max() - ang.min()), "unwrapped": unwrapped,
                    "dropped_wrap_averaged": dropped,
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
    with open(os.path.join(out, "manifest.json"), "x") as fh:
        json.dump(manifest, fh, indent=1)
    cmd_root = a.command_archive or a.archive
    def there(path: str) -> str:
        """The path as seen from ``--command-archive``."""
        return os.path.join(cmd_root, os.path.relpath(path, a.archive))
    cubes, angles, psfs, refs = ([there(x) for x in lst] for lst in (cubes, angles, psfs, refs))
    run_dir = there(os.path.join(night_dir, "klip_tpe", f"run_{a.tag}"))
    q = shlex.quote
    lines = ["#!/usr/bin/env bash",
             f"# KLIP-TPE generic run for {a.night} ({a.tag}); made by reduce_nomic tools/klip_tpe_inputs.py",
             "set -euo pipefail",
             'KT="${KT:?set KT to the klip-tpe executable of the pinned clone}"',
             f'RUN={q(run_dir)}',
             '[ ! -e "${RUN:?}" ] || { echo "${RUN} exists"; exit 1; }',
             '"${KT:?}" generic \\',
             "  --cube " + " ".join(q(c) for c in cubes) + " \\",
             "  --angles " + " ".join(q(x) for x in angles) + " \\",
             "  --psf " + " ".join(q(x) for x in psfs) + " \\"]
    if refs:
        lines.append("  --ref-cube " + " ".join(q(x) for x in refs) + " \\")
    head = list(lines)
    lines += ["  --names " + " ".join(names) + " \\",
              "  --pxscale 0.0179 --lam 11.1e-6 --diam 8.4 \\",
              f"  --ann-edges {' '.join(a.ann_edges)} --n-iter {' '.join(a.n_iter)} --n-init {' '.join(a.n_init)} \\",
              f"  --n-remeasure {a.n_remeasure} --n-top 6 --n-valid 8 --contrast0 {a.contrast0} \\",
              f'  --workers "${{WORKERS:-{a.workers}}}" --seed 1 --verify {display} \\',
              '  --run-dir "${RUN:?}"']
    smoke_dir = there(os.path.join(night_dir, "klip_tpe", f"smoke_{a.tag}"))
    smoke = [ln.replace(f"RUN={q(run_dir)}", f"RUN={q(smoke_dir)}") for ln in head]
    smoke[1] = f"# KLIP-TPE smoke test for {a.night} ({a.tag}): 10 evaluations on the first annulus; its numbers mean nothing"
    smoke += ["  --names " + " ".join(names) + " \\",
              "  --pxscale 0.0179 --lam 11.1e-6 --diam 8.4 \\",
              f"  --ann-edges {' '.join(a.ann_edges[:2])} --n-iter 10 --n-init 5 --n-top 1 --n-valid 1 --contrast0 {a.contrast0} \\",
              f'  --workers "${{WORKERS:-16}}" --seed 1 {display} \\',
              '  --run-dir "${RUN:?}"']
    for name, body in ((f"run_{a.tag}.sh", lines), (f"run_{a.tag}_smoke.sh", smoke)):
        sh = os.path.join(night_dir, "klip_tpe", name)
        with open(sh, "x") as fh:
            fh.write("\n".join(body) + "\n")
        os.chmod(sh, 0o755)
    print(f"wrote {out}/, run_{a.tag}.sh and run_{a.tag}_smoke.sh in {os.path.join(night_dir, 'klip_tpe')}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
