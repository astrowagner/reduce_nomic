"""Prepare one night's port products as KLIP-TPE generic inputs, and the command that runs it.

usage: python tools/klip_tpe_inputs.py ARCHIVE NIGHT --tag TAG [--sides left right] [--half 120]
       [--command-archive PATH]
       [--psf-half 30] [--product all|clean] [--ref NIGHT ...] [--ann-edges 11 34 67]
       [--n-iter 400 400] [--n-init 60 60] [--n-remeasure 3] [--contrast0 1e-3] [--workers auto]

KLIP-TPE's generic adapter (``klip-tpe generic``) takes one registered cube, its derotation angles
and a PSF template per partition, and optionally a reference cube (RDI). For each side and chop of
``NIGHT`` this writes, under ``<ARCHIVE>/<NIGHT>/klip_tpe/inputs_<TAG>/``:

* the binned frames (``--product all``: every binned frame, ``<pre>_cube<n>_align_cen_bin.fits``,
  for KLIP-TPE's own frame selection; ``clean``: the IDL clean selection) cut to (2 half)² pixels
  with the star at the array center, so that KLIP-TPE's default center applies;
* their parallactic angles (KLIP-TPE derotates counterclockwise by angle + truenorth, as the IDL
  predecessor does with ``rot(frame, -angle - truenorth)``);
* the median PSF of the IDL-clean frames cut to (2 psf_half)² pixels as the injection template:
  KLIP-TPE normalizes it to unit total flux, so contrasts are flux ratios to the star's flux
  within that box (the frames are peak-scaled, so every frame shares the unit);
* with ``--ref``: the same side and chop of the given nights' IDL-clean frames, cut the same way
  and stacked, as the reference library.

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
--command-archive : str, default ARCHIVE
    Archive root to write into the command and the manifest, when the archive is mounted
    elsewhere where this runs (the device sandbox sees /mnt/RAID64TB/NOMIC as ~/mnt/NOMIC).
--ann-edges, --n-iter, --n-init, --n-remeasure, --contrast0, --workers
    Written into the command (KLIP-TPE options of the same names; annulus edges in pixels).

Returns
-------
inputs_<TAG>/ : files
    ``<pre>_cube<n>_<product>_c<2 half>.fits`` (float32, shape (n_frames, 2 half, 2 half)),
    ``<pre>_angles<n>_<product>.fits`` (degrees), ``<pre>_cube<n>_psf<2 psf_half>.fits``, with
    ``--ref`` ``<pre>_cube<n>_ref_c<2 half>.fits``, and ``manifest.json`` (sources with their
    SHA-256, cuts, frame counts, angle ranges).
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
from typing import Dict, List, Optional, Sequence

import numpy as np
from astropy.io import fits


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
    a = ap.parse_args(argv)
    night_dir = os.path.join(a.archive, a.night)
    red = os.path.join(night_dir, "reduced_py")
    out = os.path.join(night_dir, "klip_tpe", f"inputs_{a.tag}")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; choose another --tag")
    os.makedirs(out)
    tag = "_clean" if a.product == "clean" else ""
    manifest: Dict = {"night": a.night, "tag": a.tag, "product": a.product, "half": a.half,
                      "psf_half": a.psf_half, "ref_nights": a.ref, "partitions": []}
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
            rc, used = None, []
            if a.ref:
                stack = []
                for rn in a.ref:
                    rsrc = os.path.join(a.archive, rn, "reduced_py", f"{rn}_{side}_cube{cn}_align_cen_clean_bin.fits")
                    if os.path.exists(rsrc):
                        stack.append(cut_frames(rsrc, a.half))
                        used.append(rn)
                if not stack:
                    print(f"{pre} cube{cn}: no reference frames from {a.ref}; skipped")
                    continue
                rc = np.concatenate(stack)
            cube = cut_frames(src, a.half)
            ang = np.asarray(fits.getdata(asrc), dtype=np.float64).ravel()
            if len(ang) != cube.shape[0]:
                raise SystemExit(f"{asrc}: {len(ang)} angles for {cube.shape[0]} frames")
            psf = cut_frames(psrc, a.psf_half)
            c_out = os.path.join(out, f"{pre}_cube{cn}_{a.product}_c{2 * a.half}.fits")
            a_out = os.path.join(out, f"{pre}_angles{cn}_{a.product}.fits")
            p_out = os.path.join(out, f"{pre}_cube{cn}_psf{2 * a.psf_half}.fits")
            fits.PrimaryHDU(cube).writeto(c_out)
            fits.PrimaryHDU(ang).writeto(a_out)
            fits.PrimaryHDU(psf).writeto(p_out)
            part = {"name": f"{side}_{'AB'[cn - 1]}", "cube": c_out, "angles": a_out, "psf": p_out,
                    "frames": int(cube.shape[0]), "angle_min": float(ang.min()), "angle_max": float(ang.max()),
                    "sources": {os.path.basename(p): sha256_of(p) for p in (src, asrc, psrc)}}
            if rc is not None:
                r_out = os.path.join(out, f"{pre}_cube{cn}_ref_c{2 * a.half}.fits")
                fits.PrimaryHDU(rc).writeto(r_out)
                part.update(ref=r_out, ref_frames=int(rc.shape[0]), ref_nights_used=used)
                refs.append(r_out)
            manifest["partitions"].append(part)
            cubes.append(c_out); angles.append(a_out); psfs.append(p_out); names.append(part["name"])
            print(f"{pre} cube{cn}: {cube.shape[0]} frames, angles {ang.min():.1f}..{ang.max():.1f} deg"
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
              f'  --workers "${{WORKERS:-{a.workers}}}" --seed 1 --verify --no-display \\',
              '  --run-dir "${RUN:?}"']
    smoke_dir = there(os.path.join(night_dir, "klip_tpe", f"smoke_{a.tag}"))
    smoke = [ln.replace(f"RUN={q(run_dir)}", f"RUN={q(smoke_dir)}") for ln in head]
    smoke[1] = f"# KLIP-TPE smoke test for {a.night} ({a.tag}): 10 evaluations on the first annulus; its numbers mean nothing"
    smoke += ["  --names " + " ".join(names) + " \\",
              "  --pxscale 0.0179 --lam 11.1e-6 --diam 8.4 \\",
              f"  --ann-edges {' '.join(a.ann_edges[:2])} --n-iter 10 --n-init 5 --n-top 1 --n-valid 1 --contrast0 {a.contrast0} \\",
              '  --workers "${WORKERS:-16}" --seed 1 --no-display \\',
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
