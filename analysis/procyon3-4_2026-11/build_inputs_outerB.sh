#!/usr/bin/env bash
# Build the inputs of the joint Procyon_3 + 4 KLIP-TPE run that extends adi3ann5kn (0-2.5") out
# past Procyon B: three annuli from 2.5 to 5.5", with Procyon B as the known source.
#
# usage: ARCHIVE="$HOME/mnt/NOMIC" COMMAND_ARCHIVE=/mnt/RAID64TB/NOMIC PY="$HOME/venv/bin/python" \
#        bash analysis/procyon3-4_2026-11/build_inputs_outerB.sh
#
# Kevin, 2026-11-25 22:26 MST: "let's also run an extension of the 3+4 run that would cover out to
# Procyon B". Procyon B sits at 4.910", PA 333.0 deg on 2021-12-20 (procyon_b_position.py, Bond et
# al. 2015's orbit, which reproduces their HST positions to 0.004" and 0.09 deg), 274.3 px from
# Procyon A. The run is adi3ann5kn's set-up carried outward:
#   * the same four partitions (P3_left_A, P3_left_B, P4_left_A, P4_left_B), every binned frame,
#     bin, filter, angsep and anglemax linked across the partitions, 16 trials in flight, 10,000
#     evaluations per annulus of which 500 are random warm-up draws;
#   * annuli 2.5-3.5-4.5-5.5" (edges 139.66, 195.53, 251.40 and 307.26 px), each 1.0" wide like
#     adi3ann5kn's outer annulus, beginning where adi3ann5kn ends. Procyon B lies in the third,
#     22.9 px (1.5 FWHM) inside its inner edge and 33.0 px (2.1 FWHM) inside its outer edge;
#   * every annulus' injection contrast calibrated;
#   * --known 4.910 333.0: the search's and the verify's noise rings leave Procyon B out, no
#     injection lands within 1.5 FWHM of it, and the verify measures it as candidate 1;
#   * a 660 px cut of every binned frame, kept with the NaN past the detector's edge (--keep-nan),
#     and --min-coverage 0.5: a source is injected, and a noise aperture samples, only where at least
#     half of the frames hold data. 330 px is the outer edge plus 22.7 px, the margin adi3ann5kn's
#     320 px cut left past its outer edge (20.3 px). This needs klip-tpe bf99474 or later
#     (~/venvs/klip-tpe-bf99474); klip-tpe aa0702b (~/venvs/klip-tpe-vega) drops every frame with a
#     NaN within the cut.
# Coverage (position_coverage.py on the port's binned cubes, every frame): Procyon B's position has
# data in all of P3_left_A's and P4_left_A's frames, 59 % of P3_left_B's and 60 % of P4_left_B's,
# 80 % of all 1,104; on the circle through it 49 % of the position angles (PA 197.5 to 13.0 deg)
# reach the 0.5 coverage, against 84 % at 3.5", 65 % at 4.0" and 46 % at 5.5".
#
# Parameters
# ----------
# ARCHIVE : environment variable, path
#     NOMIC archive root where this runs (~/mnt/NOMIC in the device sandbox).
# COMMAND_ARCHIVE : environment variable, path, default $ARCHIVE
#     The same root as seen where KLIP-TPE runs (/mnt/RAID64TB/NOMIC on patrick).
# PY : environment variable, path, default python3
#     Python with numpy and astropy.
#
# Outputs
# -------
# $ARCHIVE/Procyon_20211219-20211220/klip_tpe/inputs_outerB/ : folder
#     Cut cubes, angles, PSF templates of both nights, manifest.json and klip_tpe_linked.py
#     (tools/klip_tpe_inputs.py).
# $ARCHIVE/Procyon_20211219-20211220/klip_tpe/run_outerB.sh, run_outerB_smoke.sh and resume_outerB.sh : files
#     The KLIP-TPE command, its 10-evaluation smoke test on the first annulus, and its resume.
set -euo pipefail

ARCHIVE="${ARCHIVE:?set ARCHIVE to the NOMIC archive root}"
COMMAND_ARCHIVE="${COMMAND_ARCHIVE:-${ARCHIVE}}"
PY="${PY:-python3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:?}")" && pwd)"
TOOL="${HERE:?}/../../tools/klip_tpe_inputs.py"

"${PY:?}" "${TOOL:?}" "${ARCHIVE:?}" Procyon_20211219 Procyon_20211220 \
    --out-folder Procyon_20211219-20211220 --labels P3 P4 \
    --tag outerB --sides left --half 330 --keep-nan --min-coverage 0.5 \
    --ann-edges 139.66 195.53 251.40 307.26 \
    --n-iter 10000 10000 10000 --n-init 500 500 500 \
    --link bin filter angsep anglemax --batch 16 --known 4.910 333.0 \
    --command-archive "${COMMAND_ARCHIVE:?}"
