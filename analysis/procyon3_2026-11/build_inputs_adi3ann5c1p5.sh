#!/usr/bin/env bash
# Build the Procyon_3 (Procyon_20211219, left) KLIP-TPE inputs: adi3ann5 with annulus 1 at 1.5e-3.
#
# usage: ARCHIVE="$HOME/mnt/NOMIC" COMMAND_ARCHIVE=/mnt/RAID64TB/NOMIC PY="$HOME/venv/bin/python" \
#        bash analysis/procyon3_2026-11/build_inputs_adi3ann5c1p5.sh
#
# The same run as build_inputs_adi3ann5.sh but for the first annulus' fixed contrast, 1.5e-3 in
# place of 5e-4 (Kevin, 2026-11-23 22:55 MST: "yes, increase contrast please"). At 5e-4 the
# adi3ann5 search's best score in annulus 1 was 2.8 after 4,464 evaluations (Procyon_4 alone
# reached 15 at that contrast): with 45 deg of rotation and about 40 % of Procyon_4's exposure,
# Procyon_3 alone barely sees a 5e-4 source at 0.42". Otherwise: annuli 0-0.7-1.5-2.5" (edges 0,
# 39.11, 83.80 and 139.66 px); 10,000 evaluations per annulus of which 500 are random warm-up
# draws; annuli 2 and 3 calibrated; bin, filter, angsep and anglemax linked across the chops; 16
# trials in flight; the 320 px cut of every binned frame. The card and the worker budget are chosen
# when the run starts (arguments and WORKERS).
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
# $ARCHIVE/Procyon_20211219/klip_tpe/inputs_adi3ann5c1p5/ : folder
#     Cut cubes, angles, PSF templates, manifest.json and klip_tpe_linked.py
#     (tools/klip_tpe_inputs.py).
# $ARCHIVE/Procyon_20211219/klip_tpe/run_adi3ann5c1p5.sh, run_adi3ann5c1p5_smoke.sh and resume_adi3ann5c1p5.sh : files
#     The KLIP-TPE command, its 10-evaluation smoke test on the first annulus, and its resume.
set -euo pipefail

ARCHIVE="${ARCHIVE:?set ARCHIVE to the NOMIC archive root}"
COMMAND_ARCHIVE="${COMMAND_ARCHIVE:-${ARCHIVE}}"
PY="${PY:-python3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:?}")" && pwd)"
TOOL="${HERE:?}/../../tools/klip_tpe_inputs.py"

"${PY:?}" "${TOOL:?}" "${ARCHIVE:?}" Procyon_20211219 --tag adi3ann5c1p5 --sides left --half 160 \
    --ann-edges 0 39.11 83.80 139.66 \
    --n-iter 10000 10000 10000 --n-init 500 500 500 --use-contrast 1.5e-3 \
    --link bin filter angsep anglemax --batch 16 \
    --command-archive "${COMMAND_ARCHIVE:?}"
