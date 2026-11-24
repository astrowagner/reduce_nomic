#!/usr/bin/env bash
# Build the Procyon_4 (Procyon_20211220, left) KLIP-TPE inputs: adi3ann5 with the suspected source
# at 0.456", PA 156 deg as a known source.
#
# usage: ARCHIVE="$HOME/mnt/NOMIC" COMMAND_ARCHIVE=/mnt/RAID64TB/NOMIC PY="$HOME/venv/bin/python" \
#        bash analysis/procyon4_2026-11/build_inputs_adi3ann5kn.sh
#
# Kevin, 2026-11-24 09:31 MST: "run these three again (including the 3+4 version) with a suspected
# source included as known - at 0.456", 156 deg PA E of N". The same run as
# build_inputs_adi3ann5.sh (annulus 1 at a fixed 5e-4; annuli 2 and 3 calibrated; annuli
# 0-0.7-1.5-2.5", edges 0, 39.11, 83.80 and 139.66 px; 10,000 evaluations per annulus of which 500
# are random warm-up draws; bin, filter, angsep and anglemax linked across the chops; 16 trials in
# flight; the 320 px cut of every binned frame) plus ``--known 0.456 156``: the search's and the
# verify's noise rings leave the source out, no injection lands within 1.5 FWHM of it, and the
# verify measures it as a candidate (tools/klip_tpe_inputs.py, --known). The source sits at
# 25.5 px, inside annulus 1. The card and the worker budget are chosen when the run starts
# (arguments and WORKERS).
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
# $ARCHIVE/Procyon_20211220/klip_tpe/inputs_adi3ann5kn/ : folder
#     Cut cubes, angles, PSF templates, manifest.json and klip_tpe_linked.py
#     (tools/klip_tpe_inputs.py).
# $ARCHIVE/Procyon_20211220/klip_tpe/run_adi3ann5kn.sh, run_adi3ann5kn_smoke.sh and resume_adi3ann5kn.sh : files
#     The KLIP-TPE command, its 10-evaluation smoke test on the first annulus, and its resume.
set -euo pipefail

ARCHIVE="${ARCHIVE:?set ARCHIVE to the NOMIC archive root}"
COMMAND_ARCHIVE="${COMMAND_ARCHIVE:-${ARCHIVE}}"
PY="${PY:-python3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:?}")" && pwd)"
TOOL="${HERE:?}/../../tools/klip_tpe_inputs.py"

"${PY:?}" "${TOOL:?}" "${ARCHIVE:?}" Procyon_20211220 --tag adi3ann5kn --sides left --half 160 \
    --ann-edges 0 39.11 83.80 139.66 \
    --n-iter 10000 10000 10000 --n-init 500 500 500 --use-contrast 5e-4 \
    --link bin filter angsep anglemax --batch 16 --known 0.456 156 \
    --command-archive "${COMMAND_ARCHIVE:?}"
