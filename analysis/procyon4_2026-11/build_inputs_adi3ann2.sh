#!/usr/bin/env bash
# Build the Procyon_4 (Procyon_20211220, left) KLIP-TPE inputs for annuli 0-0.7-1.5-2.5", the first at a fixed contrast of 2e-4.
#
# usage: ARCHIVE="$HOME/mnt/NOMIC" COMMAND_ARCHIVE=/mnt/RAID64TB/NOMIC PY="$HOME/venv/bin/python" \
#        bash analysis/procyon4_2026-11/build_inputs_adi3ann2.sh
#
# The same run as build_inputs_adi3ann.sh (edges 0, 39.11, 83.80 and 139.66 px; 10,000
# evaluations per annulus of which 500 are random warm-up draws; KLIP-TPE's full production
# search space; the 320 px cut of every binned frame), except that the first annulus injects at a
# fixed contrast of 2e-4 instead of a calibrated one (Kevin, 2026-11-21 12:56 MST). The other two
# annuli calibrate as before.
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
# $ARCHIVE/Procyon_20211220/klip_tpe/inputs_adi3ann2/ : folder
#     Cut cubes, angles, PSF templates and manifest.json (tools/klip_tpe_inputs.py).
# $ARCHIVE/Procyon_20211220/klip_tpe/run_adi3ann2.sh and run_adi3ann2_smoke.sh : files
#     The KLIP-TPE command, and its 10-evaluation smoke test on the first annulus.
set -euo pipefail

ARCHIVE="${ARCHIVE:?set ARCHIVE to the NOMIC archive root}"
COMMAND_ARCHIVE="${COMMAND_ARCHIVE:-${ARCHIVE}}"
PY="${PY:-python3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:?}")" && pwd)"
TOOL="${HERE:?}/../../tools/klip_tpe_inputs.py"

"${PY:?}" "${TOOL:?}" "${ARCHIVE:?}" Procyon_20211220 --tag adi3ann2 --sides left --half 160 \
    --ann-edges 0 39.11 83.80 139.66 \
    --n-iter 10000 10000 10000 --n-init 500 500 500 --use-contrast 2e-4 \
    --command-archive "${COMMAND_ARCHIVE:?}"
