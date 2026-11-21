#!/usr/bin/env bash
# Build the Procyon_4 (Procyon_20211220, left) KLIP-TPE inputs and run script for three annuli, 0-0.7-1.5-2.5".
#
# usage: ARCHIVE="$HOME/mnt/NOMIC" COMMAND_ARCHIVE=/mnt/RAID64TB/NOMIC PY="$HOME/venv/bin/python" \
#        bash analysis/procyon4_2026-11/build_inputs_adi3ann.sh
#
# Kevin's annuli of 2026-11-21 12:40 MST, replacing the five 0.5" annuli of
# build_inputs_adi10k.sh: edges 0, 39.11, 83.80 and 139.66 px (0, 0.7, 1.5 and 2.5" at
# 0.0179"/px), 10,000 evaluations per annulus of which 500 are random warm-up draws, and
# KLIP-TPE's full production search space (nine parameters per chop and the two drop slots).
# Injections start at 1.5 FWHM (0.42"), so the first annulus is scored over 0.42-0.7". The
# cut, --half 160, is the largest every binned frame is finite within (the detector's edge lies
# 162-173 px from the star); it holds the outer edge, 139.66 px, plus half of KLIP-TPE's widest
# high-pass, 12.5 px. The frames, angles and PSF templates are the same as adi10k's.
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
# $ARCHIVE/Procyon_20211220/klip_tpe/inputs_adi3ann/ : folder
#     Cut cubes, angles, PSF templates and manifest.json (tools/klip_tpe_inputs.py).
# $ARCHIVE/Procyon_20211220/klip_tpe/run_adi3ann.sh and run_adi3ann_smoke.sh : files
#     The KLIP-TPE command, and its 10-evaluation smoke test on the first annulus.
set -euo pipefail

ARCHIVE="${ARCHIVE:?set ARCHIVE to the NOMIC archive root}"
COMMAND_ARCHIVE="${COMMAND_ARCHIVE:-${ARCHIVE}}"
PY="${PY:-python3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:?}")" && pwd)"
TOOL="${HERE:?}/../../tools/klip_tpe_inputs.py"

"${PY:?}" "${TOOL:?}" "${ARCHIVE:?}" Procyon_20211220 --tag adi3ann --sides left --half 160 \
    --ann-edges 0 39.11 83.80 139.66 \
    --n-iter 10000 10000 10000 --n-init 500 500 500 \
    --command-archive "${COMMAND_ARCHIVE:?}"
