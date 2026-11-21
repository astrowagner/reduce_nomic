#!/usr/bin/env bash
# Build the Procyon_4 (Procyon_20211220, left) KLIP-TPE inputs for the outer annulus 2.5-4.0" on frames padded with NaN past the detector's edge.
#
# usage: ARCHIVE="$HOME/mnt/NOMIC" COMMAND_ARCHIVE=/mnt/RAID64TB/NOMIC PY="$HOME/venv/bin/python" \
#        bash analysis/procyon4_2026-11/build_inputs_pad_outer.sh
#
# The detector's edge lies 162-173 px from the star, so a 480 px cut (--half 240) holds NaN in
# every frame; --keep-nan keeps those frames for klip-tpe bf99474 and later, which take NaN as no
# data. One annulus, 139.66-223.46 px (2.5-4.0" at 0.0179"/px; the cut holds its outer edge plus
# half of KLIP-TPE's widest high-pass, 12.5 px), 10,000 evaluations of which 500 are random
# warm-up draws, KLIP-TPE's full production search space, and --min-coverage 0.5: sources and
# noise apertures only where at least half the frames have data (Kevin, 2026-11-21: 13 of 20
# position angles reach that at 4.0", against the 12 of the IDL curve). The injection contrast
# calibrates. It runs beside adi3ann2/adi3ann3, which cover 0-2.5".
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
# $ARCHIVE/Procyon_20211220/klip_tpe/inputs_pad_outer/ : folder
#     Cut cubes with their NaN, angles, PSF templates and manifest.json (tools/klip_tpe_inputs.py).
# $ARCHIVE/Procyon_20211220/klip_tpe/run_pad_outer.sh and run_pad_outer_smoke.sh : files
#     The KLIP-TPE command (needs KT from a klip-tpe at bf99474 or later), and its smoke test.
set -euo pipefail

ARCHIVE="${ARCHIVE:?set ARCHIVE to the NOMIC archive root}"
COMMAND_ARCHIVE="${COMMAND_ARCHIVE:-${ARCHIVE}}"
PY="${PY:-python3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:?}")" && pwd)"
TOOL="${HERE:?}/../../tools/klip_tpe_inputs.py"

"${PY:?}" "${TOOL:?}" "${ARCHIVE:?}" Procyon_20211220 --tag pad_outer --sides left --half 240 --keep-nan \
    --ann-edges 139.66 223.46 --n-iter 10000 --n-init 500 --min-coverage 0.5 \
    --command-archive "${COMMAND_ARCHIVE:?}"
