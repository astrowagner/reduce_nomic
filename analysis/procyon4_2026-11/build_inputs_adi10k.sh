#!/usr/bin/env bash
# Build the Procyon_4 (Procyon_20211220, left) KLIP-TPE inputs and run script for the 10,000-trial ADI run.
#
# usage: ARCHIVE="$HOME/mnt/NOMIC" COMMAND_ARCHIVE=/mnt/RAID64TB/NOMIC PY="$HOME/venv/bin/python" \
#        bash analysis/procyon4_2026-11/build_inputs_adi10k.sh
#
# 0.5" annuli from 0.2" to 2.5" (edges 11.17, 27.93, 55.87, 83.80, 111.73 and 139.66 px at
# 0.0179"/px), 10,000 evaluations per annulus of which 500 are random warm-up draws, and
# KLIP-TPE's full production search space (nine parameters per chop and the two drop slots). Every
# binned frame of both chops is finite within +-160 px of the star, and the nearest detector edge
# lies 162-173 px away, so the cut stops at --half 160: it holds the outer edge, 139.66 px, plus
# half of KLIP-TPE's widest high-pass, 12.5 px. An annulus out to 3.0" (167.6 px) would need
# frames finite to 180 px, which none is.
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
# $ARCHIVE/Procyon_20211220/klip_tpe/inputs_adi10k/ : folder
#     Cut cubes, angles, PSF templates and manifest.json (tools/klip_tpe_inputs.py).
# $ARCHIVE/Procyon_20211220/klip_tpe/run_adi10k.sh and run_adi10k_smoke.sh : files
#     The KLIP-TPE command, and its 10-evaluation smoke test on the first annulus.
set -euo pipefail

ARCHIVE="${ARCHIVE:?set ARCHIVE to the NOMIC archive root}"
COMMAND_ARCHIVE="${COMMAND_ARCHIVE:-${ARCHIVE}}"
PY="${PY:-python3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:?}")" && pwd)"
TOOL="${HERE:?}/../../tools/klip_tpe_inputs.py"

"${PY:?}" "${TOOL:?}" "${ARCHIVE:?}" Procyon_20211220 --tag adi10k --sides left --half 160 \
    --ann-edges 11.17 27.93 55.87 83.80 111.73 139.66 \
    --n-iter 10000 10000 10000 10000 10000 --n-init 500 500 500 500 500 \
    --command-archive "${COMMAND_ARCHIVE:?}"
