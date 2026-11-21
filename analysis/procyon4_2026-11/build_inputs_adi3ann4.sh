#!/usr/bin/env bash
# Build the Procyon_4 (Procyon_20211220, left) KLIP-TPE inputs for annuli 0-0.7-1.5-2.5", the first at a fixed contrast of 5e-4, with bin, filter, angsep and anglemax linked across the chops.
#
# usage: ARCHIVE="$HOME/mnt/NOMIC" COMMAND_ARCHIVE=/mnt/RAID64TB/NOMIC PY="$HOME/venv/bin/python" \
#        bash analysis/procyon4_2026-11/build_inputs_adi3ann4.sh
#
# The same run as build_inputs_adi3ann3.sh (edges 0, 39.11, 83.80 and 139.66 px; 10,000
# evaluations per annulus of which 500 are random warm-up draws; KLIP-TPE's full production
# search space; the 320 px cut of every binned frame), with two changes (Kevin, 2026-11-21
# 15:42 MST): the first annulus at a fixed contrast of 5e-4 instead of 4e-4, and the high-pass
# width (filter), angsep and anglemax the same for both chops, as bin already was (KLIP-TPE links
# bin and filter by default; tools/klip_tpe_linked.py adds angsep and anglemax). The other two
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
# $ARCHIVE/Procyon_20211220/klip_tpe/inputs_adi3ann4/ : folder
#     Cut cubes, angles, PSF templates, manifest.json and klip_tpe_linked.py
#     (tools/klip_tpe_inputs.py).
# $ARCHIVE/Procyon_20211220/klip_tpe/run_adi3ann4.sh, run_adi3ann4_smoke.sh and resume_adi3ann4.sh : files
#     The KLIP-TPE command, its 10-evaluation smoke test on the first annulus, and its resume.
set -euo pipefail

ARCHIVE="${ARCHIVE:?set ARCHIVE to the NOMIC archive root}"
COMMAND_ARCHIVE="${COMMAND_ARCHIVE:-${ARCHIVE}}"
PY="${PY:-python3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:?}")" && pwd)"
TOOL="${HERE:?}/../../tools/klip_tpe_inputs.py"

"${PY:?}" "${TOOL:?}" "${ARCHIVE:?}" Procyon_20211220 --tag adi3ann4 --sides left --half 160 \
    --ann-edges 0 39.11 83.80 139.66 \
    --n-iter 10000 10000 10000 --n-init 500 500 500 --use-contrast 5e-4 \
    --link bin filter angsep anglemax \
    --command-archive "${COMMAND_ARCHIVE:?}"
