#!/usr/bin/env bash
# Build the joint Procyon_3 + Procyon_4 KLIP-TPE inputs (four partitions) with the adi3ann5 settings.
#
# usage: ARCHIVE="$HOME/mnt/NOMIC" COMMAND_ARCHIVE=/mnt/RAID64TB/NOMIC PY="$HOME/venv/bin/python" \
#        bash analysis/procyon3-4_2026-11/build_inputs_adi3ann5.sh
#
# Kevin, 2026-11-22: "let's do Procyon_3 next, then Procyon_3 and 4 jointly." Both nights' left
# sides, each chop a partition: P3_left_A, P3_left_B (Procyon_20211219, 164 binned frames each)
# and P4_left_A, P4_left_B (Procyon_20211220, 388 each), in the combined folder
# Procyon_20211219-20211220 (IDL's Procyon_3-4). The settings are adi3ann5's: annuli 0-0.7-1.5-2.5"
# (edges 0, 39.11, 83.80 and 139.66 px); 10,000 evaluations per annulus of which 500 are random
# warm-up draws; the first annulus at a fixed contrast of 5e-4, the others calibrated; bin, filter,
# angsep and anglemax linked across all four partitions; 16 trials in flight; the 320 px cut of
# every binned frame. KLIP-TPE's generic adapter takes one bin range for every partition, from the
# smallest (1 to 164 / 8 = 20 frames per bin); on both nights a binned frame holds 100 chop pairs.
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
# $ARCHIVE/Procyon_20211219-20211220/klip_tpe/inputs_adi3ann5/ : folder
#     Cut cubes, angles, PSF templates of both nights, manifest.json and klip_tpe_linked.py
#     (tools/klip_tpe_inputs.py).
# $ARCHIVE/Procyon_20211219-20211220/klip_tpe/run_adi3ann5.sh, run_adi3ann5_smoke.sh and resume_adi3ann5.sh : files
#     The KLIP-TPE command, its 10-evaluation smoke test on the first annulus, and its resume.
set -euo pipefail

ARCHIVE="${ARCHIVE:?set ARCHIVE to the NOMIC archive root}"
COMMAND_ARCHIVE="${COMMAND_ARCHIVE:-${ARCHIVE}}"
PY="${PY:-python3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:?}")" && pwd)"
TOOL="${HERE:?}/../../tools/klip_tpe_inputs.py"

"${PY:?}" "${TOOL:?}" "${ARCHIVE:?}" Procyon_20211219 Procyon_20211220 \
    --out-folder Procyon_20211219-20211220 --labels P3 P4 \
    --tag adi3ann5 --sides left --half 160 \
    --ann-edges 0 39.11 83.80 139.66 \
    --n-iter 10000 10000 10000 --n-init 500 500 500 --use-contrast 5e-4 \
    --link bin filter angsep anglemax --batch 16 \
    --command-archive "${COMMAND_ARCHIVE:?}"
