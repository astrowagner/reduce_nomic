#!/usr/bin/env bash
# Move the Vega products superseded on 2026-11-20 (late) into the archive's _to_delete/ folder.
#
# usage: ARCHIVE=/mnt/RAID64TB/NOMIC bash analysis/vega_2026-11/supersede_2026-11-20.sh
#
# Two problems found after the first Vega KLIP-TPE runs make these products obsolete:
#
# * Vega 2022-05-20 left was reduced before the split-stage fix (359124c), which dropped chunk 6
#   of chop B (500 of 5,490 frames; the chop-B detection there was a bad-pixel artifact at
#   (1002, 497)). Its products move aside so that the port reruns it; the batch and the port
#   refuse to write over existing products.
# * Every KLIP-TPE input set built from 2022 data has NaN rows: the star sat 78-110 px from the
#   detector's edge in y, so the 240 px cut reaches the padding, and KLIP-TPE's high-pass spreads
#   one NaN over the whole frame (the 2022-05-21 ADI run stopped with "every frame was dropped as
#   empty after filtering and binning"). Those inputs, their run scripts and the failed run move
#   aside, to be rebuilt by tools/klip_tpe_inputs.py with --half 80, which also leaves out the
#   frames still holding NaN. The 2021-02-27 and 2022-05-15 ADI sets have no NaN and stay; their
#   RDI sets use 2022 reference frames and move.
#
# Each path moves to ``$ARCHIVE/_to_delete/<folder>/<same relative path>``, nothing is deleted,
# and a path already moved is reported and skipped, so the script can be run again. A destination
# that already exists stops the script before anything is overwritten.
#
# Parameters
# ----------
# ARCHIVE : environment variable, path
#     NOMIC archive root (/mnt/RAID64TB/NOMIC on patrick, ~/mnt/NOMIC in the device sandbox).
#
# Outputs
# -------
# $ARCHIVE/_to_delete/superseded_split_2026-11-20/Vega_20220520/reduced_py/ : files
#     The 27 Vega_20220520_left_* products and logs of the run before the split-stage fix.
# $ARCHIVE/_to_delete/klip_tpe_inputs_superseded_2026-11-20d/<night>/klip_tpe/ : folders and files
#     The inputs_<tag>/ folders, run_<tag>.sh and run_<tag>_smoke.sh of 2022-05-20 (adi, rdi),
#     2022-05-21 (adi, with the failed run_adi/), 2022-05-22 (adi), 2022-06-12 (adi), 2021-02-27
#     (rdi) and 2022-05-15 (rdi).
# stdout : text
#     One line per path moved or already gone.
set -euo pipefail

ARCHIVE="${ARCHIVE:?set ARCHIVE to the NOMIC archive root}"
[ -d "${ARCHIVE:?}/_to_delete" ] || { echo "${ARCHIVE}/_to_delete is not a folder"; exit 1; }

move() {
    # Move ARCHIVE/$1 to ARCHIVE/_to_delete/$2/$1, unless it is already gone.
    local rel="${1:?}" folder="${2:?}"
    local src="${ARCHIVE:?}/${rel:?}"
    local dst="${ARCHIVE:?}/_to_delete/${folder:?}/${rel:?}"
    if [ ! -e "${src:?}" ]; then
        echo "already gone: ${rel}"
        return 0
    fi
    if [ -e "${dst:?}" ]; then
        echo "destination exists, stopping: ${dst}"
        exit 1
    fi
    mkdir -p "$(dirname "${dst:?}")"
    mv "${src:?}" "${dst:?}"
    echo "moved ${rel} -> _to_delete/${folder}/${rel}"
}

# 1. Vega 2022-05-20 left: every product and log of the run made before the split-stage fix.
SPLIT=superseded_split_2026-11-20
RED=Vega_20220520/reduced_py
while IFS= read -r name; do
    move "${RED:?}/${name:?}" "${SPLIT:?}"
done < <(ls "${ARCHIVE:?}/${RED:?}" | grep '^Vega_20220520_left_' || true)

# 2. KLIP-TPE inputs cut through the NaN padding, their scripts, and the failed 2022-05-21 run.
INPUTS=klip_tpe_inputs_superseded_2026-11-20d
for set in Vega_20220520:adi Vega_20220520:rdi Vega_20220521:adi Vega_20220522:adi Vega_20220612:adi \
           Vega_20210227:rdi Vega_20220515:rdi; do
    night="${set%%:*}"
    tag="${set##*:}"
    for item in "inputs_${tag}" "run_${tag}.sh" "run_${tag}_smoke.sh"; do
        move "${night:?}/klip_tpe/${item:?}" "${INPUTS:?}"
    done
done
move "Vega_20220521/klip_tpe/run_adi" "${INPUTS:?}"
