#!/usr/bin/env bash
# Move the superseded Procyon_4 adi10k KLIP-TPE set (five 0.5" annuli) into the archive's _to_delete/ folder.
#
# usage: ARCHIVE=/mnt/RAID64TB/NOMIC bash analysis/procyon4_2026-11/supersede_adi10k_2026-11-21.sh
#
# Kevin stopped the adi10k run at 19:42:47 UTC on 2026-11-21 (annulus 1, warm-up evaluation 294
# of 10,000) and asked for annuli of 0-0.7, 0.7-1.5 and 1.5-2.5" instead, built as adi3ann by
# build_inputs_adi3ann.sh. This moves the adi10k inputs, its two scripts and the stopped run.
# Nothing is deleted; a path already moved is reported and skipped, an existing destination
# stops the script, and the run is not moved while its heartbeat is less than 10 minutes old.
#
# Parameters
# ----------
# ARCHIVE : environment variable, path
#     NOMIC archive root (/mnt/RAID64TB/NOMIC on patrick, ~/mnt/NOMIC in the device sandbox).
#
# Outputs
# -------
# $ARCHIVE/_to_delete/klip_tpe_superseded_2026-11-21_procyon4/Procyon_20211220/klip_tpe/ : folder
#     inputs_adi10k/, run_adi10k.sh, run_adi10k_smoke.sh and run_adi10k/.
# stdout : text
#     One line per path moved or already gone.
set -euo pipefail

ARCHIVE="${ARCHIVE:?set ARCHIVE to the NOMIC archive root}"
[ -d "${ARCHIVE:?}/_to_delete" ] || { echo "${ARCHIVE}/_to_delete is not a folder"; exit 1; }
FOLDER=klip_tpe_superseded_2026-11-21_procyon4
KT_DIR=Procyon_20211220/klip_tpe

HB="${ARCHIVE:?}/${KT_DIR:?}/run_adi10k/heartbeat.json"
if [ -f "${HB:?}" ]; then
    age=$(( $(date +%s) - $(stat -c %Y "${HB:?}") ))
    if [ "${age}" -lt 600 ]; then
        echo "run_adi10k's heartbeat is ${age} s old; is the run still going? stopping"
        exit 1
    fi
fi

mkdir -p "${ARCHIVE:?}/_to_delete/${FOLDER:?}/${KT_DIR:?}"
for name in inputs_adi10k run_adi10k.sh run_adi10k_smoke.sh run_adi10k; do
    src="${ARCHIVE:?}/${KT_DIR:?}/${name:?}"
    dst="${ARCHIVE:?}/_to_delete/${FOLDER:?}/${KT_DIR:?}/${name:?}"
    if [ ! -e "${src:?}" ]; then
        echo "already gone: ${KT_DIR}/${name}"
        continue
    fi
    if [ -e "${dst:?}" ]; then
        echo "destination exists, stopping: ${dst}"
        exit 1
    fi
    mv "${src:?}" "${dst:?}"
    echo "moved ${KT_DIR}/${name} -> _to_delete/${FOLDER}/${KT_DIR}/${name}"
done
