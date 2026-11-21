#!/usr/bin/env bash
# Move one superseded Procyon_4 KLIP-TPE set (inputs, scripts and runs of one tag) into the archive's _to_delete/ folder.
#
# usage: ARCHIVE=/mnt/RAID64TB/NOMIC bash analysis/procyon4_2026-11/supersede_set.sh TAG
#
# Moves Procyon_20211220/klip_tpe/inputs_TAG/, run_TAG.sh, run_TAG_smoke.sh, resume_TAG.sh,
# run_TAG/ and smoke_TAG/ (those that exist) into _to_delete/klip_tpe_superseded_2026-11-21_procyon4/, the
# folder of the earlier adi10k and adi3ann sets. Nothing is deleted; a path already moved is
# reported and skipped, an existing destination stops the script, and nothing is moved while a
# run's heartbeat is less than 10 minutes old.
#
# Parameters
# ----------
# TAG : argument, str
#     The set's tag, e.g. adi3ann2 (letters, digits and underscores).
# ARCHIVE : environment variable, path
#     NOMIC archive root (/mnt/RAID64TB/NOMIC on patrick, ~/mnt/NOMIC in the device sandbox).
#
# Outputs
# -------
# $ARCHIVE/_to_delete/klip_tpe_superseded_2026-11-21_procyon4/Procyon_20211220/klip_tpe/ : folder
#     The moved paths.
# stdout : text
#     One line per path moved or already gone.
set -euo pipefail

TAG="${1:?give the tag, e.g. adi3ann2}"
[[ "${TAG}" =~ ^[A-Za-z0-9_]+$ ]] || { echo "tag ${TAG}: letters, digits and underscores only"; exit 1; }
ARCHIVE="${ARCHIVE:?set ARCHIVE to the NOMIC archive root}"
[ -d "${ARCHIVE:?}/_to_delete" ] || { echo "${ARCHIVE}/_to_delete is not a folder"; exit 1; }
FOLDER=klip_tpe_superseded_2026-11-21_procyon4
KT_DIR=Procyon_20211220/klip_tpe

for run in "run_${TAG:?}" "smoke_${TAG:?}"; do
    HB="${ARCHIVE:?}/${KT_DIR:?}/${run:?}/heartbeat.json"
    if [ -f "${HB:?}" ]; then
        age=$(( $(date +%s) - $(stat -c %Y "${HB:?}") ))
        if [ "${age}" -lt 600 ]; then
            echo "${run}'s heartbeat is ${age} s old; is the run still going? stopping"
            exit 1
        fi
    fi
done

mkdir -p "${ARCHIVE:?}/_to_delete/${FOLDER:?}/${KT_DIR:?}"
for name in "inputs_${TAG:?}" "run_${TAG:?}.sh" "run_${TAG:?}_smoke.sh" "resume_${TAG:?}.sh" "run_${TAG:?}" \
            "smoke_${TAG:?}"; do
    src="${ARCHIVE:?}/${KT_DIR:?}/${name:?}"
    dst="${ARCHIVE:?}/_to_delete/${FOLDER:?}/${KT_DIR:?}/${name:?}"
    if [ ! -e "${src:?}" ]; then
        echo "already gone or never made: ${KT_DIR}/${name}"
        continue
    fi
    if [ -e "${dst:?}" ]; then
        echo "destination exists, stopping: ${dst}"
        exit 1
    fi
    mv "${src:?}" "${dst:?}"
    echo "moved ${KT_DIR}/${name} -> _to_delete/${FOLDER}/${KT_DIR}/${name}"
done
