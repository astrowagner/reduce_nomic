#!/usr/bin/env bash
# Move the Vega 2022-05-21 ADI inputs built at 07:47 UTC on 2026-11-21 into the archive's _to_delete/ folder.
#
# usage: ARCHIVE=/mnt/RAID64TB/NOMIC bash analysis/vega_2026-11/supersede_2026-11-21.sh
#
# That input set (built from the reruns of 2022-05-21 left with 359124c) holds 2 binned frames of
# left chop B, from chunk 88, with the star 40 px off the center: the chunk's Gaussian center fit
# failed and the fallback estimator put the star at y = 443.9. tools/klip_tpe_inputs.py now leaves
# such frames out (--max-frame-offset), and the set is rebuilt with it. Nothing is deleted; a path
# already moved is reported and skipped, and an existing destination stops the script.
#
# Parameters
# ----------
# ARCHIVE : environment variable, path
#     NOMIC archive root (/mnt/RAID64TB/NOMIC on patrick, ~/mnt/NOMIC in the device sandbox).
#
# Outputs
# -------
# $ARCHIVE/_to_delete/klip_tpe_inputs_superseded_2026-11-21/Vega_20220521/klip_tpe/ : folder and files
#     inputs_adi/, run_adi.sh and run_adi_smoke.sh.
# stdout : text
#     One line per path moved or already gone.
set -euo pipefail

ARCHIVE="${ARCHIVE:?set ARCHIVE to the NOMIC archive root}"
[ -d "${ARCHIVE:?}/_to_delete" ] || { echo "${ARCHIVE}/_to_delete is not a folder"; exit 1; }
FOLDER=klip_tpe_inputs_superseded_2026-11-21

for rel in Vega_20220521/klip_tpe/inputs_adi Vega_20220521/klip_tpe/run_adi.sh Vega_20220521/klip_tpe/run_adi_smoke.sh; do
    src="${ARCHIVE:?}/${rel:?}"
    dst="${ARCHIVE:?}/_to_delete/${FOLDER:?}/${rel:?}"
    if [ ! -e "${src:?}" ]; then
        echo "already gone: ${rel}"
        continue
    fi
    if [ -e "${dst:?}" ]; then
        echo "destination exists, stopping: ${dst}"
        exit 1
    fi
    if [ -e "${ARCHIVE:?}/Vega_20220521/klip_tpe/run_adi" ]; then
        echo "Vega_20220521/klip_tpe/run_adi exists: a run used these inputs; not moving them"
        exit 1
    fi
    mkdir -p "$(dirname "${dst:?}")"
    mv "${src:?}" "${dst:?}"
    echo "moved ${rel} -> _to_delete/${FOLDER}/${rel}"
done
