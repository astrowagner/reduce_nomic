#!/usr/bin/env bash
# Run the three known-source KLIP-TPE runs one after another: Procyon_3 alone (adi3ann5c1p5kn),
# Procyon_4 alone (adi3ann5kn) and the joint Procyon_3 + 4 run (adi3ann5kn), each with the
# suspected source at 0.456", PA 156 deg as a known source.
#
# usage: KT=~/venvs/klip-tpe-vega/bin/klip-tpe bash run_known_sequence_kn.sh
#
# Kevin, 2026-11-24 09:31 MST: "Let's run them in sequence, with max resources on CPU and using
# Alice (GPU 0) on auto". Each run's own script (run_<tag>.sh, made by tools/klip_tpe_inputs.py
# with --known 0.456 156; build_inputs_adi3ann5c1p5kn.sh and the two build_inputs_adi3ann5kn.sh)
# runs with WORKERS and the card options. Before each run the sequence:
#
# * stops if the file STOP_AFTER_CURRENT is in this script's folder (touch it to end the sequence
#   once the run in progress has finished; move it away to let a later sequence through);
# * skips a run that has finished (its final_results.json is there);
# * stops at a run that was started but has not finished (its folder is there without
#   final_results.json): resume it with resume_<tag>.sh, then start the sequence again.
#
# A run that fails, or ends without final_results.json, stops the sequence. Each run's terminal
# output also goes to run_<tag>.log beside its script.
#
# Parameters
# ----------
# KT : environment variable, path
#     The klip-tpe executable of the pinned clone's virtual environment
#     (~/venvs/klip-tpe-vega/bin/klip-tpe, the earlier Procyon runs' klip-tpe).
# ARCHIVE : environment variable, default /mnt/RAID64TB/NOMIC
#     NOMIC archive root.
# WORKERS : environment variable, default max
#     klip-tpe's worker budget: max (or all) for every logical CPU, auto for one per physical core,
#     or a number (the earlier runs used 96).
# GPU_DEVICE : environment variable, default GPU-eb5590c8-23aa-43e8-d3ca-1225db70ca0e
#     The card, by the UUID nvidia-smi -L shows; the default is Alice (GPU 0).
# GPU_SHARE : environment variable, default auto
#     klip-tpe's --gpu-share.
# GPU_WORKERS : environment variable, default 6
#     klip-tpe's --gpu-workers.
#
# Returns
# -------
# run_<tag>/ : folders
#     The three runs, in each night's or the joint folder's klip_tpe/.
# run_<tag>.log : files
#     Each run's terminal output, beside its script.
set -euo pipefail
KT="${KT:?set KT to the klip-tpe executable of the pinned clone}"
export KT
ARCHIVE="${ARCHIVE:-/mnt/RAID64TB/NOMIC}"
export WORKERS="${WORKERS:-max}"
GPU_DEVICE="${GPU_DEVICE:-GPU-eb5590c8-23aa-43e8-d3ca-1225db70ca0e}"
GPU_SHARE="${GPU_SHARE:-auto}"
GPU_WORKERS="${GPU_WORKERS:-6}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:?}")" && pwd)"
STOP="${HERE:?}/STOP_AFTER_CURRENT"
RUNS=("Procyon_20211219 adi3ann5c1p5kn" "Procyon_20211220 adi3ann5kn" "Procyon_20211219-20211220 adi3ann5kn")
[ -x "${KT:?}" ] || { echo "${KT} is not an executable"; exit 1; }
for r in "${RUNS[@]}"; do
  read -r FOLDER TAG <<< "${r}"
  [ -f "${ARCHIVE:?}/${FOLDER:?}/klip_tpe/run_${TAG:?}.sh" ] || { echo "no run_${TAG}.sh in ${FOLDER}/klip_tpe"; exit 1; }
done
for r in "${RUNS[@]}"; do
  read -r FOLDER TAG <<< "${r}"
  D="${ARCHIVE:?}/${FOLDER:?}/klip_tpe"
  RUN="${D:?}/run_${TAG:?}"
  if [ -e "${STOP:?}" ]; then
    echo "$(date '+%F %T %Z') ${STOP} is there: stopping before ${FOLDER} ${TAG}"
    exit 0
  fi
  if [ -f "${RUN:?}/final_results.json" ]; then
    echo "$(date '+%F %T %Z') ${FOLDER} ${TAG} has finished: skipped"
    continue
  fi
  if [ -e "${RUN:?}" ]; then
    echo "$(date '+%F %T %Z') ${RUN} is there without final_results.json: resume it with resume_${TAG}.sh, then start the sequence again"
    exit 1
  fi
  echo "$(date '+%F %T %Z') starting ${FOLDER} ${TAG}: WORKERS=${WORKERS}, card ${GPU_DEVICE}, share ${GPU_SHARE}, ${GPU_WORKERS} card workers"
  bash "${D:?}/run_${TAG:?}.sh" --gpu-share "${GPU_SHARE:?}" --gpu-device "${GPU_DEVICE:?}" \
    --gpu-workers "${GPU_WORKERS:?}" 2>&1 | tee -a "${D:?}/run_${TAG:?}.log"
  [ -f "${RUN:?}/final_results.json" ] || { echo "$(date '+%F %T %Z') ${FOLDER} ${TAG} ended without final_results.json: stopping"; exit 1; }
  echo "$(date '+%F %T %Z') ${FOLDER} ${TAG} finished"
done
echo "$(date '+%F %T %Z') all three runs done"
