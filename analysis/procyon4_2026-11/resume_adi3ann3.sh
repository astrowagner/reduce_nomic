#!/usr/bin/env bash
# Usage: KT=<klip-tpe executable> [WORKERS=N] bash resume_adi3ann3.sh
#
# Resume the stopped KLIP-TPE run run_adi3ann3 (Procyon_20211220 left side, annuli 0-0.7",
# 0.7-1.5" and 1.5-2.5", annulus 1 at the forced injection contrast 4e-4) from its checkpoint.
#
# `klip-tpe resume --run-dir` alone refuses a generic run: the run record restores the data
# arguments only for runs made with --root, so this script gives --instrument generic and the
# data arguments of run_adi3ann3.sh.  The search configuration (annuli, budgets, forced
# contrast, seed, validation, --verify) comes from the checkpoint.  The registry still refuses
# the resume while the run's own process is alive.
#
# Environment
# -----------
# KT : str
#     The klip-tpe executable of the pinned clone the run was started with (aa0702b,
#     ~/venvs/klip-tpe-vega/bin/klip-tpe).
# WORKERS : str, default auto
#     Worker budget, as klip-tpe --workers takes it ('auto' is one worker per physical core).
#
# Outputs
# -------
# The run goes on writing its files in run_adi3ann3/ (checkpoint, results, steps, products).
set -euo pipefail
KT="${KT:?set KT to the klip-tpe executable of the pinned clone}"
D=/mnt/RAID64TB/NOMIC/Procyon_20211220/klip_tpe
RUN="${D:?}/run_adi3ann3"
IN="${D:?}/inputs_adi3ann3"
[ -f "${RUN:?}/checkpoint.json" ] || { echo "${RUN} has no checkpoint.json"; exit 1; }
[ ! -e "${RUN:?}/final_results.json" ] || { echo "${RUN} is already finished"; exit 1; }
"${KT:?}" resume --instrument generic \
  --cube "${IN:?}/Procyon_20211220_left_cube1_all_c320.fits" "${IN:?}/Procyon_20211220_left_cube2_all_c320.fits" \
  --angles "${IN:?}/Procyon_20211220_left_angles1_all.fits" "${IN:?}/Procyon_20211220_left_angles2_all.fits" \
  --psf "${IN:?}/Procyon_20211220_left_cube1_psf60.fits" "${IN:?}/Procyon_20211220_left_cube2_psf60.fits" \
  --names left_A left_B \
  --pxscale 0.0179 --lam 11.1e-6 --diam 8.4 \
  --workers "${WORKERS:-auto}" --show \
  --run-dir "${RUN:?}"
