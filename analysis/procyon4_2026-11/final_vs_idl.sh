#!/usr/bin/env bash
# Compare a finished KLIP-TPE run of Procyon_4 with Kevin's IDL curves at S/N 5: table and figure.
#
# usage: bash analysis/procyon4_2026-11/final_vs_idl.sh [RUN] [OUT_DIR]
#
# Runs compare_run_idl.py at the IDL table's separations and at the final-image grid's own
# (0.35, 0.583, 0.833, 1.1, 1.367, 1.6, 1.8, 2.2, 2.4 and 2.5 arcsec), then plot_run_vs_idl.py, with
# the annulus edges of adi3ann5 (0.70 and 1.50 arcsec) marked.
#
# Parameters
# ----------
# RUN : str, default /mnt/RAID64TB/NOMIC/Procyon_20211220/klip_tpe/run_adi3ann5
#     The finished run directory.
# OUT_DIR : str, default /mnt/RAID64TB/NOMIC/_analysis/procyon4_2026-11
#     Directory of the outputs; the IDL table procyon4_idl_curves_snr5.csv is read from it.
# PY : environment variable, default python3
#     Python with numpy and matplotlib.
#
# Returns
# -------
# OUT_DIR/<run name>_final_vs_idl.txt : file
#     The comparison tables, one per curve (compare_run_idl.py's stdout).
# OUT_DIR/<run name>_final_vs_idl.csv : file
#     The same rows as CSV.
# OUT_DIR/<run name>_final_vs_idl.png : file
#     The final curve with its 16th-84th percentile band against the IDL curves.
# An existing output is not overwritten: the script stops before writing anything.
set -euo pipefail
RUN="${1:-/mnt/RAID64TB/NOMIC/Procyon_20211220/klip_tpe/run_adi3ann5}"
OUT_DIR="${2:-/mnt/RAID64TB/NOMIC/_analysis/procyon4_2026-11}"
PY="${PY:-python3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:?}")" && pwd)"
IDL="${OUT_DIR:?}/procyon4_idl_curves_snr5.csv"
TAG="$(basename "${RUN:?}" | sed 's/^run_//')"
[ -f "${RUN:?}/final_results.json" ] || { echo "no final_results.json in ${RUN:?}: the run has not finished" >&2; exit 1; }
[ -f "${IDL:?}" ] || { echo "no IDL table ${IDL:?}" >&2; exit 1; }
for ext in txt csv png; do
  if [ -e "${OUT_DIR:?}/${TAG:?}_final_vs_idl.${ext:?}" ]; then
    echo "${OUT_DIR:?}/${TAG:?}_final_vs_idl.${ext:?} exists; move it away first" >&2
    exit 1
  fi
done
"${PY:?}" "${HERE:?}/compare_run_idl.py" "${RUN:?}" "${IDL:?}" \
  --sep 0.35 0.583 0.833 1.1 1.367 1.6 1.8 2.2 2.4 2.5 \
  --out "${OUT_DIR:?}/${TAG:?}_final_vs_idl.csv" > "${OUT_DIR:?}/${TAG:?}_final_vs_idl.txt"
"${PY:?}" "${HERE:?}/plot_run_vs_idl.py" "${RUN:?}" "${IDL:?}" "${OUT_DIR:?}/${TAG:?}_final_vs_idl.png" \
  --title "Procyon_4 (2021 Dec 20): KLIP-TPE ${TAG:?} Final Image and IDL Curves" --edges 0.70 1.50
echo "wrote ${OUT_DIR:?}/${TAG:?}_final_vs_idl.{txt,csv,png}"
