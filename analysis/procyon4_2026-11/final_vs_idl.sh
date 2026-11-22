#!/usr/bin/env bash
# Compare a finished KLIP-TPE run with Kevin's IDL curves at S/N 5: table and figure.
#
# usage: bash analysis/procyon4_2026-11/final_vs_idl.sh [RUN] [OUT_DIR] [IDL_TABLE] [NAME]
#
# Runs compare_run_idl.py at the IDL table's separations, at the run's final-image grid
# separations (read from RUN/final_grid/curve.txt) and at 2.5 arcsec, then plot_run_vs_idl.py with
# the annulus edges in EDGES marked. The defaults are Procyon_4's run_adi3ann5.
#
# Parameters
# ----------
# RUN : str, default /mnt/RAID64TB/NOMIC/Procyon_20211220/klip_tpe/run_adi3ann5
#     The finished run directory.
# OUT_DIR : str, default /mnt/RAID64TB/NOMIC/_analysis/procyon4_2026-11
#     Directory of the outputs.
# IDL_TABLE : str, default OUT_DIR/procyon4_idl_curves_snr5.csv
#     The IDL curves at S/N 5, as idl_curves_compare.py writes them.
# NAME : str, default "Procyon_4 (2021 Dec 20)"
#     The data set's name at the start of the plot's title.
# EDGES : environment variable, default "0.70 1.50"
#     Annulus edges to mark, arcsec, separated by spaces.
# PY : environment variable, default python3
#     Python with numpy and matplotlib.
#
# Returns
# -------
# OUT_DIR/<run tag>_final_vs_idl.txt : file
#     The comparison tables, one per curve (compare_run_idl.py's stdout).
# OUT_DIR/<run tag>_final_vs_idl.csv : file
#     The same rows as CSV.
# OUT_DIR/<run tag>_final_vs_idl.png : file
#     The final curve with its 16th-84th percentile band against the IDL curves.
# An existing output is not overwritten: the script stops before writing anything.
set -euo pipefail
RUN="${1:-/mnt/RAID64TB/NOMIC/Procyon_20211220/klip_tpe/run_adi3ann5}"
OUT_DIR="${2:-/mnt/RAID64TB/NOMIC/_analysis/procyon4_2026-11}"
IDL="${3:-${OUT_DIR:?}/procyon4_idl_curves_snr5.csv}"
NAME="${4:-Procyon_4 (2021 Dec 20)}"
EDGES="${EDGES:-0.70 1.50}"
PY="${PY:-python3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:?}")" && pwd)"
TAG="$(basename "${RUN:?}" | sed 's/^run_//')"
[ -f "${RUN:?}/final_results.json" ] || { echo "no final_results.json in ${RUN:?}: the run has not finished" >&2; exit 1; }
[ -f "${RUN:?}/final_grid/curve.txt" ] || { echo "no final_grid/curve.txt in ${RUN:?}" >&2; exit 1; }
[ -f "${IDL:?}" ] || { echo "no IDL table ${IDL:?}" >&2; exit 1; }
for ext in txt csv png; do
  if [ -e "${OUT_DIR:?}/${TAG:?}_final_vs_idl.${ext:?}" ]; then
    echo "${OUT_DIR:?}/${TAG:?}_final_vs_idl.${ext:?} exists; move it away first" >&2
    exit 1
  fi
done
# the final grid's own separations, rounded to 3 decimals as contrast_curve.txt writes them
read -r -a SEPS <<< "$(awk '!/^#/ && NF {printf "%.3f ", $1}' "${RUN:?}/final_grid/curve.txt")"
[ "${#SEPS[@]}" -gt 0 ] || { echo "no separations in ${RUN:?}/final_grid/curve.txt" >&2; exit 1; }
read -r -a EDGE_LIST <<< "${EDGES:?}"
"${PY:?}" "${HERE:?}/compare_run_idl.py" "${RUN:?}" "${IDL:?}" \
  --sep "${SEPS[@]}" 2.5 \
  --out "${OUT_DIR:?}/${TAG:?}_final_vs_idl.csv" > "${OUT_DIR:?}/${TAG:?}_final_vs_idl.txt"
"${PY:?}" "${HERE:?}/plot_run_vs_idl.py" "${RUN:?}" "${IDL:?}" "${OUT_DIR:?}/${TAG:?}_final_vs_idl.png" \
  --title "${NAME:?}: KLIP-TPE ${TAG:?} Final Image and IDL Curves" --edges "${EDGE_LIST[@]}"
echo "wrote ${OUT_DIR:?}/${TAG:?}_final_vs_idl.{txt,csv,png}"
