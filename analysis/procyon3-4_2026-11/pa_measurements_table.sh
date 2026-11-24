#!/usr/bin/env bash
# Tabulate every injection-recovery measurement behind Procyon's six contrast curves in one CSV.
#
# usage: bash analysis/procyon3-4_2026-11/pa_measurements_table.sh [ARCHIVE] [OUT_DIR]
#
# Runs pa_measurements_table.py on the three KLIP-TPE runs of procyon_curves_figure_1sigma.sh
# (Procyon_20211219 run_adi3ann5c1p5, Procyon_20211220 run_adi3ann5, Procyon_20211219-20211220
# run_adi3ann5; every position of each final-image grid, with every reduction's contrast and
# aperture S/N) and the IDL versions at S/N 5 the figures choose from (Procyon_3 Dec 2025 and
# Jan 2024, Procyon_4 Dec 2025 and the Dec 2022 PDF, Procyon_3-4 Jan 2025; every position angle
# and separation). Each row is flagged by its part in the unclipped and the 3-sigma-clipped band
# figures, whose tables must exist in OUT_DIR, and every plotted point is recomputed from the rows
# as a check.
#
# Parameters
# ----------
# ARCHIVE : str, default /mnt/RAID64TB/NOMIC
#     The NOMIC archive holding the runs and the IDL files.
# OUT_DIR : str, default ARCHIVE/_analysis/procyon3-4_2026-11
#     Directory of the output, holding the figures' tables procyon_klip_tpe_vs_idl_snr5_1sigma.csv
#     and procyon_klip_tpe_vs_idl_snr5_1sigma_clip3.csv.
# PY : environment variable, default python3
#     Python with numpy, scipy and PyMuPDF.
#
# Returns
# -------
# OUT_DIR/procyon_pa_measurements_snr5.csv : file
#     One row per measurement (columns in pa_measurements_table.py).
# OUT_DIR/procyon_pa_measurements_snr5.txt : file
#     The rows per run and IDL version and the check of every plotted point.
# An existing output is not overwritten: the script stops before writing anything.
set -euo pipefail
ARCHIVE="${1:-/mnt/RAID64TB/NOMIC}"
OUT_DIR="${2:-${ARCHIVE:?}/_analysis/procyon3-4_2026-11}"
PY="${PY:-python3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:?}")" && pwd)"
OUT="${OUT_DIR:?}/procyon_pa_measurements_snr5"
FIG="${OUT_DIR:?}/procyon_klip_tpe_vs_idl_snr5_1sigma.csv"
FIG_CLIP="${OUT_DIR:?}/procyon_klip_tpe_vs_idl_snr5_1sigma_clip3.csv"
P3_RUN="${ARCHIVE:?}/Procyon_20211219/klip_tpe/run_adi3ann5c1p5"
P4_RUN="${ARCHIVE:?}/Procyon_20211220/klip_tpe/run_adi3ann5"
JOINT_RUN="${ARCHIVE:?}/Procyon_20211219-20211220/klip_tpe/run_adi3ann5"
P3_DEC2025="${ARCHIVE:?}/Procyon_20211219/IDL/contrast/Procyon_3_contrast_output.sav"
P3_JAN2024="${ARCHIVE:?}/Procyon_20211219/IDL/contrast-old/Procyon_3_contrast_output.sav"
P4_DEC2025="${ARCHIVE:?}/Procyon_20211220/IDL/contrast/Procyon_4_contrast_output.sav"
P4_DEC2022="${ARCHIVE:?}/Procyon_20211220/IDL/contrast/Procyon_4_contrast_curve_repalce_5sig.pdf"
JOINT_JAN2025="${ARCHIVE:?}/Procyon_20211219-20211220/IDL/contrast/Procyon_3-4_contrast_output.sav"
[ -d "${OUT_DIR:?}" ] || { echo "no directory ${OUT_DIR:?}" >&2; exit 1; }
for d in "${P3_RUN:?}" "${P4_RUN:?}" "${JOINT_RUN:?}"; do
  [ -f "${d:?}/final_grid/curve.json" ] || { echo "no final_grid/curve.json in ${d:?}" >&2; exit 1; }
done
for f in "${FIG:?}" "${FIG_CLIP:?}" "${P3_DEC2025:?}" "${P3_JAN2024:?}" "${P4_DEC2025:?}" "${P4_DEC2022:?}" \
         "${JOINT_JAN2025:?}"; do
  [ -f "${f:?}" ] || { echo "missing ${f:?}" >&2; exit 1; }
done
for f in "${OUT:?}.csv" "${OUT:?}.txt"; do
  if [ -e "${f:?}" ]; then echo "${f:?} exists; move it away first" >&2; exit 1; fi
done
"${PY:?}" "${HERE:?}/pa_measurements_table.py" "${OUT:?}.csv" --archive "${ARCHIVE:?}" \
  --run "Procyon_3" adi3ann5c1p5 "${P3_RUN:?}" \
  --run "Procyon_4" adi3ann5 "${P4_RUN:?}" \
  --run "Procyon_3 + 4" adi3ann5 "${JOINT_RUN:?}" \
  --idl "Procyon_3" dec2025_sav "${P3_DEC2025:?}" \
  --idl "Procyon_3" jan2024_sav "${P3_JAN2024:?}" \
  --idl "Procyon_4" dec2025_sav "${P4_DEC2025:?}" \
  --idl "Procyon_4" dec2022_5sig "${P4_DEC2022:?}" \
  --idl "Procyon_3 + 4" jan2025_sav "${JOINT_JAN2025:?}" \
  --figure "${FIG:?}" --figure-clip "${FIG_CLIP:?}" --clip 3 > "${OUT:?}.txt"
cat "${OUT:?}.txt"
