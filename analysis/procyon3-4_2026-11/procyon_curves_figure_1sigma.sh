#!/usr/bin/env bash
# Draw Procyon's six contrast curves in one boxed plot with bands of ±1 standard deviation over
# position angle: the KLIP-TPE final images of Procyon_3 alone, Procyon_4 alone and the joint
# Procyon_3 + 4 run, and the best IDL curve at S/N 5 of each, as in procyon_curves_figure.sh.
#
# usage: bash analysis/procyon3-4_2026-11/procyon_curves_figure_1sigma.sh [ARCHIVE] [OUT_DIR]
#
# First tabulates, with idl_pa_scatter.py, each IDL version's mean, sample standard deviation and
# count over position angle, from the files the IDL tables were read from:
#   Procyon_3: the Dec 2025 (Procyon_20211219/IDL/contrast/) and Jan 2024 (contrast-old/) save files.
#   Procyon_4: the Dec 2025 save file and the Dec 2022 PDF (Procyon_20211220/IDL/contrast/
#     Procyon_4_contrast_curve_repalce_5sig.pdf), which was drawn at test 156 of 160, so only its
#     first 19 of 20 position angles are complete; its black line is their mean.
#   Procyon_3-4: the Jan 2025 save file (Procyon_20211219-20211220/IDL/contrast/).
# Then runs plot_procyon_curves.py with --bands: each run's band from the settled positions in its
# final_grid/curve.json, each IDL point's band from the version it came from. With CLIP set, every
# curve's values are sigma-clipped over position angle first (pa_clip.py): the runs' curves become
# the medians of the settled contrasts kept, the IDL curves the deepest clipped means, and every
# output name ends in _clip<CLIP>.
#
# Parameters
# ----------
# ARCHIVE : str, default /mnt/RAID64TB/NOMIC
#     The NOMIC archive holding the three runs, the IDL files and the IDL tables in _analysis/.
# OUT_DIR : str, default ARCHIVE/_analysis/procyon3-4_2026-11
#     Directory of the outputs, which must exist.
# PY : environment variable, default python3
#     Python with numpy, scipy, matplotlib and PyMuPDF.
# CLIP : environment variable, optional
#     Clipping threshold over position angle, in standard deviations (e.g. 3); none by default.
#
# Returns
# -------
# OUT_DIR/procyon3_idl_pa_scatter_snr5[_clipCLIP].{csv,txt} : file
#     Procyon_3's IDL versions (dec2025_sav, jan2024_sav): per separation the mean, standard
#     deviation and count over position angle (CSV), and the printed table with its sources and
#     the values clipped (TXT).
# OUT_DIR/procyon4_idl_pa_scatter_snr5[_clipCLIP].{csv,txt} : file
#     The same for Procyon_4 (dec2025_sav, dec2022_5sig).
# OUT_DIR/procyon3-4_idl_pa_scatter_snr5[_clipCLIP].{csv,txt} : file
#     The same for Procyon_3-4 (jan2025_sav).
# OUT_DIR/procyon_klip_tpe_vs_idl_snr5_1sigma[_clipCLIP].png : file
#     The figure, 200 dpi.
# OUT_DIR/procyon_klip_tpe_vs_idl_snr5_1sigma[_clipCLIP].pdf : file
#     The same figure as PDF, without a creation date.
# OUT_DIR/procyon_klip_tpe_vs_idl_snr5_1sigma[_clipCLIP].csv : file
#     The plotted values, one row per curve and separation, with the standard deviation, the values
#     kept and clipped, the band's edges and the source of each value.
# OUT_DIR/procyon_klip_tpe_vs_idl_snr5_1sigma[_clipCLIP].txt : file
#     The printed values, with the points whose band reaches below zero.
# An existing output is not overwritten: the script stops before writing anything.
set -euo pipefail
ARCHIVE="${1:-/mnt/RAID64TB/NOMIC}"
OUT_DIR="${2:-${ARCHIVE:?}/_analysis/procyon3-4_2026-11}"
PY="${PY:-python3}"
CLIP="${CLIP:-}"
if [ -n "${CLIP}" ]; then
  SUFFIX="_clip${CLIP:?}"
  CLIP_OPT=(--clip "${CLIP:?}")
else
  SUFFIX=""
  CLIP_OPT=()
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:?}")" && pwd)"
P3_RUN="${ARCHIVE:?}/Procyon_20211219/klip_tpe/run_adi3ann5c1p5"
P4_RUN="${ARCHIVE:?}/Procyon_20211220/klip_tpe/run_adi3ann5"
JOINT_RUN="${ARCHIVE:?}/Procyon_20211219-20211220/klip_tpe/run_adi3ann5"
P3_IDL="${ARCHIVE:?}/_analysis/procyon3_2026-11/procyon3_idl_curves_snr5.csv"
P4_IDL="${ARCHIVE:?}/_analysis/procyon4_2026-11/procyon4_idl_curves_snr5.csv"
JOINT_IDL="${ARCHIVE:?}/_analysis/procyon3-4_2026-11/procyon3-4_idl_curves_snr5.csv"
P3_DEC2025="${ARCHIVE:?}/Procyon_20211219/IDL/contrast/Procyon_3_contrast_output.sav"
P3_JAN2024="${ARCHIVE:?}/Procyon_20211219/IDL/contrast-old/Procyon_3_contrast_output.sav"
P4_DEC2025="${ARCHIVE:?}/Procyon_20211220/IDL/contrast/Procyon_4_contrast_output.sav"
P4_DEC2022="${ARCHIVE:?}/Procyon_20211220/IDL/contrast/Procyon_4_contrast_curve_repalce_5sig.pdf"
JOINT_JAN2025="${ARCHIVE:?}/Procyon_20211219-20211220/IDL/contrast/Procyon_3-4_contrast_output.sav"
P3_SCAT="${OUT_DIR:?}/procyon3_idl_pa_scatter_snr5${SUFFIX}"
P4_SCAT="${OUT_DIR:?}/procyon4_idl_pa_scatter_snr5${SUFFIX}"
JOINT_SCAT="${OUT_DIR:?}/procyon3-4_idl_pa_scatter_snr5${SUFFIX}"
BASE="${OUT_DIR:?}/procyon_klip_tpe_vs_idl_snr5_1sigma${SUFFIX}"
# one color per data set: orange, blue (the documented palette's first two slots) and near-black
C3="#eb6834"
C4="#2a78d6"
C34="#1a1a1a"
[ -d "${OUT_DIR:?}" ] || { echo "no directory ${OUT_DIR:?}" >&2; exit 1; }
for d in "${P3_RUN:?}" "${P4_RUN:?}" "${JOINT_RUN:?}"; do
  for f in curve.txt curve.json; do
    [ -f "${d:?}/final_grid/${f:?}" ] || { echo "no final_grid/${f:?} in ${d:?}" >&2; exit 1; }
  done
done
for f in "${P3_IDL:?}" "${P4_IDL:?}" "${JOINT_IDL:?}" "${P3_DEC2025:?}" "${P3_JAN2024:?}" "${P4_DEC2025:?}" \
         "${P4_DEC2022:?}" "${JOINT_JAN2025:?}"; do
  [ -f "${f:?}" ] || { echo "missing ${f:?}" >&2; exit 1; }
done
for f in "${P3_SCAT:?}.csv" "${P3_SCAT:?}.txt" "${P4_SCAT:?}.csv" "${P4_SCAT:?}.txt" "${JOINT_SCAT:?}.csv" \
         "${JOINT_SCAT:?}.txt" "${BASE:?}.png" "${BASE:?}.pdf" "${BASE:?}.csv" "${BASE:?}.txt"; do
  if [ -e "${f:?}" ]; then echo "${f:?} exists; move it away first" >&2; exit 1; fi
done
"${PY:?}" "${HERE:?}/idl_pa_scatter.py" "${P3_SCAT:?}.csv" 5 \
  "dec2025_sav:5:${P3_DEC2025:?}" "jan2024_sav:5:${P3_JAN2024:?}" ${CLIP_OPT[@]+"${CLIP_OPT[@]}"} > "${P3_SCAT:?}.txt"
"${PY:?}" "${HERE:?}/idl_pa_scatter.py" "${P4_SCAT:?}.csv" 5 \
  "dec2025_sav:5:${P4_DEC2025:?}" "dec2022_5sig:5:${P4_DEC2022:?}" ${CLIP_OPT[@]+"${CLIP_OPT[@]}"} > "${P4_SCAT:?}.txt"
"${PY:?}" "${HERE:?}/idl_pa_scatter.py" "${JOINT_SCAT:?}.csv" 5 \
  "jan2025_sav:5:${JOINT_JAN2025:?}" ${CLIP_OPT[@]+"${CLIP_OPT[@]}"} > "${JOINT_SCAT:?}.txt"
"${PY:?}" "${HERE:?}/plot_procyon_curves.py" "${BASE:?}.png" \
  --run "KLIP-TPE Procyon_3" "${C3:?}" "${P3_RUN:?}" \
  --run "KLIP-TPE Procyon_4" "${C4:?}" "${P4_RUN:?}" \
  --run "KLIP-TPE Procyon_3 + 4" "${C34:?}" "${JOINT_RUN:?}" \
  --idl "IDL Procyon_3" "${C3:?}" "${P3_IDL:?}" "dec2025_sav,jan2024_sav" \
  --idl "IDL Procyon_4" "${C4:?}" "${P4_IDL:?}" "dec2025_sav,dec2022_5sig" \
  --idl "IDL Procyon_3 + 4" "${C34:?}" "${JOINT_IDL:?}" "jan2025_sav" \
  --bands ${CLIP_OPT[@]+"${CLIP_OPT[@]}"} \
  --idl-scatter "IDL Procyon_3" "${P3_SCAT:?}.csv" \
  --idl-scatter "IDL Procyon_4" "${P4_SCAT:?}.csv" \
  --idl-scatter "IDL Procyon_3 + 4" "${JOINT_SCAT:?}.csv" \
  --pdf "${BASE:?}.pdf" --csv "${BASE:?}.csv" > "${BASE:?}.txt"
cat "${P3_SCAT:?}.txt" "${P4_SCAT:?}.txt" "${JOINT_SCAT:?}.txt" "${BASE:?}.txt"
