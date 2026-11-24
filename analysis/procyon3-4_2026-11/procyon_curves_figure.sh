#!/usr/bin/env bash
# Draw Procyon's six contrast curves in one boxed plot: the KLIP-TPE final images of Procyon_3
# alone, Procyon_4 alone and the joint Procyon_3 + 4 run, and the best IDL curve at S/N 5 of each.
#
# usage: bash analysis/procyon3-4_2026-11/procyon_curves_figure.sh [ARCHIVE] [OUT_DIR]
#
# Runs plot_procyon_curves.py on the three finished runs (Procyon_20211219 run_adi3ann5c1p5,
# Procyon_20211220 run_adi3ann5, Procyon_20211219-20211220 run_adi3ann5) and the three IDL tables.
# The IDL curve of each data set is, at each separation, the deepest of its versions computed at
# S/N 5: Procyon_3 Dec 2025 and Jan 2024; Procyon_4 Dec 2025 and Dec 2022 (its Jan 2022 version,
# computed at S/N 3 and scaled to 5, is left out); Procyon_3 + 4 Jan 2025. Each data set keeps one
# color: KLIP-TPE solid with filled circles, IDL dashed with open squares.
#
# Parameters
# ----------
# ARCHIVE : str, default /mnt/RAID64TB/NOMIC
#     The NOMIC archive holding the three runs and the IDL tables in _analysis/.
# OUT_DIR : str, default ARCHIVE/_analysis/procyon3-4_2026-11
#     Directory of the outputs, which must exist.
# PY : environment variable, default python3
#     Python with numpy and matplotlib.
#
# Returns
# -------
# OUT_DIR/procyon_klip_tpe_vs_idl_snr5.png : file
#     The figure, 200 dpi.
# OUT_DIR/procyon_klip_tpe_vs_idl_snr5.pdf : file
#     The same figure as PDF, without a creation date.
# OUT_DIR/procyon_klip_tpe_vs_idl_snr5.csv : file
#     The plotted values, one row per curve and separation, with the IDL column each came from.
# An existing output is not overwritten: the script stops before writing anything.
set -euo pipefail
ARCHIVE="${1:-/mnt/RAID64TB/NOMIC}"
OUT_DIR="${2:-${ARCHIVE:?}/_analysis/procyon3-4_2026-11}"
PY="${PY:-python3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:?}")" && pwd)"
P3_RUN="${ARCHIVE:?}/Procyon_20211219/klip_tpe/run_adi3ann5c1p5"
P4_RUN="${ARCHIVE:?}/Procyon_20211220/klip_tpe/run_adi3ann5"
JOINT_RUN="${ARCHIVE:?}/Procyon_20211219-20211220/klip_tpe/run_adi3ann5"
P3_IDL="${ARCHIVE:?}/_analysis/procyon3_2026-11/procyon3_idl_curves_snr5.csv"
P4_IDL="${ARCHIVE:?}/_analysis/procyon4_2026-11/procyon4_idl_curves_snr5.csv"
JOINT_IDL="${ARCHIVE:?}/_analysis/procyon3-4_2026-11/procyon3-4_idl_curves_snr5.csv"
# one color per data set: orange, blue (the documented palette's first two slots) and near-black
C3="#eb6834"
C4="#2a78d6"
C34="#1a1a1a"
BASE="procyon_klip_tpe_vs_idl_snr5"
[ -d "${OUT_DIR:?}" ] || { echo "no directory ${OUT_DIR:?}" >&2; exit 1; }
for d in "${P3_RUN:?}" "${P4_RUN:?}" "${JOINT_RUN:?}"; do
  [ -f "${d:?}/final_grid/curve.txt" ] || { echo "no final_grid/curve.txt in ${d:?}" >&2; exit 1; }
done
for t in "${P3_IDL:?}" "${P4_IDL:?}" "${JOINT_IDL:?}"; do
  [ -f "${t:?}" ] || { echo "no IDL table ${t:?}" >&2; exit 1; }
done
for ext in png pdf csv; do
  if [ -e "${OUT_DIR:?}/${BASE:?}.${ext:?}" ]; then
    echo "${OUT_DIR:?}/${BASE:?}.${ext:?} exists; move it away first" >&2
    exit 1
  fi
done
"${PY:?}" "${HERE:?}/plot_procyon_curves.py" "${OUT_DIR:?}/${BASE:?}.png" \
  --run "KLIP-TPE Procyon_3" "${C3:?}" "${P3_RUN:?}" \
  --run "KLIP-TPE Procyon_4" "${C4:?}" "${P4_RUN:?}" \
  --run "KLIP-TPE Procyon_3 + 4" "${C34:?}" "${JOINT_RUN:?}" \
  --idl "IDL Procyon_3" "${C3:?}" "${P3_IDL:?}" "dec2025_sav,jan2024_sav" \
  --idl "IDL Procyon_4" "${C4:?}" "${P4_IDL:?}" "dec2025_sav,dec2022_5sig" \
  --idl "IDL Procyon_3 + 4" "${C34:?}" "${JOINT_IDL:?}" "jan2025_sav" \
  --pdf "${OUT_DIR:?}/${BASE:?}.pdf" --csv "${OUT_DIR:?}/${BASE:?}.csv"
