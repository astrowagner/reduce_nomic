#!/usr/bin/env bash
# Make Procyon_3's own run's final comparison products: table and figure against the IDL
# Procyon_3 curves, the final curve against Procyon_4 alone and against the joint Procyon_3 + 4
# run, and the final S/N map checks.
#
# usage: bash analysis/procyon3_2026-11/final_products_adi3ann5c1p5.sh [ARCHIVE] [OUT_DIR]
#
# Runs final_vs_idl.sh on run_adi3ann5c1p5 of Procyon_20211219 with Procyon_4 alone's
# run_adi3ann5 drawn beside it, compare_final_curves.py against Procyon_4 alone's run_adi3ann5 and
# against the joint run_adi3ann5 of Procyon_20211219-20211220, and snr_map_check.py on the final
# S/N map with its injected map and stitched verify report (the ring at 25.5 px and PAs 158 and
# 248 deg are the Procyon_3-4 position in Kevin's log, "angle = 248-90").
#
# Parameters
# ----------
# ARCHIVE : str, default /mnt/RAID64TB/NOMIC
#     The NOMIC archive holding the three runs and _analysis/procyon3_2026-11/.
# OUT_DIR : str, default ARCHIVE/_analysis/procyon3_2026-11
#     Directory of the outputs, which must exist. The IDL table is always read from
#     ARCHIVE/_analysis/procyon3_2026-11/procyon3_idl_curves_snr5.csv.
# PY : environment variable, default python3
#     Python with numpy, astropy and matplotlib.
#
# Returns
# -------
# OUT_DIR/adi3ann5c1p5_final_vs_idl.{txt,csv,png} : files
#     final_vs_idl.sh's tables and figure (see that script).
# OUT_DIR/adi3ann5c1p5_vs_procyon4_final.txt : file
#     The final-image curves of Procyon_3 alone and of Procyon_4 alone, separation by separation.
# OUT_DIR/adi3ann5c1p5_vs_joint_final.txt : file
#     The final-image curves of Procyon_3 alone and of the joint Procyon_3 + 4 run, separation by
#     separation.
# OUT_DIR/adi3ann5c1p5_snr_map_check.txt : file
#     The final S/N map's extremes between 0.30 and 2.45 arcsec, the ring at 0.456 arcsec, and the
#     injected sources' peaks.
# An existing output is not overwritten: the script stops before writing anything.
set -euo pipefail
ARCHIVE="${1:-/mnt/RAID64TB/NOMIC}"
OUT_DIR="${2:-${ARCHIVE:?}/_analysis/procyon3_2026-11}"
PY="${PY:-python3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:?}")" && pwd)"
P4_TOOLS="${HERE:?}/../procyon4_2026-11"
JOINT_TOOLS="${HERE:?}/../procyon3-4_2026-11"
RUN="${ARCHIVE:?}/Procyon_20211219/klip_tpe/run_adi3ann5c1p5"
P4_RUN="${ARCHIVE:?}/Procyon_20211220/klip_tpe/run_adi3ann5"
JOINT_RUN="${ARCHIVE:?}/Procyon_20211219-20211220/klip_tpe/run_adi3ann5"
IDL="${ARCHIVE:?}/_analysis/procyon3_2026-11/procyon3_idl_curves_snr5.csv"
[ -f "${IDL:?}" ] || { echo "no IDL table ${IDL:?}" >&2; exit 1; }
[ -d "${OUT_DIR:?}" ] || { echo "no directory ${OUT_DIR:?}" >&2; exit 1; }
[ -f "${JOINT_RUN:?}/final_grid/curve.txt" ] || { echo "no final_grid/curve.txt in ${JOINT_RUN:?}" >&2; exit 1; }
for f in adi3ann5c1p5_final_vs_idl.txt adi3ann5c1p5_final_vs_idl.csv adi3ann5c1p5_final_vs_idl.png \
         adi3ann5c1p5_vs_procyon4_final.txt adi3ann5c1p5_vs_joint_final.txt \
         adi3ann5c1p5_snr_map_check.txt; do
  if [ -e "${OUT_DIR:?}/${f:?}" ]; then
    echo "${OUT_DIR:?}/${f:?} exists; move it away first" >&2
    exit 1
  fi
done
RUN_LABEL="KLIP-TPE Procyon_3 alone, final image (median, S/N 5)" \
RUN_SHORT="Procyon_3 alone" \
COMPARE_RUN="${P4_RUN:?}" \
COMPARE_LABEL="KLIP-TPE Procyon_4 alone, final image (median, S/N 5)" \
COMPARE_SHORT="Procyon_4 alone" \
PY="${PY:?}" \
bash "${P4_TOOLS:?}/final_vs_idl.sh" "${RUN:?}" "${OUT_DIR:?}" "${IDL:?}" "Procyon_3 (2021 Dec 19)"
"${PY:?}" "${JOINT_TOOLS:?}/compare_final_curves.py" "${RUN:?}" "${P4_RUN:?}" \
  --labels "Procyon_3 alone adi3ann5c1p5" "Procyon_4 alone adi3ann5" \
  > "${OUT_DIR:?}/adi3ann5c1p5_vs_procyon4_final.txt"
"${PY:?}" "${JOINT_TOOLS:?}/compare_final_curves.py" "${RUN:?}" "${JOINT_RUN:?}" \
  --labels "Procyon_3 alone adi3ann5c1p5" "Procyon_3 + 4 adi3ann5" \
  > "${OUT_DIR:?}/adi3ann5c1p5_vs_joint_final.txt"
"${PY:?}" "${JOINT_TOOLS:?}/snr_map_check.py" "${RUN:?}/klip_stitched_snr.fits" \
  --inj-map "${RUN:?}/klip_stitched_snr_inj.fits" --verify-report "${RUN:?}/stitched_verify_report.txt" \
  --ring-pa 158 248 > "${OUT_DIR:?}/adi3ann5c1p5_snr_map_check.txt"
echo "wrote ${OUT_DIR:?}/adi3ann5c1p5_vs_procyon4_final.txt, adi3ann5c1p5_vs_joint_final.txt and adi3ann5c1p5_snr_map_check.txt"
