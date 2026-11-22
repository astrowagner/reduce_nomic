#!/usr/bin/env bash
# Tabulate Kevin's IDL contrast curves of Procyon_3 and of Procyon_3-4 combined at S/N 5.
#
# usage: bash analysis/procyon3_2026-11/idl_tables.sh [ARCHIVE]
#
# Runs ../procyon4_2026-11/idl_curves_compare.py on the save files of contrastcurve_nomic.pro:
#   Procyon_3 (Procyon_20211219/IDL/): contrast/ (Dec 2025, S/N 5; at 0.40 and 0.50" every position
#     angle failed, and at 0.55-0.80" 13-15 of 20 were recovered) and contrast-old/ (Jan 2024, S/N 5,
#     all 20 position angles to 2.0").
#   Procyon_3-4 (Procyon_20211219-20211220/IDL/contrast/): Jan 2025, S/N 5, 20 position angles.
#     "Procyon_3-4_contrast_curve 2.pdf" (Jul 2024) stops at test 207 of 240 and is left out.
#
# Parameters
# ----------
# ARCHIVE : str, default /mnt/RAID64TB/NOMIC
#     The NOMIC archive.
# PY : environment variable, default python3
#     Python with numpy, scipy and PyMuPDF (idl_curves_compare.py imports idl_curve_pdf.py).
#
# Returns
# -------
# ARCHIVE/_analysis/procyon3_2026-11/procyon3_idl_curves_snr5.{csv,txt} : file
#     The Procyon_3 table (columns dec2025_sav, jan2024_sav) and the printed table with the
#     position-angle counts.
# ARCHIVE/_analysis/procyon3-4_2026-11/procyon3-4_idl_curves_snr5.{csv,txt} : file
#     The Procyon_3-4 table (column jan2025_sav).
# An existing output is not overwritten: the script stops before writing anything.
set -euo pipefail
ARCHIVE="${1:-/mnt/RAID64TB/NOMIC}"
PY="${PY:-python3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:?}")" && pwd)"
CMP="${HERE:?}/../procyon4_2026-11/idl_curves_compare.py"
P3="${ARCHIVE:?}/Procyon_20211219/IDL"
P34="${ARCHIVE:?}/Procyon_20211219-20211220/IDL/contrast"
OUT3="${ARCHIVE:?}/_analysis/procyon3_2026-11"
OUT34="${ARCHIVE:?}/_analysis/procyon3-4_2026-11"
for f in "${OUT3:?}/procyon3_idl_curves_snr5.csv" "${OUT3:?}/procyon3_idl_curves_snr5.txt" \
         "${OUT34:?}/procyon3-4_idl_curves_snr5.csv" "${OUT34:?}/procyon3-4_idl_curves_snr5.txt"; do
  if [ -e "${f:?}" ]; then echo "${f:?} exists; move it away first" >&2; exit 1; fi
done
for f in "${P3:?}/contrast/Procyon_3_contrast_output.sav" "${P3:?}/contrast-old/Procyon_3_contrast_output.sav" \
         "${P34:?}/Procyon_3-4_contrast_output.sav"; do
  [ -f "${f:?}" ] || { echo "missing ${f:?}" >&2; exit 1; }
done
mkdir -p "${OUT3:?}" "${OUT34:?}"
"${PY:?}" "${CMP:?}" "${OUT3:?}/procyon3_idl_curves_snr5.csv" 5 \
  "dec2025_sav:5:${P3:?}/contrast/Procyon_3_contrast_output.sav" \
  "jan2024_sav:5:${P3:?}/contrast-old/Procyon_3_contrast_output.sav" > "${OUT3:?}/procyon3_idl_curves_snr5.txt"
"${PY:?}" "${CMP:?}" "${OUT34:?}/procyon3-4_idl_curves_snr5.csv" 5 \
  "jan2025_sav:5:${P34:?}/Procyon_3-4_contrast_output.sav" > "${OUT34:?}/procyon3-4_idl_curves_snr5.txt"
cat "${OUT3:?}/procyon3_idl_curves_snr5.txt" "${OUT34:?}/procyon3-4_idl_curves_snr5.txt"
