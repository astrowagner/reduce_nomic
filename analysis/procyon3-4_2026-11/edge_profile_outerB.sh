#!/usr/bin/env bash
# edge_profile_outerB.sh -- pixel statistics against the distance to each frame's NaN edge, for outerB's inputs.
#
# Usage: NOMIC=/mnt/RAID64TB/NOMIC PY=python3 bash analysis/procyon3-4_2026-11/edge_profile_outerB.sh
#
#   NOMIC  the NOMIC data root (default /mnt/RAID64TB/NOMIC).
#   PY     a Python with numpy, scipy, astropy and matplotlib (default python3).
#
# Runs edge_profile.py on the four 660 px cubes of Procyon_3 + 4's outerB run, every 10th frame, raw and with
# the 25 px boxcar high-pass that the winners of annuli 1 and 2 use, between 139.66 and 307.26 px from the star,
# and writes $NOMIC/_analysis/procyon3-4_2026-11/edge_profile_outerB.csv and .png (refused if they exist).
set -euo pipefail
NOMIC="${NOMIC:-/mnt/RAID64TB/NOMIC}"
PY="${PY:-python3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IN="${NOMIC:?}/Procyon_20211219-20211220/klip_tpe/inputs_outerB"
OUT="${NOMIC:?}/_analysis/procyon3-4_2026-11/edge_profile_outerB"
"${PY:?}" "${HERE:?}/edge_profile.py" \
  "${IN:?}/Procyon_20211219_left_cube1_all_c660.fits" "${IN:?}/Procyon_20211219_left_cube2_all_c660.fits" \
  "${IN:?}/Procyon_20211220_left_cube1_all_c660.fits" "${IN:?}/Procyon_20211220_left_cube2_all_c660.fits" \
  --labels P3_left_A P3_left_B P4_left_A P4_left_B --frames 10 --widths 0 25 --out "${OUT:?}"
