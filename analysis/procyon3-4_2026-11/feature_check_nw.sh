#!/usr/bin/env bash
# Measure the two bumps about 2" northwest of Procyon in the joint KLIP-TPE run (Procyon_3 + 4,
# run_adi3ann5kn, annulus 3, which finished on Nov 25 2026) in every reduction of the two nights
# at hand, with feature_check.py. Kevin saw "something that looks like a high-pass filtered PSF
# at about 2 arcsec, to the NW" on the run's display; the two bumps there are
#   suspect 1 at 1.72", PA 325 deg: the highest local maximum of the 1.65-2.35" annulus in the
#     joint image by the aperture S/N, and in the evaluation on the display when Kevin looked;
#   suspect 2 at 2.00", PA 313 deg: the peak of the joint run's own S/N map in that direction.
# The reductions, in the figure's rows:
#   1. the joint run: its validated annulus-3 winner (evaluation 9784, annulus03/best_clean.fits),
#      the same with its injections (best_inj.fits, for scale: one source of contrast 1.03e-4 lies
#      at 2.037", PA 307.2 deg, 0.2" from suspect 2), the evaluation on the display (9741) and the
#      best one before validation (8625);
#   2. the joint winner's reduction of each chop position of each night alone
#      (best_clean_partitions.fits: P3_left_A, P3_left_B, P4_left_A, P4_left_B);
#   3. each night alone: the KLIP-TPE final stitches (Procyon_3: run_adi3ann5c1p5kn, Procyon_4:
#      run_adi3ann5kn) and the IDL predecessor's classical ADI (Procyon_N_left_adi.fits);
#   4. the IDL predecessor's classical ADI of each night's cubes 1 and 2 (its two chop positions).
# The IDL predecessor's KLIP products (*_klip.fits, *_contrastcurve_testnoise.fits) hold only the
# last zone its contrast loop reduced, 3.7-4.3", so its classical ADI stands in for it here.
#
# usage: bash analysis/procyon3-4_2026-11/feature_check_nw.sh [ARCHIVE] [OUT_DIR]
#
# Parameters
# ----------
# ARCHIVE : str, default /mnt/RAID64TB/NOMIC
#     The NOMIC archive holding the three runs and the IDL reductions.
# OUT_DIR : str, default ARCHIVE/_analysis/procyon3-4_2026-11
#     Directory of the outputs, which must exist.
# PY : environment variable, default python3
#     Python with numpy, scipy, astropy and matplotlib.
#
# Returns
# -------
# OUT_DIR/procyon_nw_feature_check.csv : file
#     One row per reduction and suspect: the peak's separation, PA and S/N, its rank among the
#     local maxima of the 1.65-2.35" annulus, the annulus' highest maximum, and, where the
#     reduction's injections give the flux scale and the shape of a point source (the joint winner
#     and the two final stitches), the contrast, the radius where the profile turns negative, and
#     the same radius for 100 fake point sources at the suspect's S/N (feature_check.py --fakes).
# OUT_DIR/procyon_nw_feature_check.txt : file
#     The same as two tables, one per suspect, with each reduction's five highest maxima.
# OUT_DIR/procyon_nw_feature_check.png : file
#     A cutout of each reduction around the two suspects, smoothed by half a FWHM, in units of its
#     annulus' sigma, with numbered ticks at each suspect's peak.
# An existing output is not overwritten: feature_check.py stops before writing anything.
set -euo pipefail
ARCHIVE="${1:-/mnt/RAID64TB/NOMIC}"
OUT_DIR="${2:-${ARCHIVE:?}/_analysis/procyon3-4_2026-11}"
PY="${PY:-python3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
J="${ARCHIVE:?}/Procyon_20211219-20211220/klip_tpe/run_adi3ann5kn"
P3="${ARCHIVE:?}/Procyon_20211219/klip_tpe/run_adi3ann5c1p5kn"
P4="${ARCHIVE:?}/Procyon_20211220/klip_tpe/run_adi3ann5kn"
I3="${ARCHIVE:?}/Procyon_20211219/IDL/reduced"
I4="${ARCHIVE:?}/Procyon_20211220/IDL/reduced"
[ -d "${OUT_DIR:?}" ] || { echo "no directory ${OUT_DIR}" >&2; exit 1; }
"${PY:?}" "${HERE:?}/feature_check.py" "${OUT_DIR:?}/procyon_nw_feature_check" \
  --pos 1.72 325 --pos 2.00 313 --fakes 100 --seed 1 \
  "Joint Winner (Eval 9784)=${J:?}/annulus03/best_clean.fits" \
  "Joint Winner with Injections=${J:?}/annulus03/best_inj.fits" \
  "Joint, Eval 9741 (on the Display)=${J:?}/annulus03/evals/eval9741_clean.fits.gz" \
  "Joint, Eval 8625 (Best Before Validation)=${J:?}/annulus03/evals/eval8625_clean.fits.gz" \
  "Joint Winner, Procyon_3 Chop A=${J:?}/annulus03/best_clean_partitions.fits:P3_left_A" \
  "Joint Winner, Procyon_3 Chop B=${J:?}/annulus03/best_clean_partitions.fits:P3_left_B" \
  "Joint Winner, Procyon_4 Chop A=${J:?}/annulus03/best_clean_partitions.fits:P4_left_A" \
  "Joint Winner, Procyon_4 Chop B=${J:?}/annulus03/best_clean_partitions.fits:P4_left_B" \
  "KLIP-TPE Run, Procyon_3 Alone=${P3:?}/klip_stitched.fits" \
  "KLIP-TPE Run, Procyon_4 Alone=${P4:?}/klip_stitched.fits" \
  "idl:IDL ADI, Procyon_3=${I3:?}/Procyon_3_left_adi.fits" \
  "idl:IDL ADI, Procyon_4=${I4:?}/Procyon_4_left_adi.fits" \
  "idl:IDL ADI, Procyon_3 Cube 1=${I3:?}/Procyon_3_left_cube1_adi.fits" \
  "idl:IDL ADI, Procyon_3 Cube 2=${I3:?}/Procyon_3_left_cube2_adi.fits" \
  "idl:IDL ADI, Procyon_4 Cube 1=${I4:?}/Procyon_4_left_cube1_adi.fits" \
  "idl:IDL ADI, Procyon_4 Cube 2=${I4:?}/Procyon_4_left_cube2_adi.fits" \
  --inj "Joint Winner (Eval 9784)=${J:?}/annulus03/best_inj.fits" \
  --inj "KLIP-TPE Run, Procyon_3 Alone=${P3:?}/klip_stitched_inj.fits" \
  --inj "KLIP-TPE Run, Procyon_4 Alone=${P4:?}/klip_stitched_inj.fits" \
  --title "Procyon: The Two Bumps About 2\" Northwest (1: 1.72\", PA 325°; 2: 2.00\", PA 313°) in Each Reduction"
