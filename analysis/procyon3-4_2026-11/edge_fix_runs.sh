#!/usr/bin/env bash
# Usage: edge_fix_runs.sh KT_SRC RUN_DIR INPUTS OUT [KT_NEW]
#
# Reproduce the edge-artifact prototype on annulus 2's winner of the Procyon_3 + 4 outerB run with
# edge_fix_prototype.py: every variant's reduction, the stage profiles, the comparison, the
# injection-recovery sensitivity and the before/after figure. With KT_NEW, also check klip-tpe's
# own implementation of the fixes against the prototype. A reduction or profile whose output
# exists is skipped, so the script resumes where it stopped; the prototype never overwrites one.
# The comparison, sensitivity and figure files are made again on every run.
#
#   KT_SRC   directory holding the klip_tpe package with bf99474's code
#   RUN_DIR  the run folder: annulus02/annulus_done.json, results.jsonl, run_setup.json,
#            coverage.fits and pixel_mask.fits
#   INPUTS   the run's inputs folder: manifest.json and the winner's cubes, angles and PSFs
#   OUT      output folder, made if missing
#   KT_NEW   optional: directory holding the klip_tpe package at 1a42207 or later
#
# Each reduction takes about a minute on two threads and needs about 3 GB of memory.
set -euo pipefail

if [ "$#" -ne 4 ] && [ "$#" -ne 5 ]; then
    sed -n '2,19p' "$0"
    exit 2
fi
KT_SRC=$1
RUN_DIR=$2
INPUTS=$3
OUT=$4
KT_NEW=${5:-}
HERE=$(cd "$(dirname "$0")" && pwd)
PROTO="${HERE:?}/edge_fix_prototype.py"
mkdir -p "${OUT:?}"

common=(--kt-src "${KT_SRC:?}" --run-dir "${RUN_DIR:?}" --annulus 2)
red=(reduce "${common[@]}" --inputs "${INPUTS:?}" --out "${OUT:?}" --workers 2)

# reduce_once LABEL CHECK ARGS...: reduce with --label LABEL unless OUT/CHECK exists
reduce_once() {
    local label=$1 check=$2
    shift 2
    if [ -e "${OUT:?}/${check:?}" ]; then
        echo "skip ${label}: ${check} exists"
        return 0
    fi
    python3 "${PROTO:?}" "${red[@]}" --label "${label:?}" "$@"
}

# KLIP-TPE's one weight per partition: as it is, the edge masks, and two of its own options
reduce_once base base_clean.fits --variant base --inject
reduce_once hpmask hpmask_clean.fits --variant hpmask --inject
reduce_once hpmask2 hpmask2_clean.fits --variant hpmask2 --inject
reduce_once hpmask_klip hpmask_klip_clean.fits --variant hpmask_klip --inject
reduce_once nopost nopost_clean.fits --variant base --param post_hp=false --inject
reduce_once mean mean_clean.fits --variant base --param comb_type=mean --inject
reduce_once hk_nopost_mean hk_nopost_mean_clean.fits --variant hpmask_klip --param post_hp=false \
    --param comb_type=mean --inject

# per-pixel inverse-variance partition weights, alone and with each patch
for pair in base_pix:base hm_pix:hpmask mk_pix:mklip nv_pix:nanvar mkn_pix:mklip+nanvar \
            hk_pix:hpmask_klip hkn_pix:hpmask_klip_nanvar; do
    reduce_once "${pair%%:*}" "${pair%%:*}_clean.fits" --variant "${pair#*:}" --pixcomb --inject
done

# the injected sources turned by 20 and 40 degrees: 18 positions for the sensitivity
for rot in 20 40; do
    reduce_once base "base_inj_r${rot}.fits" --variant base --inject --inj-tag "inj_r${rot}" \
        --rotate-sources "${rot}" --skip-clean
    for pair in base_pix:base nv_pix:nanvar mkn_pix:mklip+nanvar hkn_pix:hpmask_klip_nanvar; do
        reduce_once "${pair%%:*}" "${pair%%:*}_inj_r${rot}.fits" --variant "${pair#*:}" --pixcomb --inject \
            --inj-tag "inj_r${rot}" --rotate-sources "${rot}" --skip-clean
    done
done

# the saved partition images combined with coverage weights (n_p / N_p, equal where every frame has data)
for pair in base_cov:base_pix mkn_cov:mkn_pix; do
    if [ -e "${OUT:?}/${pair%%:*}_clean.fits" ]; then
        echo "skip ${pair%%:*}: ${pair%%:*}_clean.fits exists"
        continue
    fi
    python3 "${PROTO:?}" recombine --out "${OUT:?}" --source "${pair#*:}" --label "${pair%%:*}" --mode coverage \
        --tags clean inj inj_r20 inj_r40
done

# each step of one partition's reduction against the distance to the detector's edge
for v in base hpmask_klip; do
    for pp in P3_left_B:P3_left_B P4_left_A:P4A P4_left_B:P4_left_B; do
        label="${v}_${pp#*:}"
        if [ -e "${OUT:?}/${label}_stages.txt" ]; then
            echo "skip ${label}: ${label}_stages.txt exists"
            continue
        fi
        python3 "${PROTO:?}" stages "${common[@]}" --inputs "${INPUTS:?}" --out "${OUT:?}" --workers 2 \
            --variant "${v}" --partition "${pp%%:*}" --label "${label}"
    done
done

# the comparison, the sensitivity and the figure
python3 "${PROTO:?}" compare "${common[@]}" --out "${OUT:?}" \
    --only base base_pix hm_pix mk_pix nv_pix mkn_pix hk_pix hkn_pix --counts base:base_pix
python3 "${PROTO:?}" sensitivity "${common[@]}" --out "${OUT:?}" \
    --labels base base_pix nv_pix mkn_pix hkn_pix --tags inj inj_r20 inj_r40 --counts base:base_pix \
    --names "base:KLIP-TPE As Is" --names "base_pix:Pixel Weights" --names "nv_pix:+ NaN Variance" \
    --names "mkn_pix:+ Masked KLIP (Recommended)" --names "hkn_pix:+ High-Pass Edge Mask" \
    --legend-title "Each Adds to the One Above"
python3 "${PROTO:?}" sensitivity "${common[@]}" --out "${OUT:?}" --prefix sensitivity_weights \
    --labels base base_pix base_cov mkn_pix mkn_cov --tags inj inj_r20 inj_r40 --counts base:base_pix \
    --names "base:KLIP-TPE As Is" --names "base_pix:Inverse-Variance Weights" --names "base_cov:Coverage Weights" \
    --names "mkn_pix:Masked KLIP + NaN Var., Inverse-Variance" --names "mkn_cov:Masked KLIP + NaN Var., Coverage"
python3 "${PROTO:?}" figure "${common[@]}" --out "${OUT:?}" --before base --after mkn_pix \
    --before-title "Before: KLIP-TPE As Is" --after-title "After: Masked KLIP, NaN Variance, Pixel Weights"

# klip-tpe's own implementation (KT_NEW, 1a42207 or later): the old rules must give the run's image to
# the bit; the four changes, with the old eigenvalue cut and stamp aperture (kta_*) or as a new run
# makes them (kt_*), are measured as the prototype was
if [ -n "${KT_NEW}" ]; then
    new=(reduce --kt-src "${KT_NEW:?}" --run-dir "${RUN_DIR:?}" --annulus 2 --inputs "${INPUTS:?}" --out "${OUT:?}"
         --workers 2 --variant base)
    old_rules=(--param klip_eig_cut=absolute --param ee_aperture=whole)
    fixes=(--param klip_nan_fit=masked --param nw_var=data)
    if [ ! -e "${OUT:?}/kt_old2_clean.fits" ]; then
        python3 "${PROTO:?}" "${new[@]}" "${old_rules[@]}" --label kt_old2 --combine scalar \
            --param klip_nan_fit=zero --param nw_var=all --inject
    fi
    # the old rules without the old eigenvalue cut and stamp aperture: the eigenvalue cut's own effect
    if [ ! -e "${OUT:?}/kt_old_clean.fits" ]; then
        python3 "${PROTO:?}" "${new[@]}" --label kt_old --combine scalar --param klip_nan_fit=zero --param nw_var=all \
            --inject
    fi
    for spec in kta_cov:coverage:old kta_iv:inverse_variance:old kt_cov:coverage:new kt_iv:inverse_variance:new; do
        IFS=: read -r label mode rules <<< "${spec}"
        extra=()
        if [ "${rules}" = old ]; then
            extra=("${old_rules[@]}")
        fi
        if [ ! -e "${OUT:?}/${label}_clean.fits" ]; then
            python3 "${PROTO:?}" "${new[@]}" "${extra[@]}" "${fixes[@]}" --label "${label}" --combine "${mode}" --inject
        fi
        for rot in 20 40; do
            if [ ! -e "${OUT:?}/${label}_inj_r${rot}.fits" ]; then
                python3 "${PROTO:?}" "${new[@]}" "${extra[@]}" "${fixes[@]}" --label "${label}" --combine "${mode}" \
                    --inject --inj-tag "inj_r${rot}" --rotate-sources "${rot}" --skip-clean
            fi
        done
    done
    # the clean images once more, now with the S/N maps' floored noise maps (the same images)
    for spec in ktn_cov:coverage ktn_iv:inverse_variance; do
        if [ ! -e "${OUT:?}/${spec%%:*}_clean.fits" ]; then
            python3 "${PROTO:?}" "${new[@]}" "${fixes[@]}" --label "${spec%%:*}" --combine "${spec#*:}"
        fi
    done
    python3 "${PROTO:?}" sensitivity --kt-src "${KT_NEW:?}" --run-dir "${RUN_DIR:?}" --annulus 2 --out "${OUT:?}" \
        --prefix sensitivity_kt1a42207 --labels base kta_cov kta_iv kt_cov kt_iv --tags inj inj_r20 inj_r40 \
        --counts base:base_pix --names "base:KLIP-TPE bf99474" --names "kta_cov:1a42207 Coverage, Old Eig. Cut" \
        --names "kta_iv:1a42207 Inverse Var., Old Eig. Cut" --names "kt_cov:1a42207 Coverage (New Run)" \
        --names "kt_iv:1a42207 Inverse Var. (New Run)"
    python3 "${PROTO:?}" ktsnr --kt-src "${KT_NEW:?}" --run-dir "${RUN_DIR:?}" --annulus 2 --out "${OUT:?}" \
        --labels base kt_old2 ktn_cov ktn_iv
    python3 "${PROTO:?}" pixdiff --run-dir "${RUN_DIR:?}" --annulus 2 --out "${OUT:?}" \
        --pairs kt_old2:run kt_old2:base kt_old:kt_old2 kta_cov:mkn_cov kta_iv:mkn_pix kt_cov:kta_cov kt_iv:kta_iv
fi
