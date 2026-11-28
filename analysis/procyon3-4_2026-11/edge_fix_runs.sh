#!/usr/bin/env bash
# Usage: edge_fix_runs.sh KT_SRC RUN_DIR INPUTS OUT
#
# Reproduce the edge-artifact prototype on annulus 2's winner of the Procyon_3 + 4 outerB run with
# edge_fix_prototype.py: every variant's reduction, the stage profiles, the comparison, the
# injection-recovery sensitivity and the before/after figure. A reduction or profile whose output
# exists is skipped, so the script resumes where it stopped; the prototype never overwrites one.
# The comparison, sensitivity and figure files are made again on every run.
#
#   KT_SRC   directory holding the klip_tpe package with bf99474's code
#   RUN_DIR  the run folder: annulus02/annulus_done.json, results.jsonl, run_setup.json,
#            coverage.fits and pixel_mask.fits
#   INPUTS   the run's inputs folder: manifest.json and the winner's cubes, angles and PSFs
#   OUT      output folder, made if missing
#
# Each reduction takes about a minute on two threads and needs about 3 GB of memory.
set -euo pipefail

if [ "$#" -ne 4 ]; then
    sed -n '2,17p' "$0"
    exit 2
fi
KT_SRC=$1
RUN_DIR=$2
INPUTS=$3
OUT=$4
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
python3 "${PROTO:?}" figure "${common[@]}" --out "${OUT:?}" --before base --after mkn_pix \
    --before-title "Before: KLIP-TPE As Is" --after-title "After: Masked KLIP, NaN Variance, Pixel Weights"
