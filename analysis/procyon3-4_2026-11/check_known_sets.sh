#!/usr/bin/env bash
# Check the three known-source input sets against the sets they repeat.
#
# usage: ARCHIVE="$HOME/mnt/NOMIC" PY="$HOME/venv/bin/python" bash analysis/procyon3-4_2026-11/check_known_sets.sh
#
# The pairs (folder, set repeated, new set): Procyon_20211219 adi3ann5c1p5 -> adi3ann5c1p5kn,
# Procyon_20211220 adi3ann5 -> adi3ann5kn, Procyon_20211219-20211220 adi3ann5 -> adi3ann5kn. For
# each pair:
#
# * every file of the input folder but manifest.json is byte-identical (SHA-256);
# * the run, smoke and resume scripts are identical once the new tag is read as the old one and
#   " --known 0.456 156" is taken out, and each new script holds that option exactly once;
# * the manifests are equal once the new tag is read as the old one, apart from ``known``, which is
#   [[0.456, 156.0]] in the new manifest and absent or None in the old, and apart from fields the
#   builder gained after the old set was made (absent from the old manifest): each must hold its
#   default, None, or for ``nights`` and each partition's ``night`` the set's own night(s);
# * qa_klip_tpe_inputs.py passes on each night's cubes in the new set.
#
# Parameters
# ----------
# ARCHIVE : environment variable, path
#     NOMIC archive root where this runs (~/mnt/NOMIC in the device sandbox).
# PY : environment variable, path, default python3
#     Python with numpy and astropy.
#
# Returns
# -------
# stdout : text
#     Each check and its result; the script stops with status 1 at the first difference.
set -euo pipefail
ARCHIVE="${ARCHIVE:?set ARCHIVE to the NOMIC archive root}"
PY="${PY:-python3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:?}")" && pwd)"
QA="${HERE:?}/../procyon4_2026-11/qa_klip_tpe_inputs.py"
OPT=" --known 0.456 156"
for pair in "Procyon_20211219 adi3ann5c1p5 adi3ann5c1p5kn Procyon_20211219" \
            "Procyon_20211220 adi3ann5 adi3ann5kn Procyon_20211220" \
            "Procyon_20211219-20211220 adi3ann5 adi3ann5kn Procyon_20211219,Procyon_20211220"; do
  read -r FOLDER OLD NEW NIGHTS <<< "${pair}"
  KT="${ARCHIVE:?}/${FOLDER:?}/klip_tpe"
  echo "=== ${FOLDER}: ${OLD} -> ${NEW}"
  # input files but the manifest, byte for byte
  sums_old="$(cd "${KT:?}/inputs_${OLD:?}" && find . -maxdepth 1 -type f ! -name manifest.json -printf '%f\n' | sort | xargs sha256sum)"
  sums_new="$(cd "${KT:?}/inputs_${NEW:?}" && find . -maxdepth 1 -type f ! -name manifest.json -printf '%f\n' | sort | xargs sha256sum)"
  if [ "${sums_old}" != "${sums_new}" ]; then
    echo "input files differ"; diff <(echo "${sums_old}") <(echo "${sums_new}") || true; exit 1
  fi
  echo "input files: $(echo "${sums_new}" | wc -l) byte-identical (SHA-256)"
  # scripts: the same once the tag is mapped back and the option taken out
  for s in "run_%s.sh" "run_%s_smoke.sh" "resume_%s.sh"; do
    f_old="${KT:?}/$(printf "${s}" "${OLD}")"
    f_new="${KT:?}/$(printf "${s}" "${NEW}")"
    n_opt="$(grep -o -F -- "${OPT}" "${f_new:?}" | wc -l)"
    [ "${n_opt}" -eq 1 ] || { echo "${f_new##*/}: ${n_opt} copies of '${OPT}'"; exit 1; }
    if ! diff <(sed -e "s/${NEW}/${OLD}/g" -e "s/${OPT}//" "${f_new:?}") "${f_old:?}" > /dev/null; then
      echo "${f_new##*/} differs from ${f_old##*/} beyond the tag and the option"
      diff <(sed -e "s/${NEW}/${OLD}/g" -e "s/${OPT}//" "${f_new:?}") "${f_old:?}" || true; exit 1
    fi
    echo "${f_new##*/}: as ${f_old##*/} but for the tag and '${OPT# }'"
  done
  # manifests: equal once the tag is mapped back, apart from known
  "${PY:?}" - "${KT:?}/inputs_${OLD:?}/manifest.json" "${KT:?}/inputs_${NEW:?}/manifest.json" "${OLD:?}" "${NEW:?}" \
    "${NIGHTS:?}" <<'EOF'
import json, sys
old_path, new_path, old, new, nights = sys.argv[1:6]
nights = nights.split(",")
old_m = json.load(open(old_path))
new_m = json.loads(open(new_path).read().replace(new, old))
known = new_m.pop("known", None)
old_known = old_m.pop("known", None)
if known != [[0.456, 156.0]] or old_known is not None:
    sys.exit(f"manifest known: new {known}, old {old_known}")
# fields the builder gained after the old set was made hold their default or the set's night(s)
added = []
for k in sorted(set(new_m) - set(old_m)):
    v = new_m.pop(k)
    if not (v is None or (k == "nights" and v == nights)):
        sys.exit(f"manifest field {k} absent from the old set holds {v!r}")
    added.append(k)
for i, (po, pn) in enumerate(zip(old_m.get("partitions", []), new_m.get("partitions", []))):
    for k in sorted(set(pn) - set(po)):
        v = pn.pop(k)
        if not (v is None or (k == "night" and v in nights)):
            sys.exit(f"manifest partition {i} field {k} absent from the old set holds {v!r}")
        added.append(f"partitions[{i}].{k}")
if new_m != old_m:
    keys = sorted(k for k in set(new_m) | set(old_m) if new_m.get(k) != old_m.get(k))
    sys.exit(f"manifests differ beyond the tag and known: {keys}")
print(f"manifest: equal to the old one but for the tag and known {known}"
      + (f"; fields added since, at their defaults or the night(s): {', '.join(added)}" if added else ""))
EOF
  # each night's cubes against its port products
  IFS=',' read -r -a NLIST <<< "${NIGHTS}"
  for night in "${NLIST[@]}"; do
    echo "qa_klip_tpe_inputs.py ${night}_left:"
    "${PY:?}" "${QA:?}" "${ARCHIVE:?}/${night:?}/reduced_py" "${KT:?}/inputs_${NEW:?}" "${night:?}_left" 160 | tail -4
  done
done
echo "all three sets check out"
