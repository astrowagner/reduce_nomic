#!/usr/bin/env bash
# memlog.sh -- append a memory snapshot of this machine and of this user's largest processes to a CSV file.
#
# Usage: tools/memlog.sh OUT_CSV [INTERVAL_S]
#
#   OUT_CSV     file the snapshots are appended to; its header line is written first when it does not exist.
#   INTERVAL_S  seconds between snapshots, an integer (default 60, at least 5); 0 writes one snapshot and exits.
#
# Each line holds:
#   time                 local time, ISO 8601 with the UTC offset;
#   mem_total_gb         MemTotal of /proc/meminfo, in GiB;
#   mem_available_gb     MemAvailable, the memory the kernel can give new work without swapping, in GiB;
#   cached_gb            Cached (page cache, which the kernel drops under pressure), in GiB;
#   swap_used_gb         SwapTotal less SwapFree, in GiB;
#   n_procs              this user's processes whose memory could be read;
#   pss_sum_gb           their summed Pss, in GiB: each shared page is split between the processes that map
#                        it, so forked workers sharing one copy of the data are not counted once per worker
#                        (top and htop count it in every worker's RES);
#   top5                 the five largest by Pss, as pid:GiB:name, space-separated.
#
# Reads only /proc and appends only to OUT_CSV. Stop it with Ctrl-C.
# Its command line holds neither "klip_tpe" nor a run's folder name, so KLIP-TPE's liveness scan never counts
# it as one of a run's processes: give OUT_CSV a name without the run's folder name.

set -euo pipefail

OUT="${1:?usage: memlog.sh OUT_CSV [INTERVAL_S]}"
DT="${2:-60}"
if ! [[ "${DT}" =~ ^[0-9]+$ ]] || (( 10#${DT} != 0 && 10#${DT} < 5 )); then
  echo "memlog.sh: INTERVAL_S must be 0 or an integer of at least 5, not '${DT}'" >&2
  exit 2
fi
DT=$((10#${DT}))
[ -r /proc/meminfo ] || { echo "memlog.sh: no /proc/meminfo (Linux only)" >&2; exit 2; }
if [ ! -e "${OUT:?}" ]; then
  echo "time,mem_total_gb,mem_available_gb,cached_gb,swap_used_gb,n_procs,pss_sum_gb,top5" > "${OUT:?}"
fi

snapshot() {
  # One CSV line on standard output (see the header above).
  local t mem procs n sum top d pss name
  t=$(date +%Y-%m-%dT%H:%M:%S%z)
  mem=$(awk '/^MemTotal:/{t=$2} /^MemAvailable:/{a=$2} /^Cached:/{c=$2} /^SwapTotal:/{st=$2} /^SwapFree:/{sf=$2}
             END{printf "%.1f,%.1f,%.1f,%.1f", t/1048576, a/1048576, c/1048576, (st-sf)/1048576}' /proc/meminfo)
  procs=$(
    for d in /proc/[0-9]*; do
      [ -O "${d}" ] || continue                        # this user's processes only
      pss=$(awk '/^Pss:/{print $2; exit}' "${d}/smaps_rollup" 2>/dev/null) || continue
      [ -n "${pss}" ] || continue                      # gone, or a kernel thread
      name=$(tr -d '\0 ,' < "${d}/comm" 2>/dev/null) || name="?"
      echo "${d#/proc/} ${pss} ${name:-?}"
    done | sort -k2,2nr
  )
  n=$(printf '%s\n' "${procs}" | awk 'NF{n++} END{print n+0}')
  sum=$(printf '%s\n' "${procs}" | awk 'NF{s+=$2} END{printf "%.1f", s/1048576}')
  top=$(printf '%s\n' "${procs}" | awk 'NF && NR<=5{printf "%s%s:%.1f:%s", (NR>1?" ":""), $1, $2/1048576, $3}')
  echo "${t},${mem},${n},${sum},${top}"
}

while true; do
  snapshot >> "${OUT:?}"
  (( DT == 0 )) && break
  sleep "${DT}"
done
