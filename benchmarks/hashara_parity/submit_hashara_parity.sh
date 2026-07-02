#!/bin/bash
# submit_hashara_parity.sh [SCALE ...] — submit JOLT-vs-naive parity jobs for the given scale(s).
# No arg = all four scales. Each scale submits 4 jobs: {jolt,naive} × {AA,DNA}.
# Per-scale walltime/mem/jobfs sized for the SLOW side (naive MF at 10M ≈ 12h).
set -uo pipefail
DRV=/home/272/as1708/setonix-iq/gadi-ci/gems/gems_hashara_parity.sh
SCALES=("$@"); [ ${#SCALES[@]} -eq 0 ] && SCALES=(10000 100000 1000000 10000000)

res() { case "$1" in           #  walltime      mem      jobfs
  10000)     echo "00:30:00    90GB     20GB" ;;
  100000)    echo "01:30:00    90GB     30GB" ;;
  1000000)   echo "08:00:00   180GB     80GB" ;;
  10000000)  echo "36:00:00   180GB    120GB" ;;
  *) echo "" ;; esac; }

for SCALE in "${SCALES[@]}"; do
  read -r WT MEM JFS <<<"$(res "$SCALE")"
  [ -z "${WT:-}" ] && { echo "skip unknown scale $SCALE"; continue; }
  for TOOL in jolt naive; do
    for TYPE in AA DNA; do
      jid=$(qsub -N "hpar-${TOOL:0:1}${TYPE}${SCALE}" \
                 -l "walltime=$WT" -l "mem=$MEM" -l "jobfs=$JFS" \
                 -v "TOOL=$TOOL,TYPE=$TYPE,SCALE=$SCALE" "$DRV" 2>&1)
      printf "  %-6s %-3s %-9s wt=%s mem=%s → %s\n" "$TOOL" "$TYPE" "$SCALE" "$WT" "$MEM" "$jid"
    done
  done
done
echo "submitted. results land in /g/data/um09/as1708/gems-provenance/hashara_parity/<TYPE>_<SCALE>_<TOOL>.tsv"
