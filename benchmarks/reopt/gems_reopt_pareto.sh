#!/bin/bash
# gems_reopt_pareto.sh — the GPU tree-search REOPT-DEPTH Pareto sweep, MULTI-SEED, on the FROZEN thesis binary.
# Sweeps the per-round JOLT branch-length reopt iteration count (env JOLT_BRLEN_MAXITER) over {1,2,4,12} for the
# --ts-fused GPU search, at multiple seeds, and reports final lnL, RF-vs-maxiter12, and wall per cell. This turns
# the "how few reopt iterations before quality degrades?" lever into a durable, multi-seed Pareto figure and
# CLOSES the seed-1-only gap in the existing single-seed sweep (job 172400909). Canonical binary 8cc3cb84 defaults
# to maxiter=2 and exposes the env knob; the OLD dev tree (2c931f41) hardcodes 12 with NO knob — must NOT be used.
#
# Submit AA: qsub -q gpuhopper -lngpus=1 -lncpus=12 -lmem=90GB -lwalltime=05:00:00 -v TYPE=AA,SCALE=100000,SEEDS="1 2 3" gems_reopt_pareto.sh
# Submit DNA:qsub -q gpuhopper -lngpus=1 -lncpus=12 -lmem=90GB -lwalltime=02:30:00 -v TYPE=DNA,SCALE=100000,SEEDS="1 2 3" gems_reopt_pareto.sh
#PBS -N gems_reopt
#PBS -P dx61
#PBS -l storage=scratch/dx61+scratch/rc29+gdata/um09
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cuda/12.5.1 2>/dev/null || true
module load eigen/3.3.7 2>/dev/null || true
module load boost/1.84.0 2>/dev/null || true
export LD_LIBRARY_PATH="${CUDA_HOME:-/apps/cuda/12.5.1}/lib64:${LD_LIBRARY_PATH:-}"

TYPE="${TYPE:-AA}"; SCALE="${SCALE:-100000}"; SEEDS="${SEEDS:-1 2 3}"; MAXITERS="${MAXITERS:-12 4 2 1}"
NT="${PBS_NCPUS:-12}"
BIN=/g/data/um09/as1708/gems-bin/iqtree3-thesis-sweep-8cc3cb84
EXPECT_MD5=8cc3cb844e807fe3ecbcd0b9ff5428df
[ -x "$BIN" ] || { echo "FATAL: frozen binary missing"; exit 2; }
[ "$(md5sum "$BIN"|cut -d' ' -f1)" = "$EXPECT_MD5" ] || { echo "FATAL: md5 gate FAILED"; exit 2; }
# knob presence guard (the whole experiment is void without it)
grep -aq JOLT_BRLEN_MAXITER "$BIN" || { echo "FATAL: binary lacks JOLT_BRLEN_MAXITER knob (wrong tree)"; exit 2; }

[ "$TYPE" = AA ] && { SUB=LG+I+G4; MODEL="${MODEL:-LG+G4}"; } || { SUB=GTR+I+G4; MODEL="${MODEL:-GTR+G4}"; }
# MODEL override (e.g. MODEL=LG+R4) lets this probe the rate-heterogeneous "hard regime" the red-team flagged.
BASE=/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared
ALN="$BASE/$TYPE/$SUB/taxa_100/len_$SCALE/tree_1/alignment_$SCALE.phy"
[ -f "$ALN" ] || { echo "FATAL: missing aln $ALN"; exit 2; }

OUT="/scratch/rc29/as1708/gems_reopt_${TYPE}_${SCALE}_${PBS_JOBID:-local}"; mkdir -p "$OUT"; cd "$OUT"
HARVEST="/g/data/um09/as1708/gems-provenance/reproductions/reopt_pareto_${TYPE}_${SCALE}_${PBS_JOBID:-local}"
TSV="$OUT/pareto.tsv"; printf "type\tscale\tseed\tmaxiter\tfinal_lnL\twall_s\tengaged\n" > "$TSV"
echo "════ REOPT PARETO $TYPE-$SCALE ($MODEL) — $(hostname) $(date -Iseconds) — bin 8cc3cb84 (md5 gate PASS) ════"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

run_cell () {
  local SEED="$1" M="$2" pre="cell_${TYPE}_${SCALE}_s${SEED}_m${M}"
  local T0=$(date +%s)
  JOLT_BRLEN_MAXITER="$M" JOLT_DEBUG=1 "$BIN" -s "$ALN" -m "$MODEL" -seed "$SEED" -nt "$NT" \
      --jolt --gpu --ts-fused -pre "$pre" -redo > "$pre.stdout" 2>&1
  local WALL=$(($(date +%s)-T0))
  local LNL=$(grep -iE "BEST SCORE FOUND|Log-likelihood of the tree" "$pre.iqtree" 2>/dev/null | grep -oE '\-[0-9]+\.[0-9]+' | head -1)
  [ -z "$LNL" ] && LNL=$(grep -iE "BEST SCORE FOUND|Log-likelihood of the tree" "$pre.stdout" 2>/dev/null | grep -oE '\-[0-9]+\.[0-9]+' | head -1)
  local ENG=$(grep -c '\[JOLT\]' "$pre.stdout" 2>/dev/null); ENG="${ENG:-0}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$TYPE" "$SCALE" "$SEED" "$M" "${LNL:-NA}" "$WALL" "$ENG" >> "$TSV"
  echo "  seed=$SEED maxiter=$M : lnL=${LNL:-NA} wall=${WALL}s engage=$ENG"
}

for SEED in $SEEDS; do
  echo "──── seed $SEED ────"
  for M in $MAXITERS; do run_cell "$SEED" "$M"; done
  # RF of each maxiter tree vs the seed-matched maxiter=12 reference (topology-identity check)
  REF="cell_${TYPE}_${SCALE}_s${SEED}_m12.treefile"
  if [ -f "$REF" ]; then
    for M in $MAXITERS; do
      T="cell_${TYPE}_${SCALE}_s${SEED}_m${M}.treefile"
      [ -f "$T" ] || continue
      "$BIN" -rf "$REF" "$T" -pre "rf_s${SEED}_m${M}" -redo > "rf_s${SEED}_m${M}.log" 2>&1 || true
      RF=$(grep -iE "Robinson-Foulds distance|RF distance" "rf_s${SEED}_m${M}.rfdist" "rf_s${SEED}_m${M}.log" 2>/dev/null | grep -oE '[0-9]+' | tail -1)
      echo "    RF(seed$SEED, m$M vs m12) = ${RF:-?}"
    done
  fi
done

echo; echo "════ PARETO TABLE ════"; column -t -s$'\t' "$TSV"
# per-seed dlnL vs maxiter=12 + speedup
python3 - "$TSV" <<'PY'
import sys,csv,collections
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
by=collections.defaultdict(dict)
for r in rows:
    try: by[r['seed']][int(r['maxiter'])]=(float(r['final_lnL']), float(r['wall_s']))
    except: pass
print("\nseed  maxiter  dlnL_vs_m12   wall_s   speedup_vs_m12")
for s in sorted(by):
    ref=by[s].get(12)
    for m in sorted(by[s],reverse=True):
        lnl,w=by[s][m]
        dl = lnl-ref[0] if ref else float('nan')
        sp = ref[1]/w if ref and w else float('nan')
        print(f"{s:>4}  {m:>7}  {dl:>+11.5f}   {w:>6.0f}   {sp:>6.2f}x")
PY

mkdir -p "$HARVEST"; cp -p "$TSV" "$OUT"/*.stdout "$OUT"/*.iqtree "$OUT"/*.treefile "$HARVEST/" 2>/dev/null
( cd "$HARVEST" && sha256sum pareto.tsv *.iqtree 2>/dev/null > SHA256SUMS.txt )
echo "harvested -> $HARVEST"
