#!/bin/bash
# gems_hashara_parity.sh — GPU-vs-GPU parity: JOLT (ours) vs Hashara's naive OpenACC port.
# ONE job = one (TOOL, TYPE, SCALE); runs BOTH comparisons on ONE H200:
#   ① ModelFinder : -m MF   (JOLT adds --ctf = subsample-rank+refine; naive = full per-model GPU evals)
#   ② Tree search : -m LG+G4 / GTR+G4, clean (NO bootstrap; user choice), JOLT adds --ts-fused
# Same alignment (byte-identical across both trees), same seed 12345, same -T 12.
# Parity is checked in the aggregator: ① identical best-fit model + winner lnL; ② identical BEST SCORE.
# Each binary is run under ITS OWN module env (JOLT=CUDA, naive=nvhpc/OpenACC) — that is why tools are
# separate jobs, not back-to-back in one process env.
#PBS -N gems-hpar
#PBS -P dx61
#PBS -q gpuhopper
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l storage=scratch/dx61+scratch/rc29+gdata/um09
#PBS -l wd
#PBS -j oe
set -uo pipefail
: "${TOOL:?set -v TOOL=jolt|naive}"; : "${TYPE:?set -v TYPE=AA|DNA}"; : "${SCALE:?set -v SCALE=10000|100000|1000000|10000000}"

JOLT=/scratch/rc29/as1708/iqtree3-gpu/build-gpu-on/iqtree3               # md5 2c931f41
NAIVE=/scratch/rc29/as1708/gems-verify/bin/iqtree3-hashara-naive-h200    # md5 e713866b (pinned copy of her build-gpu-cc90)

if [ "$TYPE" = AA ]; then SUB='LG+I+G4'; TSMODEL='LG+G4'; else SUB='GTR+I+G4'; TSMODEL='GTR+G4'; fi
ALN=/scratch/rc29/as1708/datasets/complex_data_shared/$TYPE/$SUB/taxa_100/len_$SCALE/tree_1/alignment_$SCALE.phy

if [ "$TOOL" = jolt ]; then
  BIN=$JOLT; MF_EXTRA="--ctf"; TS_EXTRA="--ts-fused"; EXP_MD5=2c931f41
  module load cuda/12.5.1 2>/dev/null || true
elif [ "$TOOL" = naive ]; then
  BIN=$NAIVE; MF_EXTRA=""; TS_EXTRA=""; EXP_MD5=e713866b
  module load openmpi/4.1.5 boost/1.84.0 nvhpc-compilers/24.7 cuda/12.5.1 gcc/12.2.0 2>/dev/null || true
else echo "bad TOOL=$TOOL"; exit 2; fi

# ---- guards: identical inputs + pinned binary are the whole point ----
got=$(md5sum "$BIN" 2>/dev/null | cut -c1-8)
[ "$got" = "$EXP_MD5" ] || { echo "GUARD FAIL: $BIN md5 got=$got want=$EXP_MD5"; exit 2; }
[ -f "$ALN" ] || { echo "GUARD FAIL: missing alignment $ALN"; exit 2; }

OUT=/g/data/um09/as1708/gems-provenance/hashara_parity; mkdir -p "$OUT"
WB=/scratch/rc29/as1708/gems-verify/hpar/${TYPE}_${SCALE}_${TOOL}_${PBS_JOBID%%.*}; mkdir -p "$WB"; cd "$WB"
echo "════ HASHARA PARITY  TOOL=$TOOL TYPE=$TYPE SCALE=$SCALE ════ $(date -Iseconds) $(hostname)"
echo "bin=$BIN md5=$got  ALN md5=$(md5sum "$ALN"|cut -c1-8)  GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"

# ---- ① ModelFinder (stops after model selection; -m MF is MF-only) ----
echo; echo "──── ① MF : $(basename "$BIN") -s ALN -m MF $MF_EXTRA -T 12 --seed 12345 ────"
/usr/bin/time -v "$BIN" -s "$ALN" -m MF $MF_EXTRA -T 12 --seed 12345 --prefix "$WB/mf" -redo > "$WB/mf.stdout" 2> "$WB/mf.time"
MF_RC=$?
MF_MODEL=$(grep -aoE 'Best-fit model according to BIC: .+' "$WB/mf.iqtree" 2>/dev/null | sed 's/.*BIC: //' | head -1)
[ -z "$MF_MODEL" ] && MF_MODEL=$(grep -aoE 'Best-fit model: [^ ]+' "$WB/mf.log" 2>/dev/null | awk '{print $NF}' | head -1)
MF_WALL=$(grep -aoE 'Total wall-clock time used: [0-9.]+' "$WB/mf.log" 2>/dev/null | grep -oE '[0-9.]+' | head -1)
MF_LNL=$(grep -aoE 'BEST SCORE FOUND : -?[0-9.]+' "$WB/mf.log" 2>/dev/null | grep -oE '\-?[0-9.]+' | head -1)
MF_KERN=$(grep -aoE 'Kernel:.*' "$WB/mf.log" 2>/dev/null | head -1)
echo "  MF exit=$MF_RC  model=${MF_MODEL:-NA}  wall=${MF_WALL:-NA}s  lnL=${MF_LNL:-NA}  [${MF_KERN:-nokern}]"

# ---- ② Tree search (clean, no bootstrap) ----
echo; echo "──── ② TS : $(basename "$BIN") -s ALN -m $TSMODEL $TS_EXTRA -T 12 --seed 12345 ────"
/usr/bin/time -v "$BIN" -s "$ALN" -m "$TSMODEL" $TS_EXTRA -T 12 --seed 12345 --prefix "$WB/ts" -redo > "$WB/ts.stdout" 2> "$WB/ts.time"
TS_RC=$?
TS_WALL=$(grep -aoE 'Wall-clock time used for tree search: [0-9.]+' "$WB/ts.log" 2>/dev/null | grep -oE '[0-9.]+' | head -1)
TS_TOT=$(grep -aoE 'Total wall-clock time used: [0-9.]+' "$WB/ts.log" 2>/dev/null | grep -oE '[0-9.]+' | head -1)
TS_SCORE=$(grep -aoE 'BEST SCORE FOUND : -?[0-9.]+' "$WB/ts.log" 2>/dev/null | grep -oE '\-?[0-9.]+' | head -1)
TS_KERN=$(grep -aoE 'Kernel:.*' "$WB/ts.log" 2>/dev/null | head -1)
echo "  TS exit=$TS_RC  [${TS_KERN:-nokern}]  ts_wall=${TS_WALL:-NA}s  total=${TS_TOT:-NA}s  BEST_SCORE=${TS_SCORE:-NA}"

# ---- record one durable cell file (aggregator reads these) ----
REC="$OUT/${TYPE}_${SCALE}_${TOOL}.tsv"
{
  printf "tool\t%s\n" "$TOOL";        printf "type\t%s\n" "$TYPE";        printf "scale\t%s\n" "$SCALE"
  printf "bin_md5\t%s\n" "$got";      printf "aln_md5\t%s\n" "$(md5sum "$ALN"|cut -c1-8)"
  printf "mf_model\t%s\n" "${MF_MODEL:-NA}"; printf "mf_wall\t%s\n" "${MF_WALL:-NA}"; printf "mf_lnl\t%s\n" "${MF_LNL:-NA}"; printf "mf_rc\t%s\n" "$MF_RC"; printf "mf_kern\t%s\n" "${MF_KERN:-NA}"
  printf "ts_wall\t%s\n" "${TS_WALL:-NA}"; printf "ts_total\t%s\n" "${TS_TOT:-NA}"; printf "ts_score\t%s\n" "${TS_SCORE:-NA}"; printf "ts_rc\t%s\n" "$TS_RC"; printf "ts_kern\t%s\n" "${TS_KERN:-NA}"
  printf "jobid\t%s\n" "${PBS_JOBID%%.*}"; printf "when\t%s\n" "$(date -Iseconds)"
} > "$REC"
echo; echo "recorded → $REC"
echo "════ DONE $(date -Iseconds) ════"
