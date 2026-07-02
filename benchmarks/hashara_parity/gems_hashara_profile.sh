#!/bin/bash
# gems_hashara_profile.sh — nsys PROFILE pass (SEPARATE from the clean-wall sweep, which it must never
# contaminate). One job = one (TOOL, TYPE, SCALE, PHASE). Answers "where does the time go" for
# JOLT vs Hashara's naive OpenACC: GPU kernel time, H2D/D2H transfer time/bytes, CUDA-API/host time,
# and GPU-active fraction of wall. nsys traces CUDA kernels from BOTH (her OpenACC lowers to CUDA),
# so the kernel breakdowns are directly comparable.
# NOTE: the wall time reported HERE is nsys-contaminated — the clean speedup lives in the sweep.
#PBS -N gems-hprof
#PBS -P dx61
#PBS -q gpuhopper
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l storage=scratch/dx61+scratch/rc29+gdata/um09
#PBS -l wd
#PBS -j oe
set -uo pipefail
: "${TOOL:?jolt|naive}"; : "${TYPE:?AA|DNA}"; : "${SCALE:?}"; : "${PHASE:?mf|ts}"
module load cuda/12.5.1 2>/dev/null || true
NSYS=/apps/cuda/12.5.1/bin/nsys
JOLT=/scratch/rc29/as1708/iqtree3-gpu/build-gpu-on/iqtree3
NAIVE=/scratch/rc29/as1708/gems-verify/bin/iqtree3-hashara-naive-h200
if [ "$TYPE" = AA ]; then SUB='LG+I+G4'; TSMODEL='LG+G4'; else SUB='GTR+I+G4'; TSMODEL='GTR+G4'; fi
ALN=/scratch/rc29/as1708/datasets/complex_data_shared/$TYPE/$SUB/taxa_100/len_$SCALE/tree_1/alignment_$SCALE.phy

if [ "$TOOL" = jolt ]; then
  BIN=$JOLT; EXP=2c931f41
  [ "$PHASE" = mf ] && ARGS=(-m MF --ctf) || ARGS=(-m "$TSMODEL" --ts-fused)
else
  BIN=$NAIVE; EXP=e713866b
  module load openmpi/4.1.5 boost/1.84.0 nvhpc-compilers/24.7 gcc/12.2.0 2>/dev/null || true
  [ "$PHASE" = mf ] && ARGS=(-m MF) || ARGS=(-m "$TSMODEL")
fi
got=$(md5sum "$BIN"|cut -c1-8); [ "$got" = "$EXP" ] || { echo "GUARD FAIL md5 $got!=$EXP"; exit 2; }
[ -f "$ALN" ] || { echo "GUARD FAIL missing $ALN"; exit 2; }

OUT=/g/data/um09/as1708/gems-provenance/hashara_profile; mkdir -p "$OUT"
WB=/scratch/rc29/as1708/gems-verify/hprof/${TYPE}_${SCALE}_${PHASE}_${TOOL}_${PBS_JOBID%%.*}; mkdir -p "$WB"; cd "$WB"
REP="$WB/prof"
echo "════ PROFILE TOOL=$TOOL TYPE=$TYPE SCALE=$SCALE PHASE=$PHASE ════ $(date -Iseconds) $(hostname)"
echo "bin md5=$got  GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
echo "cmd: $BIN -s ALN ${ARGS[*]} -T 12 --seed 12345"

# low-overhead trace: cuda kernels+memops + openacc constructs + os-runtime (host threads). No CPU backtrace sampling.
"$NSYS" profile --trace=cuda,openacc,osrt --sample=none --cpuctxsw=none --force-overwrite=true \
   -o "$REP" \
   "$BIN" -s "$ALN" "${ARGS[@]}" -T 12 --seed 12345 --prefix "$WB/run" -redo > "$WB/run.stdout" 2>&1
RC=$?; echo "  run exit=$RC"

# ---- extract comparable summaries ----
"$NSYS" stats --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum \
              --report cuda_gpu_mem_size_sum --report cuda_api_sum \
              --format table "$REP.nsys-rep" > "$WB/stats.txt" 2>&1 || echo "(stats parse warn)"

# headline numbers (best-effort parse of the _sum tables)
KTIME=$(grep -aA4 'cuda_gpu_kern_sum' "$WB/stats.txt" 2>/dev/null | grep -aoE 'Total Time.*' | head -1)
echo; echo "──── nsys summary (full table → stats.txt) ────"
echo "TOP GPU KERNELS:"; awk '/cuda_gpu_kern_sum/{f=1} f&&/%/{p=1} p&&NF{print} /cuda_gpu_mem_time_sum/{f=0;p=0}' "$WB/stats.txt" 2>/dev/null | head -12
echo; echo "GPU MEM TRANSFER (time):"; awk '/cuda_gpu_mem_time_sum/{f=1} f{print} /cuda_gpu_mem_size_sum/{f=0}' "$WB/stats.txt" 2>/dev/null | head -8

cp -p "$WB/stats.txt" "$OUT/${TYPE}_${SCALE}_${PHASE}_${TOOL}.stats.txt" 2>/dev/null
cp -p "$REP.nsys-rep" "$OUT/${TYPE}_${SCALE}_${PHASE}_${TOOL}.nsys-rep" 2>/dev/null || echo "(rep copy deferred — large)"
echo; echo "recorded → $OUT/${TYPE}_${SCALE}_${PHASE}_${TOOL}.stats.txt"
echo "════ DONE $(date -Iseconds) ════"
