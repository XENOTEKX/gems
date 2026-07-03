#!/bin/bash
# GEMS P1e2 — the §3f invariant gate, RE-VALIDATED on current HEAD + BROADENED.
# Builds two CPU-only (IQTREE_GPU=OFF) binaries — main (=upstream IQ-TREE 3.1.2) and wip-extract
# (GPU compiled OUT) — and asserts the CPU path is RESULT-IDENTICAL across a broadened panel:
#   MF  (model selection)         -> compare best-fit model + BIC score table
#   MFP (model selection + tree)  -> compare model + full .treefile (topology+brlens) + normalised report
#   GTR+G4 (fixed model, search)  -> compare .treefile + normalised report + final lnL  [exercises the
#                                     de-jargon'd tree-search path: iqtree.cpp / phylotree.* / phylotreepars.cpp]
# Single-threaded (-nt 1) for determinism (upstream's schedule(dynamic,1) is otherwise low-bit non-deterministic).
# A DIFFERENCE means a shared-file edit changed the CPU path when GPU is off (§3f violation / SEV-9 path).
#PBS -N gems_p1e2
#PBS -P dx61
#PBS -q normal
#PBS -l ncpus=16
#PBS -l mem=64GB
#PBS -l jobfs=20GB
#PBS -l walltime=02:30:00
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
set -uo pipefail
module load cmake/3.24.2 gcc/12.2.0 eigen/3.3.7 boost/1.84.0

GEMS="${GEMS:-$(cd "$(dirname "$0")/.." && pwd)}"
WORK=/scratch/rc29/as1708/gems-verify/p1e2_${PBS_JOBID%%.*}
mkdir -p "$WORK"
echo "════ GEMS P1e2: §3f CPU-OFF result-identity (upstream 3.1.2 vs wip-extract @ $(git -C "$GEMS" rev-parse --short HEAD)) ════ $(date -Iseconds) $(hostname)"

prep_src() {  # $1=ref $2=dest
  rm -rf "$2"; mkdir -p "$2"
  git -C "$GEMS" archive "$1" | tar -x -C "$2"
  cp -a "$GEMS/cmaple" "$2/cmaple"; cp -a "$GEMS/lsd2" "$2/lsd2"
  rm -rf "$2/cmaple/.git" "$2/lsd2/.git"
}
build_cpu() {  # $1=src $2=build
  cmake -S "$1" -B "$2" -DIQTREE_GPU=OFF -DCMAKE_BUILD_TYPE=Release \
        -DEIGEN3_INCLUDE_DIR=/apps/eigen/3.3.7/include/eigen3 \
        -DUSE_CMAPLE=OFF -DUSE_LSD2=OFF >"$2.cmake.log" 2>&1
  cmake --build "$2" -j >"$2.make.log" 2>&1
}

echo "── build reference (main = upstream 3.1.2), CPU OFF ──"
prep_src main        "$WORK/src-base"; build_cpu "$WORK/src-base" "$WORK/b-base"
echo "── build extract (wip-extract), CPU OFF ──"
prep_src wip-extract "$WORK/src-ext";  build_cpu "$WORK/src-ext"  "$WORK/b-ext"

BASE=$WORK/b-base/iqtree3; EXT=$WORK/b-ext/iqtree3
[ -x "$BASE" ] && [ -x "$EXT" ] || { echo "❌ BUILD FAILED (GPU-off must link — a direct GPU-symbol ref in an unguarded shared file would break here)";
  echo "--- ext cmake tail ---"; tail -15 "$WORK/b-ext.cmake.log"; echo "--- ext make tail ---"; tail -30 "$WORK/b-ext.make.log"; exit 1; }
echo "✅ both GPU-off binaries built + linked. base md5=$(md5sum "$BASE"|cut -d' ' -f1)  ext md5=$(md5sum "$EXT"|cut -d' ' -f1)"

cp "$GEMS/example/example.phy" "$WORK/"; ALN=$WORK/example.phy

run() { ( cd "$WORK" && "$1" -s "$ALN" -seed 1 -nt 1 -pre "$2" -redo $3 >"$2.console" 2>&1 ); }
# deterministic model-selection slice (winners + BIC-sorted table)
mf_slice() { grep -aE 'Best-fit model according to (BIC|AIC|AICc):' "$WORK/$1.iqtree"
  awk '/List of models sorted by BIC/{p=1} p{print} /^$/{if(p>1)exit; if(p)p++}' "$WORK/$1.iqtree"; }
# normalise a full .iqtree report: drop environment/volatile lines, keep all scientific content
norm_report() { grep -avE 'Date and [Tt]ime|CPU time|wall-clock|Total (CPU|wall)|Host:|Command:|Seed:|seconds|^Time|IQ-TREE( multicore)? version|Kernel|^NOTE:|-pre |files:|written to|\.iqtree|\.treefile|\.log|\.ckp|version [0-9]' "$WORK/$1.iqtree"; }

cmp_pair() {  # $1=mode-tag $2=produces_tree(0/1)
  local m="$1" tree="$2" ok=1
  mf_slice "base_$m" >"$WORK/s_base_$m"; mf_slice "ext_$m" >"$WORK/s_ext_$m"
  diff -q "$WORK/s_base_$m" "$WORK/s_ext_$m" >/dev/null || { ok=0; echo "  ❌ [$m] model-selection slice DIFFERS:"; diff "$WORK/s_base_$m" "$WORK/s_ext_$m"|head; }
  norm_report "base_$m" >"$WORK/n_base_$m"; norm_report "ext_$m" >"$WORK/n_ext_$m"
  diff -q "$WORK/n_base_$m" "$WORK/n_ext_$m" >/dev/null || { ok=0; echo "  ❌ [$m] normalised .iqtree report DIFFERS:"; diff "$WORK/n_base_$m" "$WORK/n_ext_$m"|head -20; }
  if [ "$tree" = 1 ]; then
    if diff -q "$WORK/base_$m.treefile" "$WORK/ext_$m.treefile" >/dev/null; then :; else
      ok=0; echo "  ❌ [$m] .treefile DIFFERS (topology or branch lengths):"; diff "$WORK/base_$m.treefile" "$WORK/ext_$m.treefile"|head; fi
  fi
  [ $ok = 1 ] && echo "  ✅ [$m] result-identical (model + report$([ "$tree" = 1 ] && echo ' + treefile'))"
  return $((1-ok))
}

echo; echo "──────── §3f result-identity panel (single-thread; base must == ext) ────────"
PASS=1
declare -A MODES=( [MF]="-m MF|0" [MFP]="-m MFP|1" [GTRG]="-m GTR+G4|1" )
for tag in MF MFP GTRG; do
  IFS='|' read -r args tree <<< "${MODES[$tag]}"
  run "$BASE" "base_$tag" "$args"; run "$EXT" "ext_$tag" "$args"
  echo "── [$tag] $args ──"; grep -aE 'BEST SCORE FOUND|Best-fit model|Log-likelihood of the tree' "$WORK/ext_$tag.iqtree" 2>/dev/null | sed 's/^/    ext: /' | head -3
  cmp_pair "$tag" "$tree" || PASS=0
done

echo
if [ $PASS -eq 1 ]; then
  echo "✅✅ P1e2 PASS: CPU-OFF extract is result-identical to upstream 3.1.2 across MF/MFP/GTR+G4 (model+report+treefile). §3f invariant holds on $(git -C "$GEMS" rev-parse --short HEAD)."
else
  echo "❌ P1e2 FAIL: a shared-file edit shifts the CPU result when GPU off — pinpoint + guard/revert (SEV-9)."
fi
echo "════ artifacts → $WORK ════"
if [ $PASS -eq 1 ]; then rm -rf "$WORK/src-base" "$WORK/src-ext" "$WORK/b-base" "$WORK/b-ext"; echo "🧹 cleaned build+src trees (PASS)"; fi
