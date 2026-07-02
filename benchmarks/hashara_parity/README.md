# GPU-vs-GPU parity — GEMS/JOLT vs a naive OpenACC port

A head-to-head between two GPU accelerations of IQ-TREE on the **same hardware (H200), same data, same
seed**: this work (GEMS/JOLT — in-tree CUDA kernels + coarse-to-fine ModelFinder) versus a contemporaneous
**naive OpenACC** port (accelerates the Felsenstein kernel directly). Both accelerate the same two phases,
so both are compared on both:

- **① ModelFinder** — `-m MF`. GEMS adds `--ctf` (rank all candidates on a subsample, refine the top-k);
  the naive port evaluates every candidate fully on the GPU.
- **② Tree search** — fixed model (`-m LG+G4` / `GTR+G4`), clean (no bootstrap). GEMS adds `--ts-fused`.

## Why this comparison is fair

- **Same data:** all 8 alignments (AA+DNA × 10K/100K/1M/10M) are **byte-identical** between the two source
  trees (md5-checked); the driver guards on md5.
- **Same protocol:** identical `-T 12 --seed 12345`; each binary is md5-pinned in the driver.
- **Clean separation of timing and profiling:** the wall-clock sweep (`gems_hashara_parity.sh`) carries **no
  profiler** so the speedup is uncontaminated; a **separate** nsys pass (`gems_hashara_profile.sh`) explains
  *where the time goes* (GPU kernel time, H2D/D2H transfer, host time, GPU-active fraction). nsys traces CUDA
  kernels from both (OpenACC lowers to CUDA), so the breakdowns are directly comparable.

## Parity gates (in the aggregator)

- ① identical best-fit model **and** matching winner lnL.
- ② identical BEST SCORE (< 2 nat).
- Speedup = naive_wall / jolt_wall per cell (> 1 ⇒ GEMS faster).

## Reproduce

```bash
# clean-wall sweep (one tier or all): submit {jolt,naive}×{AA,DNA} for the scale(s)
bash submit_hashara_parity.sh 10000 100000 1000000 10000000
# aggregate the per-cell TSVs into the parity table + speedups
python3 aggregate_hashara_parity.py
# where-time-goes profile (separate, nsys):  -v TOOL=,TYPE=,SCALE=,PHASE=
qsub -v TOOL=jolt,TYPE=AA,SCALE=1000000,PHASE=mf gems_hashara_profile.sh
```

Binaries (md5-pinned): GEMS/JOLT `2c931f41`; naive OpenACC (H200, cc90) `e713866b`.
Durable cells + logs: `/g/data/um09/as1708/gems-provenance/hashara_parity/`.

## Status (2026-07-02) — the honest crossover

Verified first cell, **AA-10K**: parity holds exactly (both pick LG+G4, ΔlnL = 0.000; tree-search BEST SCORE
Δ = 0.001 nat). Speedups **below 1× at 10K** (MF 0.69×, TS 0.38×) — expected and correct: the GPU loses on
tiny data, and `--ctf`'s subsample step is pure overhead when the subsample is ~half the data. CTF pays off
only when the subsample is a small fraction, i.e. at 1M/10M. The sweep captures that crossover; **do not
quote a small-scale cell as the result.** 100K / 1M / 10M tiers in flight.
