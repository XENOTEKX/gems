# Validation — GPU≡CPU bit-identity discipline

The correctness contract: **when the GPU path engages, its log-likelihood equals the CPU path's to the
stated relative error, in FP64, with deterministic reductions.** This document is the rigor centerpiece.
Every relative-error number below cites the job that produced it; the primary logs live in the durable
archive (`/g/data/um09/as1708/gems-provenance/`, hashed in `SHA256SUMS.txt`).

## Relative-error table

| Quantity | GPU vs CPU rel-error | Dataset / regime | Job ID | Log verification |
|---|---|---|---|---|
| JOLT joint-optimizer write-back (lnL) | **2.77e-12** | AA-100K, LG+G4 | 170302036 / ctf100k_v100 | ✅ log: "model=LG+G4 … GPU lnL=-7541976.852146 CPU lnL=-7541976.852167 rel=2.772e-12 PASS" (V100 same-device) |
| Profile-mixture lnL (G.8.0) | **1.56e-16** | LG+MEOW80+G4 | 171604565 | ✅ log: "LG+MEOW80+G4 \| 80 \| 320 \| −373611.897285 \| 1.56e-16" |
| Edge derivative | **~1e-13** | AA | — | 🟡 |
| Pattern tiling: chunked == one-shot | **4.465e-13** | AA-10M | 170976732 | ✅ "AA-10M on 1 H200 rel 4.465e-13" |
| Profile-mixture full real-data | **2.46e-13** | euk LG+MEOW80+G4 | 171733080 | ✅ "GPU lnL −1665670.997 == CPU rel 2.46e-13" |
| +R / +I (ncat ≤ 4) | **≤ 5e-16** | AA/DNA-1M | 172451291 | ✅ log: LG rel 1.395e-16, LG+R2 2.925e-16, LG+R3 2.950e-16 — all "GPU lnL==CPU lnL … PASS" |
| Tree-search screener topology | **RF = 0** (bit-identical NNI set) | AA-1M | 172194079 | ✅ log: "screener vs oracle", `wall_ratio 66.18×` |
| Parsimony score (2D-grid Fitch) | **VERIFY mismatches = 0**; score 15488909 both legs | AA-1M (score/speed) + AA-100K (VERIFY) | 172862242 | ✅ 2× `[GPUPARS-B]` engaged; `VERIFY mismatches=0`; same-node/same-binary speed tie (56.9 vs 56.8 s) |
| Reopt-depth MLE invariance (m∈{4,2} vs m=12) | **\|ΔlnL\| ≤ 0.004 nat** (topology-identical; GPU-vs-GPU across the reopt depth-dial, *not* vs CPU) | AA-100K, DNA-100K (×3 seeds); DNA-1M (×1) | 172882999 / 172883000 / 172883001 | ✅ log: AA \|Δ\|≤1e-4; DNA-100K \|Δ\|≤0.0037 both signs; DNA-1M m≥2 \|Δ\|≤0.0011. **Identity asserted from lnL, not RF** (harness RF column is a parse bug); below floor, DNA-1M m=1 collapses −3139.83 nats → see honest-negatives. |
| Full mixture real-data (LG+MEOW80+G4) | **2.46e-13** | euk, 1 H200 | 171733080 | JOLTMix engaged (172809011 `[JOLTMIX-DBG]`); full rel from 171733080 |
| BEAGLE 4.0 cross-check (lnL) | identical to 1e-9 | AA-100K/1M | 171269929 | ✅ log: tensor==cuda lnL −7118563.14953 |

> **Verification status (2026-07-02):** entries marked ✅ were re-confirmed directly from the harvested logs
> this session. The "trace pending" rows have their numbers from the run tracker / CHANGELOG but their exact
> log lines still need to be anchored into this table before it is cited as the system of record (a number in
> the rigor centerpiece that a reviewer cannot click through to a log line is the worst kind — this is an open
> P5 item).

## Determinism scope (must travel with every "bit-identical" statement)

- **Same-device GPU≡CPU holds to 1M patterns.** Reductions are ordered; on a single GPU model the FP64 result
  is reproducible run-to-run and matches the CPU kernel to the errors above.
- **Across GPU architectures at genome scale (10M), agreement is rel ≤ 5e-11**, because the FP64
  tiling/summation order differs by card (V100 nTile 27 vs H200 28 at 10M). A reproducer on a *different* card
  will not reproduce the low-order bits. State this explicitly; do not claim cross-card bit-identity.
- **FP64 is non-negotiable** for BIC model selection: a wrong low-order bit can flip a BIC tie between two
  models, so the FP32 route-around (designed, unbuilt) is deliberately not used.

## §3f — Result-identity when GPU is off (the upstream merge gate)

The invariant a maintainer requires before accepting an optional GPU module: **built with `-DIQTREE_GPU=OFF`,
the fork's CPU path behaves identically to upstream IQ-TREE 3.1.2.** This holds, verified both empirically and
structurally. The two binaries are **result-identical, not byte-identical** — the correct and stated bar (the
GPU-off build still *compiles* the inert, never-called GPU-dispatch functions, so its md5 differs).

**Empirical (job 172924601, `tests/cpu_off_parity.sh`).** Two CPU-only binaries are built from the same tree —
`main` (= upstream 3.1.2) and `wip-extract` (GPU compiled out) — and run single-threaded (upstream's
`schedule(dynamic,1)` is otherwise low-bit non-deterministic) on the bundled `example.phy` across `-m MF`,
`-m MFP`, and `-m GTR+G4`. The best-fit model, the full BIC score table, **every** scientific line of the
`.iqtree` report, and the exact `.treefile` (topology + branch lengths) are **identical** across all three
modes; the only differing line is the wall-clock timestamp. This exercises the ModelFinder path *and* the
tree-search path (`iqtree.cpp` / `phylotree.*` / `phylotreepars.cpp`). Binaries md5-differ (`282d0bc9…` base
vs `f0b9a10d…` ext) — result-identical, not byte-identical.

**Structural (why it holds).**
- `IQTREE_GPU` is a CMake option **defaulting OFF**; when off no CUDA language is enabled, no `.cu` source is
  compiled, nothing is CUDA-linked, and the `IQTREE_GPU` macro is undefined.
- The GPU kernels (`tree/gpu/*.cu`) and the host integration layer (`tree/phylotreegpu.cpp`, a single
  `#ifdef IQTREE_GPU` → an empty translation unit when off) are absent from the CPU-off build.
- The new runtime flags — `gpu`, `gpu_joint`, `ctf`, `no_gpu` — all **initialise `false`** (`utils/tools.cpp`);
  the new code is entered only through them (e.g. the CTF ModelFinder driver only via
  `if (params.ctf && runCTFModelFinder(...))`, `main/phyloanalysis.cpp`). A default run never enters it.
- The new dispatch code in shared files references **no GPU-only symbol** — GPU acceleration is installed by
  overriding the likelihood-kernel function pointers at the `setLikelihoodKernel` funnel (inside the guarded
  `phylotreegpu.cpp`), so the CPU-off build links cleanly and the CPU funnel is byte-for-byte the upstream path.

**Scope.** Verified on the DNA `example.phy` single-threaded; a broader sweep (AA data, partition models,
`+R`/mixture regimes, multi-thread) is the natural extension. `wip-extract` carries the GPU work only (the
separate `fca-cpu` CPU contributions are not in it), so this isolates the GPU module's effect vs a pristine
3.1.2 base.

## How each proof is run

The runtime cross-checks (`tests/gpu/`, the `--ts-*-check` / `GPU_PARSIMONY_VERIFY` drivers) and the §3f
result-identity job run **only on Gadi as PBS jobs** — GitHub-hosted CI runners have no GPU. CI builds the
CPU-OFF path and *compiles* the GPU path (behind a CUDA-toolkit action); it does **not** execute any GPU kernel
or the result-identity build-and-diff. The GPU work adds guarded code to shared CPU files, so the source is
*not* unchanged vs upstream — what is asserted, on Gadi, is that the CPU-OFF **build result** is identical
(job 172924601). The runtime proofs' logs are the evidence of record and are preserved (hashed) in the durable
archive.
