# Honest negatives & scope limits

These are included **on purpose**. They are the project's strongest evidence of scientific rigor: every
result below is something that was tried, measured, and found *not* to support a claim — so the claim is
not made. Each has a job ID. Reviewers and examiners should read this alongside the headline table.

## Claims we explicitly do NOT make

| Non-claim | Why (measured) | Evidence |
|---|---|---|
| "1 GPU beats a 16-node cluster" (throughput/bandwidth) | The bandwidth thesis was **falsified**; single-GPU full-data 100K ≈ **one CPU node** (0.88×). The real, defensible framing is that CTF beats the *tool's breadth problem*; the cluster's measured np16 parallel efficiency is only **28.5%**. | 170398260 |
| AA-1M / AA-10M as a **throughput speedup** | Pattern tiling is a **capability** (it runs where the CPU OOMs), not a proven bandwidth win. Present as feasibility, never as a speedup multiplier. | 170976732 |
| A finished **avian TENT tree** | The GTR+G search hit the 4 h wall at iter ~70 with no final tree; the GTR+F+I+R2 result is a **model-selection + parameter fit on a fixed/partial topology**, not a full search. Cite as motivation only. | 172437363 / 172810407 |
| Profile mixtures as **production** | Validated but **env-gated** (`JOLT_MIX_HOSTDRIVEN`); the aggregate-throughput story that would justify default-on is unwritten. | 171733080 |
| CTF "+I single-start ~2.7× over np16" | **Projected, not measured.** Cite only the measured 893 s / 1.26×. | 170581208 |
| Tree-search **66× as end-to-end search** | It is a **clean-room screener primitive**. The integrated `--ts-fused` search is ≈ one CPU node and ~1.9× behind a competing OpenACC GPU at AA-1M. Keep screener-vs-search sharp. | 172194079 |
| Parsimony **"71×"** | The 4033 s figure was a **PLL artifact**; IQ-TREE's own IQ-parsimony already beats PLL ~16–19× on CPU alone (a start-tree library swap, quality-identical). The genuine GPU-over-best-CPU win is **3.76×**. | 172524833 / 172526962 |

## Experiments run and shelved (the negatives appendix)

- **Bandwidth thesis falsified** — the `k1_node` kernel is memory-*latency*-bound at register-capped ~49%
  occupancy (DRAM ~34 % of peak, L2 13–19 %, L1 14 %, math ~0.1 %), **not** bandwidth-saturated. So the
  "saturate the bus" story does not hold. (170398260; reproduced 172813160 / 172813725.)
- **N/S breadth ceiling** — the GPU model-evaluation loop is mutex-serialized; screener fold-batch hit an
  L2 occupancy trap. (172635734.)
- **FP64 tensor-core MMA** — tested and closed: 1.6–2.8× *slower* than JOLT's scalar FP64 path for this
  access pattern.
- **async/streams ladder** — scale-null (no speedup across scales). (172609386.)
- **L-BFGS reopt** — converges *worse* than the diagonal-LM default (mean ~24 iters/reject vs 12.9);
  shelved, kept behind an off flag as insurance. (172432710.)
- **Rake batched-fold** — bit-identical but only 1.04× (compute-bound ceiling). (172635734.)
- **incremental / dirty partials** — ruled out (diagonal-LM dirties all branches).
- **site-subsample for tree search** — refuted (the projected gate gave 0/15 recall vs the native gate's 30/30).
- **Cook–Mertz tree-evaluation** — non-transferable to this problem.
- **FP32 route-around** — designed but unbuilt; FP64 parity is non-negotiable for BIC ties, so this stays a
  design note, not a result.

## Scope caveats that must travel with the numbers

- **"Bit-identical" is scoped.** Same-device GPU≡CPU holds up to **1M patterns**. Across GPU *architectures*
  at genome scale the agreement is **rel ≤ 5e-11** (V100/A100/H200 differ because FP64 tiling/summation order
  differs: V100 nTile 27 vs H200 28 at 10M). A reproducer on a different card will **not** reproduce the bits.
- **`+R` is validated for ncat ≤ 4.** Higher rate-category counts decline to CPU.
- **The CPU engine is upstream IQ-TREE's.** This project's contribution is the `tree/gpu/` module + the
  integration seams; "fork" does not claim the whole tree.
