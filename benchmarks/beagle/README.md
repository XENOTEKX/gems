# Figure 6 — JOLT vs BEAGLE 4.0 (matched kernel micro-benchmark)

## What is and is NOT claimed

BEAGLE is a **likelihood library**, not a model-selection tool — it has no ModelFinder. So this is
deliberately a **kernel-to-kernel** comparison of the one operation both do: a single likelihood sweep and a
single all-branch gradient sweep on identical problem sizes. It is **not** "our ModelFinder beats BEAGLE"
(that would be unfair and meaningless). The JOLT side uses the **same production kernels** its ModelFinder
uses (`k1_node` post-order lnL; `kj_pre` pre-order all-branch gradient).

Harness: `beagle_jolt_bench.cpp` (committed alongside logs in the provenance dir). Both sides get the same
states/taxa/patterns/rate-categories, the same eigendecomposition/frequencies/rates, warmup + N-rep
averaging. Data is **synthetic** (random tip partials, same workload both sides) — this is a kernel timing +
internal-correctness benchmark, not a real-alignment result. Correctness is established **per side**
(BEAGLE tensor==cuda lnL identical to the bit `-7118563.1495310133`; JOLT Kahan==plain; gradient
finite-difference-checked, `fd_worst ≈ 2.95e-7`), **not** by a cross-tool lnL bit-match (the two harnesses use
different RNG streams, so their absolute lnL values differ — that is expected, not a failure; a `-> FAIL`
self-check line in `jolt_k1_1m.log` is this harness comparing to a fixed oracle on different data, not a
parity failure).

## Numbers (H200; from harvested logs)

| Metric | scale | JOLT | BEAGLE CUDA-core | BEAGLE tensor-core |
|---|---|---|---|---|
| all-branch gradient / eval | 100K | **74.5 ms** | 170.4 ms (**2.29× slower**) | 82.0 ms (≈ parity) |
| lnL / eval | 1M | **111.1 ms (runs)** | **OOM** | **OOM** |

**Defensible claim:** JOLT's gradient kernel is **≈2.3× faster than CUDA-core BEAGLE** and **≈ tensor-core
BEAGLE parity** at 100K; at 1M **both BEAGLE backends run out of memory** while JOLT completes (capability,
not a speedup multiplier — see [../../docs/honest-negatives.md](../../docs/honest-negatives.md)).

## Evidence
Durable: `/g/data/um09/as1708/gems-provenance/reproductions/fig6_beagle_vs_jolt/` — `beagle_{cuda,tensor}_*.log`,
`jolt_k1_*.log`, `jolt_k7_100k.log`, `beagle_jolt_bench.cpp`, `SHA256SUMS.txt`. Job 171269929.
