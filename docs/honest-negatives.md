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
| Parsimony **"71×"** *and* **"3.76×"** | The 4033 s ("71×") was a **PLL artifact** (IQ-TREE's own IQ-parsimony already beats PLL ~16–19× on CPU alone, a start-tree library choice, quality-identical). The **"3.76×" is also retired**: its CPU baseline was a throttled 12-core run on a slower binary (`2c931f41`). On a fair same-node/same-binary (`8cc3cb84`) basis the GPU 2D-grid kernel **ties** the CPU (0.998×, 56.9 vs 56.8 s), ≤**1.57×** in a whole-box deployment frame. The kernel is **bit-identical** (score 15488909, `VERIFY mismatches=0`, `[GPUPARS-B]` engaged); its **speed is not demonstrated at 1M** (a win may appear only at 10M). | 172862242 / 172862243 |
| Reopt-depth sweep as a **basin-escape / quality win** (or as a GPU-vs-CPU speedup) | Sweeping GPU reopt depth m∈{12,4,2,1} produced **no genuine basin escape**: for m≥2, ΔlnL is noise (≤0.004 nat, both signs) at every scale/seed — same tree, less time. `m=2` is a **Pareto knee, not a quality dial**; its 1.24–2.43× figures are **GPU-vs-GPU across the reopt depth-dial, never GPU-vs-CPU**. Below the floor, DNA-1M **m=1 collapses −3139.83 nats and is *slower* than m=2** (2388 s vs 2092 s) — strictly dominated. | 172882999 / 172883000 / 172883001 |

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
- **Reopt-depth basin-escape refuted** — the DNA "escape set" (DNA-100K GTR+G4, 3 seeds) did **not** escape:
  ΔlnL both signs ≤ 0.004 nat vs m=12, and no deeper reopt found a better basin at any tested scale (m=2
  already reaches the m=12 MLE). Consistent with the earlier "same tree, less time" refutation. (172883000;
  AA-100K 172882999; DNA-1M 172883001.)
- **`gems_reopt_pareto.sh` RF-vs-m12 column is a parser bug** — it prints RF = the maxiter digit (self-check
  `RF(m12 vs m12) = 12`, which must be 0); the `grep -oE '[0-9]+' | tail -1` read the `…_m<K>` filename digits,
  not the `Robinson-Foulds distance:` value. **Never cite these RF numbers.** The reopt "same tree" claim rests
  on **lnL identity** (ΔlnL ≤ 0.004 nat for m≥2); treefile md5s differ by depth (branch lengths refine), so
  byte-identity is not the test either. Cheap follow-up: fix the regex to read the `.rfdist`.
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
