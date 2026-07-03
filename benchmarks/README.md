# GEMS benchmarks — reproduction index

Every headline number in this repository traces to a specific batch-scheduler job, a fixed input
(alignment md5 + model + seed), and a primary log preserved in a durable, hashed archive. This file is
the index from *claim → script → evidence*. Large raw artifacts (multi-GB alignments, `.nsys-rep`,
build trees) are **not committed**; each benchmark subdirectory carries the run script and a pointer to
the durable log path plus its SHA-256, so a reviewer can fetch and verify the exact bytes.

## Where the evidence lives

- **Durable archive (not purged):** `/g/data/um09/as1708/gems-provenance/`
  - `reproductions/<label>_<jobid>/` — primary logs (`*.log`, `*.iqtree`, `*.stdout`) + derived CSVs, each with its own `SHA256SUMS.txt`.
  - `scratch-survivors/<figure>/` — logs recovered off rolling-purge `/scratch`.
  - `SHA256SUMS.txt` + `MANIFEST.md` — integrity over the whole archive.
- **This repo commits:** the run scripts (`benchmarks/*/`), the small extracted CSVs the figures consume
  (`figures/data/`), and the plot scripts (`figures/`). It commits log **hashes** and durable paths, not raw logs.

## Reproduce discipline (PARITY rule)

A figure is ✅ only when its reproduction uses the **same dataset (md5), model, and seed** as its reference
**and**, for a GPU figure, **GPU lnL == CPU lnL** to the stated relative error. Feasibility/capability results
(not throughput speedups) and motivation-only results are labelled as such here and must stay labelled that way
everywhere (README, thesis, captions).

## Index (status column reflects direct-from-log verification, 2026-07-02)

| Fig | Claim (as it MUST be stated) | Job ID(s) | Script | Durable logs | Verified from log |
|---|---|---|---|---|---|
| 3 | GPU NNI **screener** clean-room primitive: AA-1M **66×** (oracle 365.22 s) — *screener primitive, never end-to-end search* | 172194079 | [screener/](screener/) | `scratch-survivors/fig3-screener-scaling/` | ✅ `wall_ratio 66.18×`, `wall_oracle_s 365.22`, labeled "screener vs oracle" |
| 4 | GPU **parsimony** 2D-grid Fitch, **bit-identical**; AA-1M speed **~1.0× same-node** (56.9 s GPU vs 56.8 s CPU IQ-pars, same node/binary/seed), **≤1.57×** deployment-frame (vs 104-core node 89.6 s) — *3.76× and 71× RETIRED* | 172862242 / 172862243 | [parsimony/](parsimony/) | `reproductions/fig4_parsimony_bulletproof_172862242/` | ✅ 2× `[GPUPARS-B]` engaged; `VERIFY mismatches=0`; both legs score 15488909 |
| 5 | **Bit-identity table**: JOLT 2.77e-12 · mixture 1.56e-16 · derv 1e-13 · tiling 4.465e-13 · +R ≤5e-16 — *same-device ≤1M; cross-arch ≤5e-11 @10M* | many | [jolt/](jolt/) → [docs/validation.md](../docs/validation.md) | `reproductions/`, `gems-validation-logs/` | 🟡 partial — screener RF & pars VERIFY=0 ✅; per-row rel trace pending |
| 6 | JOLT vs **BEAGLE 4.0**: ≈ tensor-core parity (lnL/grad), **2.3–2.4×** over CUDA-core, runs AA-1M where BEAGLE OOMs | 171269929 | [beagle/](beagle/) | ⚠ **harvest owed** — `iqtree3-gpu/beagle_vs_jolt_scale/` | ✅ CUDA 37.45 ms vs tensor 15.20 ms = 2.46×, identical lnL −7118563.14953; BEAGLE `CUDA error: Out of memory` @1M logged |
| 7 | **Profile-mixture** feasibility (MEOW): JOLTMix engaged, GPU==CPU rel ≤1e-6 — *validated, env-gated; NOT throughput* | 172809011 (+171733080) | [mixtures/](mixtures/) | `reproductions/fig7_meow_euk_172809011/` | ✅ `[JOLTMIX-DBG] outer=0..N` engaged+converging; rel parity from full-validation 171733080 |
| 8 | **CTF recall / logarithmic BIC**: ΔBIC≈Δp·ln(L) — AA slope **0.922**, DNA **0.419**; recall@3=exact@1=1.00 to L=1K | 171258771 / 172813344 / 172813345 | [ctf/](ctf/) | `reproductions/jobA_subsuff_*` | ✅✅ regression recomputed: AA 0.922 / DNA 0.419; (+ AliSim synthetic owed) |
| — | **CTF avian** (motivation): 37.35 M sites → GTR+F+I+R2 — *model-selection + fixed-topology fit, NOT a finished tree* | 172810407 | [ctf/](ctf/) | `reproductions/ctf_avian_realdata_172810407/` | ✅ CTF subsample→refine ran; winner + honest "+R may miss" WARNING in log |
| — | **Reopt-depth Pareto**: shipped `JOLT_BRLEN_MAXITER` floor **m=2** is a Pareto knee — same MLE for m≥2 (\|ΔlnL\|≤0.004 nat, 3 seeds × 3 scales), 1.24–2.43× vs m=12 *within-GPU, not vs CPU*; DNA-1M **m=1 collapses −3139.83 nats and is slower** | 172882999 / 172883000 / 172883001 | [reopt/](reopt/) | `reproductions/reopt_pareto_{AA_100000_172882999,DNA_100000_172883000,DNA_1000000_172883001}.gadi-pbs/` | ✅ lnL-identity (RF column is a parse bug — ignored) |
| 1 | JOLT depth reopt — **honest-negative** at AA-1M (≈1.0×; the 4.8× ref workload differs) | 170361630 / 172809010 | [jolt/](jolt/) → [docs/honest-negatives.md](../docs/honest-negatives.md) | `reproductions/fig1_fig10_aa1m_172809010/` | ⚠ honest-negative, recorded |
| 2 | CTF **`-m MF` vs CPU**: DNA-1M 7.4–13× — GPU-CTF leg vs FCA CPU baseline | 170843136 / 172809176 | [ctf/](ctf/) | `fca-mtest-1m-scaling/` | ⏳ pending leg |
| N | **GPU-vs-GPU** (this work vs Hashara OpenACC): ModelFinder + tree-search parity sweep 10K–10M | (sweep in flight) | [hashara_parity/](hashara_parity/) | `hashara_parity/` | 🟡 10K parity ✅ (ΔlnL=0), sweep running |

**Non-claims (do not assert — see [docs/honest-negatives.md](../docs/honest-negatives.md)):** single GPU beats a
16-node cluster on throughput (bandwidth thesis falsified, job 170398260); AA-1M/10M as a throughput speedup
(tiling is a capability, runs where CPU OOMs); a finished avian tree; mixtures as production; screener 66× as
end-to-end search.
