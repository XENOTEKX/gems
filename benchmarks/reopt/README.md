# Reopt-depth Pareto — the GPU JOLT branch-length re-optimisation floor (m=2)

## Claim (must stay scoped)
The shipped default for the per-round GPU branch-length re-optimisation depth
(env `JOLT_BRLEN_MAXITER`, default **2**) sits at a **Pareto knee, not a quality
trade**. Swept over `m ∈ {12,4,2,1}` on the frozen md5-gated binary `8cc3cb84`,
multi-seed and multi-scale: for `m ≥ 2` the final MLE is **identical to the
`m=12` reference** (|ΔlnL| ≤ 0.004 nat = numerical noise, both signs) at every
scale and seed, while wall drops sharply. `m=2` captures essentially all the
available speed-up at zero quality cost.

**These speed-ups are within-GPU — `m` vs `m=12` on the *same* GPU depth-dial —
NOT a GPU-vs-CPU speed-up.** This is a knob-tuning result on the JOLT optimiser's
inner reopt budget; it makes no claim about GPU vs CPU wall. Keep that distinction
sharp everywhere (README, thesis, caption), the same way the screener's 66× stays
"a clean-room primitive, never end-to-end search". (Distinct from the `jolt/`
Figure-1 result, which is a *GPU-vs-CPU* depth-reopt honest-negative.) See
[../../docs/honest-negatives.md](../../docs/honest-negatives.md).

## Verified from log

**AA-100K, LG+G4, 3 seeds (job 172882999).** Identical MLE **−7541976.852** at
every depth and seed; a clean speed dial:

| m | mean speed-up vs m=12 | ΔlnL vs m=12 (all 3 seeds) |
|---|---|---|
| 12 (ref) | 1.00× | 0 |
| 4 | 1.79× | −1e-4 nat (noise) |
| **2 (shipped)** | **2.43×** | **−1e-4 nat (noise)** |
| 1 | 3.04× | −1e-4 nat (noise) |

Walls: m12 1738–1902 s → m4 990–1038 s → m2 734–755 s → m1 589–604 s.

**DNA-100K, GTR+G4, 3 seeds (job 172883000 — the "escape set").** No basin
escape: ΔlnL is noise-level and appears with *both* signs.

| m | mean speed-up vs m=12 | ΔlnL vs m=12 (range over 3 seeds) |
|---|---|---|
| 12 (ref) | 1.00× | 0 |
| 4 | 1.54× | −0.0033 … +0.0002 nat |
| **2 (shipped)** | **1.72×** | −0.0037 … 0.0 nat |
| 1 | 1.87× | −0.0020 … 0.0 nat |

MLE ≈ **−5692972.98** throughout (max |ΔlnL| = 0.0037 ≤ 0.004 nat). Deeper reopt
finds no better basin — "same tree, less time".

**DNA-1M, GTR+G4, seed 1 (job 172883001 — the decisive cell).** This is what
sets the floor:

| m | wall | ΔlnL vs m=12 | speed-up | verdict |
|---|---|---|---|---|
| 12 (ref) | 2603 s | 0 | 1.00× | reference |
| 4 | 2142 s | +0.0011 | 1.22× | fine |
| **2 (shipped)** | **2092 s** | **−0.0010** | **1.24×** | **sweet spot** |
| 1 | 2388 s | **−3139.83** | 1.09× | **strictly dominated** |

At the bandwidth-bound million-site DNA scale, **`m=1` collapses by −3139.83 nats
(lnL −59211155.81 vs −59208015.98) AND runs slower than `m=2`** (2388 s vs 2092 s):
a single reopt iteration is too shallow to converge the branch lengths at this
scale, so it lands a worse optimum and — unlike the deeper settings — does not even
save wall. (The reopt engagement count rises from 3 at `m≥4` to 4 at `m≤2`; since
that added round appears at the healthy `m=2` cell too, it is *consistent with* but
not *proof of* the `m=1` slowdown — the robust facts are the two dominance axes.)
Because `m=1` is worse on *both* axes it is strictly dominated, so the floor belongs
at **`m=2`, not `m=1`**.

**Conclusion:** a Pareto knee. `m=12` is wasted work; `m=2` reaches the same MLE
at every scale/seed tested; `m=1` risks catastrophic collapse in the large-DNA
regime. No genuine speed-vs-quality trade exists above the floor.

## Topology identity is asserted from lnL, NOT from RF
The harness (`gems_reopt_pareto.sh`) prints an "RF(m vs m12)" column that is a
**parser bug**: it reports `RF(m12 vs m12) = 12` — a tree against itself must be 0 —
because `grep -oE '[0-9]+' | tail -1` grabbed the maxiter digit from the
`rf_s<seed>_m<K>` filename, not the `Robinson-Foulds distance:` value. **Do not
cite any RF number for this result.** Topology identity is carried entirely by
**lnL identity** (|ΔlnL| ≤ 0.004 nat for `m ≥ 2`). Treefile md5s also differ across
depths (branch lengths refine differently), so byte-identity is not the test
either. Recorded as a caveat in
[../../docs/honest-negatives.md](../../docs/honest-negatives.md).

## Reproduce
```bash
# Frozen canonical binary only (md5 8cc3cb844e807fe3ecbcd0b9ff5428df); it defaults
# to JOLT_BRLEN_MAXITER=2 and exposes the knob. The dev tree 2c931f41 hardcodes 12
# with no knob and MUST NOT be used. The harness self-gates on md5 + knob presence.
qsub -q gpuhopper -lngpus=1 -lncpus=12 -lmem=90GB  -lwalltime=05:00:00 \
     -v TYPE=AA,SCALE=100000,SEEDS="1 2 3"  gems_reopt_pareto.sh   # 172882999
qsub -q gpuhopper -lngpus=1 -lncpus=12 -lmem=90GB  -lwalltime=03:00:00 \
     -v TYPE=DNA,SCALE=100000,SEEDS="1 2 3" gems_reopt_pareto.sh   # 172883000
qsub -q gpuhopper -lngpus=1 -lncpus=12 -lmem=120GB -lwalltime=05:00:00 \
     -v TYPE=DNA,SCALE=1000000,SEEDS="1"    gems_reopt_pareto.sh   # 172883001
```
Each cell runs `<bin> -s <aln> -m <MODEL> -seed <s> -nt 12 --jolt --gpu --ts-fused`
with `JOLT_BRLEN_MAXITER=<m>`. Alignments under
`/scratch/dx61/sa0557/iqtree2/poc_builds/complex_data_shared/{AA,DNA}/...`
(true model LG+G4 / GTR+G4, 100 taxa).

## Evidence
Durable, hashed:
`/g/data/um09/as1708/gems-provenance/reproductions/reopt_pareto_AA_100000_172882999.gadi-pbs/`,
`…/reopt_pareto_DNA_100000_172883000.gadi-pbs/`,
`…/reopt_pareto_DNA_1000000_172883001.gadi-pbs/`
(each has `pareto.tsv`, per-cell `.iqtree`/`.stdout`/`.treefile`, and `SHA256SUMS.txt`).
Jobs **172882999 / 172883000 / 172883001**, all H200, binary `8cc3cb84` (md5 gate PASS).
