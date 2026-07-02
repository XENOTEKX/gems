# Figure 4 — GPU parsimony (2D-grid Fitch)

## Claim
GPU 2D-grid Fitch start-tree construction is **bit-identical** to the CPU parsimony score and, at AA-1M, runs
**~3.76× faster than IQ-TREE's own IQ-parsimony** (the best-available CPU baseline). Headline is **3.76×,
never "71×"** — the 71× came from comparing against PLL, a slow non-default start-tree library; bundling that
CPU-library swap into the GPU number mis-attributes a CPU change to the GPU (see
[../../docs/honest-negatives.md](../../docs/honest-negatives.md)).

## Bulletproofing note (2026-07-02)
The original run (172524833 / 172526962) measured the GPU leg (56.591 s) in-job but the CPU baseline was a
**hardcoded reference** (`REF: CPU PARS(12t)=212.7s`) echoed from an earlier, uncited run — not measured
alongside the GPU. That is not good enough for the thesis. **`fig4_parsimony_bulletproof.sh` (job 172840922)**
re-measures **both** legs — CPU IQ-parsimony *and* GPU 2D-grid — in **one job, same node, same binary, same
dataset (md5 da36879a), same seed**, plus an in-job `GPU_PARSIMONY_VERIFY` bit-identity check, and reports a
single self-contained speedup with both parsimony scores required to match. Cite that job's number.

## Parity (already confirmed)
`GPU_PARSIMONY_VERIFY=1` → **`VERIFY mismatches = 0`** on AA-10K and DNA-10K; identical parsimony score
(1461466) between GPU and CPU trajectories.

## Reproduce
```bash
qsub fig4_parsimony_bulletproof.sh    # CPU IQ-pars + GPU 2D + VERIFY, one job
```
CPU baseline is IQ-TREE parsimony (`-starttree PARS`, no `--gpu`) — **never PLL**.

## Evidence
Durable: `/g/data/um09/as1708/gems-provenance/scratch-survivors/fig4-parsimony/` (original) and
`reproductions/fig4_parsimony_bulletproof_172840922/` (self-contained re-run). Jobs 172524833 / 172526962 / 172840922.
