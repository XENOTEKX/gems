# Figure 4 — GPU parsimony (2D-grid Fitch)

## Claim (corrected 2026-07-02 — supersedes the retired "3.76×")
GPU 2D-grid Fitch start-tree construction is **bit-identical** to the CPU parsimony score. Its **speed is
same-node neutral at AA-1M**: on the same H200 node, same frozen binary (`8cc3cb84`), same 12 threads, same
seed, same alignment (md5 `da36879a`), the GPU 2D kernel and the CPU IQ-parsimony leg both take ~56.9 s
(**0.998× — a tie**). A whole-box **deployment-frame** comparison (1×H200 + host cores vs a full 104-core
`normalsr` node = 89.578 s) gives at most **1.57×**, but that answers a different question (which box you own).

**Correctness is solid; per-model speed is not demonstrated at a million sites.** State those two separately.

## RETIRED numbers — do NOT cite
- **"71×"** — a **PLL artifact**: the 4033 s came from PLL, a slow non-default start-tree library. IQ-TREE's own
  IQ-parsimony already beats PLL ~16–19× on CPU alone (a start-tree library choice, unrelated to the GPU).
- **"3.76×" / "2.34×"** — the CPU baseline was a **throttled 12-core run on the wrong, slower binary
  (`2c931f41`, ~211 s)**. Put both legs on the optimised `8cc3cb84` same-node and the win evaporates to a tie.
  See [../../docs/honest-negatives.md](../../docs/honest-negatives.md).

## Parity / engagement (confirmed, job 172862242)
The GPU leg shows **2× `[GPUPARS-B]` engage markers** (`Phase-B engaged`), so it is not a silent CPU fallback;
the separate `GPU_PARSIMONY_VERIFY=1` leg reports **`VERIFY mismatches = 0`** (bit-identical Fitch), and both
legs score the identical parsimony value (15488909 at AA-1M).

## Open (a possible win only at larger scale)
At AA-1M with 12 fast host cores the CPU tree-manipulation + fast Fitch keeps pace; a GPU offload win may only
appear at **AA-10M** (Fitch-scoring-dominated). That is a *future* measurement (jobs 172879565 / 172879566, in
progress), not a current claim.

## Reproduce
```bash
qsub fig4_parsimony_bulletproof.sh    # CPU IQ-pars + GPU 2D + VERIFY, one job, one node, one binary
```
CPU baseline is IQ-TREE parsimony (`-starttree PARS`, no `--gpu`) — **never PLL**. The binary must be `8cc3cb84`
(the only build with the GPU parsimony kernel); a `grep GPU_PARSIMONY_BATCHED` binary guard plus a `[GPUPARS-B]`
engagement guard prevent a silent-CPU-fallback false pass.

## Evidence
Fair, self-contained run: `reproductions/fig4_parsimony_bulletproof_172862242/` (GPU + CPU legs, same
node/binary/seed) and `reproductions/cpu_parsimony_fullnode_172862243/` (CPU-104t deployment leg = 89.578 s).
Jobs **172862242 / 172862243**. The old `172524833 / 172526962` and the `172840922 / 172852055` re-runs used a
mismatched binary pair and are superseded.
