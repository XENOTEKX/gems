# JOLT joint optimizer — depth reopt (fig 1) + write-back bit-identity (fig 5)

JOLT is the GPU joint optimizer at the core of GEMS: it optimizes branch lengths + rate/model parameters
together on the GPU, reaching the same MLE as the CPU in O(N)-gradient iterations, and writes the result back
only if it matches the CPU to a tight tolerance (else it declines to CPU).

## Fig 5 — write-back bit-identity (the correctness contract)
For mean-gamma models the GPU==CPU write-back is bit-tight (self-check rel ~2.77e-12; GPU≡CPU rel 0 on
same-device ≤1M patterns). The **decline path is real and correct**: models whose write-back exceeds the 1e-6
gate fall back to CPU — the logs show e.g. `write-back MISMATCH rel=8.66e-06 → CPU fallback (model=JC+R2)`.
This is the merge-gate safety property, not a failure. Full rel-error table: [../../docs/validation.md](../../docs/validation.md).

## Fig 1 — depth reopt (HONEST-NEGATIVE)
The AA-100K `-te` single-model depth-refine measured **4.8×** (47 s vs 224 s) in its reference workload
(170361630). The AA-1M re-run of the *same config* (172809010) is **≈1.0×** — at 1M the single 25-iteration
BL+model reopt is I/O-dominated and the CPU keeps pace. **This is recorded as an honest-negative**
([../../docs/honest-negatives.md](../../docs/honest-negatives.md)): do not present the AA-1M depth number as a
speedup, and do not relabel it "4.8×". The 4.8× stands only for its original heavier AA-100K workload profile,
which must be re-stated exactly before it is cited.

## Evidence
Durable: `reproductions/fig1_fig10_aa1m_172809010/`, `gems-validation-logs/`. Jobs 170302036 / 170361630 / 172809010.
