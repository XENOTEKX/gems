# Figure 3 — GPU tree-search NNI screener scaling

## Claim (must stay scoped)
The batched NNI screener is **bit-identical** to the CPU oracle (same NNI ranking / RF = 0) and, at AA-1M,
evaluates the NNI set **66× faster** than the full-postorder CPU oracle (5.52 s vs **365.22 s**).

**This is a clean-room screener PRIMITIVE, never an end-to-end tree-search speedup.** The integrated
`--ts-fused` search is ≈ one CPU node and ~1.9× *behind* a competing OpenACC GPU at AA-1M. The 66× is the
isolated NNI-evaluation kernel vs the oracle — keep screener-vs-search sharp everywhere (README, thesis,
caption). See [../../docs/honest-negatives.md](../../docs/honest-negatives.md).

## Verified from log
`wall_oracle_s 365.220750`, `wall_ratio 66.175481x`, labelled in-log "pattern-tiled batched NNI screener vs
3a oracle". Topology identity: RF = 0.

## Evidence
Durable: `/g/data/um09/as1708/gems-provenance/scratch-survivors/fig3-screener-scaling/ts2i3c_aa1m_172194079/`.
Job 172194079.
