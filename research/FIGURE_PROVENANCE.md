# GEMS figure-provenance archive

Durable (non-purged) snapshot of the primary run logs and derived-result summaries
behind the GEMS thesis figures. Created 2026-07-01 from `/scratch/rc29/as1708` (which
Gadi purges on a rolling window) and `~/setonix-iq`.

- **Location:** `/g/data/um09/as1708/gems-provenance/` (Gadi `/g/data`, persistent; not scratch)
- **Integrity:** `SHA256SUMS.txt` covers all 97 files. Verify with
  `cd /g/data/um09/as1708/gems-provenance && sha256sum -c SHA256SUMS.txt`
- **Contents:**
  - `scratch-survivors/` — raw primary logs still on scratch at archival time (figs 3, 4)
  - `gems-validation-logs/` — the GEMS P1e (CPU-OFF §3f parity) + P1f (GPU-ON GPU≡CPU) job
    `.o` logs = the current bit-identity / result-identity evidence (fig 5)
  - `research-summaries/` — the design/log markdown (`Modelfinder-gpu/`, `Treesearch/`) that
    record the derived numbers for every figure; the durable fallback when a raw log is gone

## Honest provenance status — per figure (plan §6)

| Fig | Result | Job ID(s) | Raw primary log | Provenance in this archive |
|-----|--------|-----------|-----------------|-----------------------------|
| 1 | JOLT depth speedup (4.8× AA-100K `-te`) | 170361630 | **PURGED** from scratch | derived numbers: `research-summaries/Modelfinder-gpu/` |
| 2 | CTF `-m MF` vs CPU (DNA-1M 7.4–13×, AA-1M 1.46×) | 170843136 / 170756438 | **PURGED** | `research-summaries/Modelfinder-gpu/` |
| 3 | Tree-search screener scaling (3.45× → 66×) | 172194079 | ✅ **archived** | `scratch-survivors/fig3-screener-scaling/` |
| 4 | GPU parsimony vs scale (2.34× → 3.76×) | 172524833 / 172526962 | ✅ **archived** | `scratch-survivors/fig4-parsimony/` |
| 5 | Bit-identity table (GPU≡CPU rel) | multiple | partial | `gems-validation-logs/` (rel ≤ ~2.9e-13, §3f) + `research-summaries/` |
| 6 | JOLT vs BEAGLE 4.0 | 171269929 | **PURGED** | `research-summaries/Modelfinder-gpu/` |
| 7 | Mixture feasibility wall | 171745185 / 171759796-7 | **PURGED** | `research-summaries/Modelfinder-gpu/` |
| 8 | CTF recall (30/30 + 45/45 vs 0/15) | 171258771 / 171466576 | **PURGED** | `research-summaries/Modelfinder-gpu/` |

## Data-availability caveat (for the thesis §10)

**Raw primary logs survive for only 3 of 8 figures** (3, 4, and the current bit-identity
evidence for 5). For figures 1, 2, 6, 7, 8 the original job `.o`/run directories were
**already purged from Gadi `/scratch` by the rolling-window expiry before this archival** —
exactly the risk the extraction plan flagged. Their derived results are preserved in the
`research-summaries/` design/log docs (and in the project memory), but the raw per-job logs
are not recoverable from this environment.

**Owed before thesis submission:** either (a) re-run figs 1, 2, 6, 7, 8 to regenerate raw
logs (the GEMS binary + the frozen parity binary can reproduce them), or (b) present those
figures citing the surviving `research-summaries/` numbers and state the raw-log gap
explicitly in the data-availability statement. Do not claim raw-log provenance for the five
purged figures.
