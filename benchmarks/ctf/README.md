# Figures 2 & 8 + avian — Coarse-to-Fine (CTF) ModelFinder

CTF ranks all candidate models on a small site **subsample**, then refines the top-k on full data. The thesis
uses three CTF results:

## Fig 8 — logarithmic BIC convergence (subsample sufficiency)
On the **shipped GPU JOLT pipeline** (`-m MF --jolt --gpu`), the full-data winner is recovered from a tiny
subsample: **recall@3 = exact@1 = 1.00 at every L (1000→50000)**, and ΔBIC ≈ Δp·ln(L) — the penalty-dominated
logarithmic regime. Regression slopes **recomputed from the CSVs this session**: **AA 0.922, DNA 0.419**
(matching the claimed 0.92 / 0.42). GPU==CPU parity: AA winner LG+G4, DNA winner F81+F+G4.
- Evidence: `reproductions/jobA_subsuff_aa1m_172813344/` (+ `dBIC_vs_lnL_aa1m.csv`), `jobA_subsuff_dna1m_172813345/`. Jobs 172813344 / 172813345 / 171258771.
- **Owed:** a fresh AliSim synthetic recall run to complement the real-data proof.

## avian — motivation only (NOT a finished tree)
CTF on the avian TENT (**37,350,521 sites, 21.76M distinct patterns**): coarse-rank on a 5000-site subsample →
refine top-3 → winner **GTR+F+I+R2** (matches the known avian result). **This is model selection + a
fixed/partial-topology fit, not a completed ML phylogeny** — cite as motivation/feasibility only. The log
carries the honest self-flag: *"WARNING: a +R/+I model genuinely leads on the subsample — eligible-refine may
MISS the true winner."*
- Evidence: `reproductions/ctf_avian_realdata_172810407/`. Job 172810407. 9.87 h.

## Fig 2 — CTF `-m MF` vs CPU (pending leg)
DNA-1M `-m MF` **7.4–13×** vs the CPU FCA baseline. The GPU-CTF leg vs the CPU `-m MF` baseline is still to be
finalised (FCA `-m MF` at 1M is infeasible → curve is `-m TEST` at 1M + `-m MF` at 100K).
- Evidence: `fca-mtest-1m-scaling/`. Jobs 170843136 / 172809176.
