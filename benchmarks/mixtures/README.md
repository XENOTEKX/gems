# Figure 7 — Profile-mixture feasibility (JOLTMix)

## Claim (scoped)
The GPU joint optimizer handles **profile-mixture models** (C20/C60/MEOW80-class). GPU lnL == CPU lnL to the
errors in [../../docs/validation.md](../../docs/validation.md) — full real-data validation **rel 2.46e-13**
(LG+MEOW80+G4, euk) and G.8.0 subsample **rel 1.56e-16**. At genome scale the GPU **tiles** where a CPU node
OOMs (935 GB arena > 503 GB RAM at 200k sites).

**Status: validated, env-gated (`JOLT_MIX_HOSTDRIVEN`) — NOT a production throughput claim.** The
aggregate-throughput justification for default-on is unwritten; walltime is 1.74× vs a full CPU node (grows to
2.74× @100k). Present as feasibility/capability, not a speedup headline.

## Verified from log
`[JOLTMIX-DBG] outer=0..N` shows the mixture GPU path engaged and the lnL converging monotonically. Full
parity rel from job 171733080; G.8.0 rel-16 from 171604565.

> **Naming:** "MEOW80" is this project's internal codename for an 80-component profile mixture. Before the
> repo is cited, replace it with the mixture's actual published name (a P6 rename loose-end).

## Evidence
Durable: `reproductions/fig7_meow_euk_172809011/`. Jobs 172809011 / 171733080 / 171604565 / 171745185.
