#!/usr/bin/env python3
"""aggregate_hashara_parity.py — build the JOLT-vs-naive parity table from cell TSVs.
Reads /g/data/um09/as1708/gems-provenance/hashara_parity/<TYPE>_<SCALE>_<TOOL>.tsv and, for each
(TYPE, SCALE), pairs jolt vs naive:
  ① ModelFinder — PARITY: same best-fit model + winner lnL (|Δ|<2 nat); SPEEDUP = naive_mf_wall/jolt_mf_wall
  ② Tree search — PARITY: same BEST SCORE (|Δ|<2 nat);                 SPEEDUP = naive_ts_wall/jolt_ts_wall
A speedup >1 means JOLT is faster. Below ~100K the GPU is expected to lose (crossover); that is part of
the story, not a bug. Rows with a missing/failed cell are flagged, not silently dropped.
"""
import glob, os, sys
DIR = "/g/data/um09/as1708/gems-provenance/hashara_parity"
def load(p):
    d = {}
    with open(p) as f:
        for ln in f:
            if "\t" in ln:
                k, v = ln.rstrip("\n").split("\t", 1); d[k] = v
    return d
def fnum(x):
    try: return float(x)
    except: return None
cells = {}
for p in glob.glob(os.path.join(DIR, "*.tsv")):
    d = load(p)
    key = (d.get("type"), d.get("scale"), d.get("tool"))
    cells[key] = d
types = ["AA", "DNA"]; scales = ["10000", "100000", "1000000", "10000000"]
def hdr(t): print(f"\n{'='*96}\n {t}\n{'='*96}")
hdr("① MODELFINDER  (JOLT --ctf  vs  naive per-model GPU)   speedup>1 ⇒ JOLT faster")
print(f"{'type':4} {'scale':>9} {'jolt_model':>16} {'naive_model':>16} {'model✓':>7} {'lnLΔ':>10} {'jolt_s':>9} {'naive_s':>9} {'speedup':>8}")
for ty in types:
    for sc in scales:
        j = cells.get((ty, sc, "jolt")); n = cells.get((ty, sc, "naive"))
        if not j or not n:
            print(f"{ty:4} {sc:>9}   (missing {'jolt' if not j else ''}{' naive' if not n else ''})"); continue
        jm, nm = j.get("mf_model","NA"), n.get("mf_model","NA")
        ok = "YES" if jm == nm and jm != "NA" else "**NO**"
        dl = fnum(j.get("mf_lnl")); dn = fnum(n.get("mf_lnl"))
        lnd = f"{abs(dl-dn):.3f}" if (dl is not None and dn is not None) else "NA"
        js, ns = fnum(j.get("mf_wall")), fnum(n.get("mf_wall"))
        sp = f"{ns/js:.2f}x" if (js and ns and js>0) else "NA"
        print(f"{ty:4} {sc:>9} {jm:>16} {nm:>16} {ok:>7} {lnd:>10} {js if js else 'NA':>9} {ns if ns else 'NA':>9} {sp:>8}")
hdr("② TREE SEARCH  (JOLT --ts-fused  vs  naive OpenACC, clean no-boot)   speedup>1 ⇒ JOLT faster")
print(f"{'type':4} {'scale':>9} {'jolt_BEST':>16} {'naive_BEST':>16} {'score✓':>7} {'Δnat':>10} {'jolt_s':>9} {'naive_s':>9} {'speedup':>8}")
for ty in types:
    for sc in scales:
        j = cells.get((ty, sc, "jolt")); n = cells.get((ty, sc, "naive"))
        if not j or not n:
            print(f"{ty:4} {sc:>9}   (missing {'jolt' if not j else ''}{' naive' if not n else ''})"); continue
        js_, ns_ = fnum(j.get("ts_score")), fnum(n.get("ts_score"))
        ok = "**NO**"; dn = "NA"
        if js_ is not None and ns_ is not None:
            dn = f"{abs(js_-ns_):.3f}"; ok = "YES" if abs(js_-ns_) < 2.0 else "**NO**"
        jw, nw = fnum(j.get("ts_wall")), fnum(n.get("ts_wall"))
        sp = f"{nw/jw:.2f}x" if (jw and nw and jw>0) else "NA"
        jb = f"{js_:.3f}" if js_ is not None else "NA"; nb = f"{ns_:.3f}" if ns_ is not None else "NA"
        print(f"{ty:4} {sc:>9} {jb:>16} {nb:>16} {ok:>7} {dn:>10} {jw if jw else 'NA':>9} {nw if nw else 'NA':>9} {sp:>8}")
print("\nnote: cells recorded as they finish; re-run this after each tier. 'model✓/score✓ NO' ⇒ parity FAIL, investigate.")
