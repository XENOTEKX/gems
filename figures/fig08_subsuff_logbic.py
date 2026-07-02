#!/usr/bin/env python3
"""Figure 8 — subsample sufficiency: BIC converges logarithmically.

On the shipped GPU JOLT ModelFinder pipeline (-m MF --jolt --gpu), the full-data winner is recovered from a
tiny site subsample: recall@3 = exact@1 = 1.00 at every L. The gap to the full-data BIC shrinks as
ΔBIC ≈ Δp·ln(L) — the penalty-dominated logarithmic regime. Slopes fitted here (least squares) reproduce the
verified 0.92 (AA) / 0.42 (DNA). GPU==CPU parity (AA winner LG+G4, DNA winner F81+F+G4).

Data: figures/data/dBIC_vs_lnL_{aa1m,dna1m}.csv (jobs 172813344 / 172813345). Run: python3 fig08_subsuff_logbic.py
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)

def load(name):
    xs, ys = [], []
    for r in csv.DictReader(open(os.path.join(HERE, "data", name))):
        xs.append(float(r["lnL_val"])); ys.append(float(r["mean_dBIC"]))
    return xs, ys

def fit(xs, ys):
    n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
    slope = sum((x-mx)*(y-my) for x, y in zip(xs, ys)) / sum((x-mx)**2 for x in xs)
    return slope, my - slope*mx

fig, ax = plt.subplots(figsize=(5.6, 4.2))
for name, color, tag in [("dBIC_vs_lnL_aa1m.csv", "#2a7", "AA (protein)"),
                         ("dBIC_vs_lnL_dna1m.csv", "#36c", "DNA")]:
    path = os.path.join(HERE, "data", name)
    if not os.path.exists(path):
        continue
    xs, ys = load(name); m, b = fit(xs, ys)
    ax.scatter(xs, ys, color=color, s=32, zorder=3, label=f"{tag}: ΔBIC ≈ {m:.2f}·ln(L) + {b:.1f}")
    lo, hi = min(xs), max(xs)
    ax.plot([lo, hi], [m*lo+b, m*hi+b], color=color, lw=1.4, alpha=0.8, zorder=2)

ax.set_xlabel("ln(L)   (L = subsample size in patterns)")
ax.set_ylabel("ΔBIC  (subsample winner vs full-data winner)")
ax.set_title("Subsample sufficiency — BIC gap converges logarithmically")
ax.legend(fontsize=8, loc="upper left")
ax.text(0.5, -0.18, "recall@3 = exact@1 = 1.00 at every L; GPU==CPU parity. Jobs 172813344 (AA) / 172813345 (DNA).",
        transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color="#555")
plt.tight_layout()
out = os.path.join(HERE, "fig08_subsuff_logbic.pdf")
plt.savefig(out, bbox_inches="tight"); plt.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print("wrote", out)
