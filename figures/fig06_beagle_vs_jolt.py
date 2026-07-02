#!/usr/bin/env python3
"""Figure 6 — JOLT vs BEAGLE 4.0 (matched single-sweep kernel micro-benchmark).

Honest framing (see benchmarks/beagle/README.md): kernel-to-kernel, not pipeline-vs-library. JOLT uses its
PRODUCTION k1_node / kj_pre kernels. Synthetic identical workload both sides; correctness is per-side
(FD-checked gradient, Kahan==plain lnL), not a cross-tool lnL bit-match.

Data: figures/data/beagle_vs_jolt.csv (job 171269929). Run: python3 fig06_beagle_vs_jolt.py
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
# skip '#' comment lines before the header, then parse
lines = [ln for ln in open(os.path.join(HERE, "data", "beagle_vs_jolt.csv")) if not ln.lstrip().startswith("#")]
rows = list(csv.DictReader(lines))
grad = next(r for r in rows if r["op"] == "grad")

labels = ["JOLT\n(this work)", "BEAGLE 4.0\nCUDA-core", "BEAGLE 4.0\ntensor-core"]
vals = [float(grad["jolt_ms"]), float(grad["beagle_cudacore_ms"]), float(grad["beagle_tensorcore_ms"])]
colors = ["#2a7", "#c44", "#e8a"]

fig, ax = plt.subplots(figsize=(5.2, 4.0))
bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 3, f"{v:.1f} ms", ha="center", va="bottom", fontsize=9)
sp = vals[1] / vals[0]
ax.annotate(f"{sp:.2f}× over CUDA-core", xy=(0, vals[0]), xytext=(0.5, vals[1]*0.75),
            fontsize=10, ha="center", color="#333")
ax.set_ylabel("all-branch gradient — ms / eval  (100K patterns, lower is better)")
ax.set_title("JOLT vs BEAGLE 4.0 — matched gradient kernel (H200, FP64)")
ax.set_ylim(0, max(vals) * 1.25)
ax.text(0.5, -0.20, "At 1M patterns both BEAGLE backends run out of memory; JOLT completes (111 ms).\n"
        "Kernel micro-benchmark, synthetic workload; job 171269929.",
        transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color="#555")
plt.tight_layout()
out = os.path.join(HERE, "fig06_beagle_vs_jolt.pdf")
plt.savefig(out, bbox_inches="tight"); plt.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print("wrote", out)
