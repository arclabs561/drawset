# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib"]
# ///
"""Generate throughput benchmark plot for the drawset README."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Benchmark data from criterion (Apple Silicon, NEON)
# Format: name -> {stream_size: median_time_us}
alg_l = {1_000: 15.86, 10_000: 48.95, 100_000: 243.98}
alg_r = {1_000: 7.67, 10_000: 67.23, 100_000: 647.05}
a_res = {1_000: 87.19, 10_000: 858.51, 100_000: 8384.3}

# Gumbel: vocab_size -> median_time_us
gumbel_max = {10: 0.205, 100: 1.996, 1_000: 19.84}
gumbel_topk = {100: 3.054, 1_000: 33.82}

def to_throughput(size_us: dict) -> tuple[list, list]:
    """Convert size->us to (sizes, Melem/s)."""
    sizes = sorted(size_us.keys())
    meps = [s / size_us[s] for s in sizes]  # elements/us = Melem/s
    return sizes, meps

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [3, 2]})

# Left: reservoir throughput vs stream size
colors = {"Alg L": "#2563eb", "Alg R": "#dc2626", "A-Res": "#9333ea"}
for name, data in [("Alg L", alg_l), ("Alg R", alg_r), ("A-Res", a_res)]:
    sizes, meps = to_throughput(data)
    ax1.plot(sizes, meps, "o-", label=name, color=colors[name], markersize=5, linewidth=1.8)

ax1.set_xlabel("Stream size (N, k=100)", fontsize=11)
ax1.set_ylabel("Throughput (Melem/s)", fontsize=11)
ax1.set_title("drawset: reservoir sampling throughput", fontsize=12, fontweight="bold")
ax1.legend(frameon=True, fontsize=9, loc="upper right")
ax1.set_xscale("log")
ax1.grid(True, alpha=0.3)

# Right: latency table (reservoir + gumbel)
table_data = [
    ["Alg L", "1K", f"{alg_l[1_000]:.1f} us"],
    ["Alg L", "100K", f"{alg_l[100_000]:.0f} us"],
    ["Alg R", "1K", f"{alg_r[1_000]:.1f} us"],
    ["Alg R", "100K", f"{alg_r[100_000]:.0f} us"],
    ["A-Res", "1K", f"{a_res[1_000]:.1f} us"],
    ["A-Res", "100K", f"{a_res[100_000] / 1000:.1f} ms"],
    ["gumbel_max", "V=100", f"{gumbel_max[100] * 1000:.0f} ns"],
    ["gumbel_max", "V=1K", f"{gumbel_max[1_000]:.1f} us"],
    ["gumbel_topk", "V=100", f"{gumbel_topk[100] * 1000:.0f} ns"],
    ["gumbel_topk", "V=1K", f"{gumbel_topk[1_000]:.1f} us"],
]

ax2.axis("off")
ax2.set_title("Latency (single call)", fontsize=12, fontweight="bold", pad=15)
table = ax2.table(
    cellText=table_data,
    colLabels=["sampler", "size", "latency"],
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.35)

for j in range(3):
    table[0, j].set_facecolor("#1e293b")
    table[0, j].set_text_props(color="white", fontweight="bold")
for i in range(1, len(table_data) + 1):
    for j in range(3):
        table[i, j].set_facecolor("#f8fafc" if i % 2 == 0 else "white")

fig.tight_layout(pad=2.0)
out = "/Users/arc/Documents/dev/drawset/docs/bench_throughput.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved {out}")
