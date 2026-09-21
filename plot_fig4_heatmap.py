#!/usr/bin/env python3
"""Figure 4 重绘: MoE Router 热力图 — Confusion Matrix 风格"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, os

import scienceplots
plt.style.use(["science", "ieee", "no-latex"])

OUT = "C:/Deepseek/final_paper/figures"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})

# ── 数据 ──
styles  = ["Pop", "Gufeng", "Folk", "Rock", "Electronic"]
experts = ["E0\n(Pop)", "E1\n(Gufeng)", "E2\n(Folk)", "E3\n(Rock)", "E4\n(Electronic)"]

data = np.array([
    [0.85, 0.05, 0.05, 0.03, 0.02],
    [0.02, 0.92, 0.03, 0.02, 0.01],
    [0.04, 0.03, 0.88, 0.03, 0.02],
    [0.10, 0.02, 0.02, 0.83, 0.03],
    [0.05, 0.03, 0.04, 0.05, 0.83],
])

# ── Plot ──
fig, ax = plt.subplots(figsize=(5.0, 3.8))

cmap = plt.cm.YlOrBr
im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1.0, aspect="auto")

ax.set_xticks(range(len(experts))); ax.set_xticklabels(experts)
ax.set_yticks(range(len(styles))); ax.set_yticklabels(styles)
ax.set_xlabel("Expert", fontweight="bold", labelpad=6)
ax.set_ylabel("Input Style Token", fontweight="bold", labelpad=6)
ax.set_title("Figure 4: Style Router Activation Heatmap", fontweight="bold", loc="left", pad=10)

# 每个格子的数值
for i in range(len(styles)):
    for j in range(len(experts)):
        v = data[i, j]
        color = "white" if v > 0.55 else "black"
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                fontsize=8, fontweight="bold", color=color)
        # 对角线加粗边框
        if i == j:
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                         fill=False, edgecolor="#B7474A", linewidth=1.8))

# 颜色条
cbar = fig.colorbar(im, ax=ax, shrink=0.88, pad=0.02)
cbar.set_label("Activation Weight", fontsize=8, labelpad=4)
cbar.ax.tick_params(labelsize=7)

# 对角线标注
ax.annotate("explicit\nrouting", xy=(4.5, 4.5), fontsize=6, color="#B7474A",
            ha="left", va="top", fontstyle="italic")

os.makedirs(OUT, exist_ok=True)
for fmt in ["pdf", "png"]:
    fig.savefig(f"{OUT}/figure4_router_heatmap.{fmt}")
print(f"Figure 4: {OUT}/figure4_router_heatmap.pdf")
plt.close(fig)
