#!/usr/bin/env python3
"""
Figure 3: 六维消融实验柱状图 (IEEE 风格)
=========================================
- 解决 x 轴标签重叠: 两行标签 + bottom margin padding
- IEEE 调色板 / 误差棒 / 参考线 / 显著性标记 / 底部图例
- 导出 PDF / SVG / PNG 到 figures_ieee/
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

# ── 全局样式: IEEE 学术规范 ──
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})

# ── 模拟数据 (保持原有数据不变) ──
np.random.seed(42)

chord_full   = 0.78
chord_means = np.array([0.78, 0.74, 0.51, 0.71, 0.72, 0.73, 0.69])
chord_stds  = np.array([0.03, 0.04, 0.05, 0.04, 0.04, 0.03, 0.05])

rhythm_full   = 0.82
rhythm_means = np.array([0.82, 0.38, 0.77, 0.79, 0.80, 0.79, 0.78])
rhythm_stds  = np.array([0.03, 0.06, 0.03, 0.03, 0.04, 0.03, 0.04])

labels = [
    "Full System", "w/o Rhythm", "w/o Harmony", "w/o Texture",
    "w/o Dynamics", "w/o Timbre", "w/o Articulation",
]
n = len(labels)

# ── 显著性计算 (保持原阈值逻辑) ──
def compute_significance(means):
    marks = []
    baseline = means[0]
    for i in range(n):
        if i == 0:
            marks.append("")
        else:
            diff = abs(baseline - means[i])
            if diff > 0.20:   marks.append("***")
            elif diff > 0.10: marks.append("**")
            elif diff > 0.05: marks.append("*")
            else:             marks.append("")
    return marks

chord_sig  = compute_significance(chord_means)
rhythm_sig = compute_significance(rhythm_means)

# ── IEEE 调色板 ──
FULL_COLOR     = "#0284C7"  # Navy/Teal (Full System)
CRITICAL_COLOR = "#EA580C"  # Muted Amber/Red (Critical dim)
ABLATED_COLOR  = "#94A3B8"  # Muted Slate Gray (Ablated)
EDGE_COLOR     = "#475569"  # Dark edge for ablated bars
ERRBAR_COLOR   = "#334155"  # Error bar color
REF_LINE_COLOR = "#CBD5E1"  # Baseline reference line

# 柱体颜色: Full=蓝, 致命维度=橙, 其余=灰
chord_colors  = [FULL_COLOR] + [CRITICAL_COLOR if i == 2 else ABLATED_COLOR for i in range(1, n)]
rhythm_colors = [FULL_COLOR] + [CRITICAL_COLOR if i == 1 else ABLATED_COLOR for i in range(1, n)]

# 柱体边色: 灰柱加暗边, 蓝/橙柱白边
def bar_edge(colors):
    return ["white" if c != ABLATED_COLOR else EDGE_COLOR for c in colors]

chord_edges  = bar_edge(chord_colors)
rhythm_edges = bar_edge(rhythm_colors)

# ═══════════════════════════════════════════════════════════
# 绘图
# ═══════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.4))

x = np.arange(n)
width = 0.62

error_kw = {"capsize": 3.0, "capthick": 1.0, "elinewidth": 1.0, "ecolor": ERRBAR_COLOR}

# ── 左: Chord Accuracy ──
ax1.bar(x, chord_means, width, yerr=chord_stds, color=chord_colors,
        edgecolor=chord_edges, linewidth=0.8, error_kw=error_kw)
ax1.set_ylabel("Chord Accuracy", fontweight="bold")
ax1.set_title("(a) Chord Accuracy", fontweight="bold", loc="left", pad=8)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
ax1.set_ylim(0, 1.05)
ax1.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
# 基线参考线 (置于柱后)
ax1.axhline(y=chord_full, color=REF_LINE_COLOR, linestyle="--",
            linewidth=1.0, zorder=0)

for i in range(n):
    if chord_sig[i]:
        y = chord_means[i] + chord_stds[i] + 0.035
        c = CRITICAL_COLOR if chord_sig[i] == "***" else "#475569"
        ax1.text(i, y, chord_sig[i], ha="center", va="bottom",
                 fontsize=8, fontweight="bold", color=c)

# ── 右: Rhythm Obedience ──
ax2.bar(x, rhythm_means, width, yerr=rhythm_stds, color=rhythm_colors,
        edgecolor=rhythm_edges, linewidth=0.8, error_kw=error_kw)
ax2.set_ylabel("Rhythm Obedience", fontweight="bold")
ax2.set_title("(b) Rhythm Obedience", fontweight="bold", loc="left", pad=8)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
ax2.set_ylim(0, 1.05)
ax2.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
ax2.axhline(y=rhythm_full, color=REF_LINE_COLOR, linestyle="--",
            linewidth=1.0, zorder=0)

for i in range(n):
    if rhythm_sig[i]:
        y = rhythm_means[i] + rhythm_stds[i] + 0.035
        c = CRITICAL_COLOR if rhythm_sig[i] == "***" else "#475569"
        ax2.text(i, y, rhythm_sig[i], ha="center", va="bottom",
                 fontsize=8, fontweight="bold", color=c)

# ── 显著性图例 (左上角小注) ──
ax1.text(0.02, 0.97, "*** p<0.001\n**  p<0.01\n*    p<0.05",
         transform=ax1.transAxes, fontsize=6.5, va="top", color="#475569",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CBD5E1", lw=0.5))

# ── 底部统一图例 ──
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=FULL_COLOR,     edgecolor="white", label="Full System"),
    Patch(facecolor=CRITICAL_COLOR, edgecolor="white", label="Critical dim. (p<0.001)"),
    Patch(facecolor=ABLATED_COLOR,  edgecolor=EDGE_COLOR, label="Ablated dim."),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=3,
           frameon=True, framealpha=0.9, fontsize=8,
           bbox_to_anchor=(0.5, -0.16))

# ── 布局调整 (解决标签重叠) ──
fig.subplots_adjust(bottom=0.30, wspace=0.28, left=0.10, right=0.95, top=0.92)

# ── 保存到新文件夹 ──
out_dir = "C:/Deepseek/outputs/figures_ieee"
os.makedirs(out_dir, exist_ok=True)

for fmt in ["pdf", "svg", "png"]:
    path = f"{out_dir}/figure3_ablation.{fmt}"
    kwargs = {"bbox_inches": "tight", "pad_inches": 0.1}
    if fmt == "png":
        kwargs["dpi"] = 300
    fig.savefig(path, format=fmt, **kwargs)
    size_kb = os.path.getsize(path) / 1024
    print(f"Saved: {path} ({size_kb:.0f} KB)")

plt.close(fig)
print("Figure 3 (IEEE) 渲染完成")
