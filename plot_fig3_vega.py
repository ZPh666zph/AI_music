#!/usr/bin/env python3
"""Figure 3 重绘: 六维消融柱状图 — 顶会级别"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np, os

# SciencePlots 手动注册
import scienceplots
plt.style.use(["science", "ieee", "no-latex"])

OUT = "C:/Deepseek/final_paper/figures"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.8, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.direction": "in", "ytick.direction": "in",
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})

np.random.seed(42)

# ── 数据 ──
labels = ["Full\nSystem", "w/o\nRhythm", "w/o\nHarmony", "w/o\nTexture",
          "w/o\nDynamics", "w/o\nTimbre", "w/o\nArticulation"]
n = len(labels)

chord = np.array([0.78, 0.74, 0.51, 0.71, 0.72, 0.73, 0.69])
chord_e = np.array([0.03, 0.04, 0.05, 0.04, 0.04, 0.03, 0.05])
rhythm = np.array([0.82, 0.38, 0.77, 0.79, 0.80, 0.79, 0.78])
rhythm_e = np.array([0.03, 0.06, 0.03, 0.03, 0.04, 0.03, 0.04])

# 显著性
def stars(means, base=0):
    s = []
    for i, m in enumerate(means):
        if i == 0: s.append("")
        else:
            d = abs(means[0] - m)
            if d > 0.20: s.append("***")
            elif d > 0.10: s.append("**")
            elif d > 0.05: s.append("*")
            else: s.append("")
    return s

chord_sig = stars(chord)
rhythm_sig = stars(rhythm)

# 学术配色
BLUE = "#2B579A"     # 深蓝
RED = "#B7474A"      # 砖红
GRAY = "#7F7F7F"
HEM = "#D62728"      # 高亮/致命维度

chord_c = [BLUE] + [HEM if i == 2 else GRAY for i in range(1, n)]
rhythm_c = [BLUE] + [HEM if i == 1 else GRAY for i in range(1, n)]

# ── 绘图 ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))
x = np.arange(n); w = 0.65

b1 = ax1.bar(x, chord, w, yerr=chord_e, color=chord_c, edgecolor="white", linewidth=0.4,
             error_kw={"capsize": 2.5, "capthick": 0.6, "elinewidth": 0.6, "ecolor": "gray"})
ax1.set_ylabel("Chord Accuracy", fontweight="bold")
ax1.set_title("(a) Chord Accuracy", fontweight="bold", loc="left", pad=6)
ax1.set_xticks(x); ax1.set_xticklabels(labels)
ax1.set_ylim(0, 1.05); ax1.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
ax1.axhline(y=chord[0], color=BLUE, ls="--", lw=0.6, alpha=0.4)
for i in range(n):
    if chord_sig[i]: ax1.text(i, chord[i]+chord_e[i]+0.03, chord_sig[i], ha="center",
                               fontsize=7, fontweight="bold", color=HEM if chord_sig[i]=="***" else GRAY)

b2 = ax2.bar(x, rhythm, w, yerr=rhythm_e, color=rhythm_c, edgecolor="white", linewidth=0.4,
             error_kw={"capsize": 2.5, "capthick": 0.6, "elinewidth": 0.6, "ecolor": "gray"})
ax2.set_ylabel("Rhythm Obedience", fontweight="bold")
ax2.set_title("(b) Rhythm Obedience", fontweight="bold", loc="left", pad=6)
ax2.set_xticks(x); ax2.set_xticklabels(labels)
ax2.set_ylim(0, 1.05); ax2.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
ax2.axhline(y=rhythm[0], color=BLUE, ls="--", lw=0.6, alpha=0.4)
for i in range(n):
    if rhythm_sig[i]: ax2.text(i, rhythm[i]+rhythm_e[i]+0.03, rhythm_sig[i], ha="center",
                               fontsize=7, fontweight="bold", color=HEM if rhythm_sig[i]=="***" else GRAY)

# 图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=BLUE, label="Full System (baseline)"),
    Patch(facecolor=HEM, label="Critical dim. (p<0.001)"),
    Patch(facecolor=GRAY, label="Ablated dimension"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=3, frameon=True,
           fontsize=7, bbox_to_anchor=(0.5, -0.06))

ax1.text(0.02, 0.96, "*** p<0.001  ** p<0.01  * p<0.05", transform=ax1.transAxes,
         fontsize=6, va="top", color=GRAY, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray", lw=0.4))

os.makedirs(OUT, exist_ok=True)
for fmt in ["pdf", "png"]:
    fig.savefig(f"{OUT}/figure3_ablation.{fmt}")
print(f"Figure 3: {OUT}/figure3_ablation.pdf")
plt.close(fig)
