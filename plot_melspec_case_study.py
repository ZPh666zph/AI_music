#!/usr/bin/env python3
"""
plot_melspec_case_study.py — 时序结构控制的 Mel-Spectrogram Case Study
======================================================================
上半: Prompt DSL 时序标注 (0-4s sparse/p, 4-6s crescendo, 6-10s dense/f)
下半: 模拟 Mel-Spectrogram 热力图 (符合物理规律的能量分布)
输出: PDF + PNG (IEEE 双栏横向长图)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import numpy as np
import os, sys

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# ── 学术样式 ──
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.06,
})

# ═══════════════════════════════════════════════════════════
# 1. 模拟 Mel-Spectrogram (符合物理规律)
# ═══════════════════════════════════════════════════════════
np.random.seed(42)

sr = 22050
duration = 10.0
hop_length = 512
n_mels = 96
n_frames = int(duration * sr / hop_length)  # ~430 帧

# 时间轴 (帧 → 秒)
times = np.arange(n_frames) * hop_length / sr

# 频率轴 (mel bins → Hz)
mel_edges = np.linspace(0, 8000, n_mels)  # 0-8kHz

# 能量包络: 三段时间域权重
# 段1 (0-4s): 稀疏低频, 低能量
# 段2 (4-6s): 渐强 (能量线性上升)
# 段3 (6-10s): 密集全频段, 高能量
energy_envelope = np.zeros(n_frames)
for i, t in enumerate(times):
    if t < 4.0:
        # 稀疏: 能量低, 且随时间轻微波动
        energy_envelope[i] = 0.20 + 0.05 * np.sin(2*np.pi*0.5*t)
    elif t < 6.0:
        # 渐强: 从 0.2 线性升到 0.9
        frac = (t - 4.0) / 2.0
        energy_envelope[i] = 0.20 + 0.70 * frac
    else:
        # 密集: 高能量, 轻微颤音
        energy_envelope[i] = 0.90 + 0.08 * np.sin(2*np.pi*3.0*t)

# 频率分布: 段1偏低频, 段3全频段
# 用高斯权重叠加: 每个时刻在特定 mel 频段产生谐波能量
mel_spec = np.zeros((n_mels, n_frames))

for i, t in enumerate(times):
    if t < 4.0:
        # 稀疏低频: 集中在低 mel bins, 稀疏谐波
        for harmonic in [1, 2]:
            center = 8 + harmonic * 4  # 低频段
            width = 3
            gauss = np.exp(-0.5 * ((np.arange(n_mels) - center) / width)**2)
            mel_spec[:, i] += energy_envelope[i] * gauss * (0.6 if harmonic == 1 else 0.35)
    elif t < 6.0:
        # 渐强: 频段逐步向上扩展
        frac = (t - 4.0) / 2.0
        center = 8 + 30 * frac  # 中心从低频向中频移动
        width = 6 + 8 * frac
        gauss = np.exp(-0.5 * ((np.arange(n_mels) - center) / width)**2)
        mel_spec[:, i] += energy_envelope[i] * gauss
    else:
        # 密集全频段: 多个谐波 + 宽带噪声
        for harmonic in range(1, 6):
            center = 10 + harmonic * 15
            width = 8
            gauss = np.exp(-0.5 * ((np.arange(n_mels) - center) / width)**2)
            mel_spec[:, i] += energy_envelope[i] * gauss * (0.5 / harmonic)
        # 宽带背景
        mel_spec[:, i] += energy_envelope[i] * 0.15

# 添加轻微噪声
mel_spec += np.random.randn(n_mels, n_frames) * 0.015

# 转 dB
mel_spec_db = 10 * np.log10(mel_spec + 1e-8)
mel_spec_db = np.clip(mel_spec_db, -60, 0)

# ═══════════════════════════════════════════════════════════
# 2. 绘图
# ═══════════════════════════════════════════════════════════
fig = plt.figure(figsize=(7.2, 2.6))
gs = gridspec.GridSpec(2, 1, height_ratios=[1.1, 3.0], hspace=0.25)

# ── 上半: DSL 时序标注 ──
ax_dsl = fig.add_subplot(gs[0])
ax_dsl.set_xlim(0, 10)
ax_dsl.set_ylim(0, 1)
ax_dsl.axis("off")

# 三段时间段的色块 + 标注
segments = [
    (0.0, 4.0, "[TEX:sparse] | [DYN:p]", "Close-up", "#E8F0FE", "#1A56DB"),
    (4.0, 6.0, "[DYN:CRESC]", "Zoom-in", "#FEF3E2", "#EA580C"),
    (6.0, 10.0, "[TEX:dense] | [DYN:f]", "Wide shot", "#FDE8E8", "#DC2626"),
]

for x0, x1, token, label, facecolor, edgecolor in segments:
    # 色块
    ax_dsl.add_patch(plt.Rectangle((x0, 0.08), x1-x0, 0.84,
                    facecolor=facecolor, edgecolor=edgecolor,
                    linewidth=1.2, alpha=0.85, zorder=1))
    # 中心文字: token + 镜头语言对应
    cx = (x0 + x1) / 2
    ax_dsl.text(cx, 0.68, token, ha="center", va="center",
                fontsize=7.5, fontweight="bold", color="#1F2937", zorder=2)
    ax_dsl.text(cx, 0.32, f"({label})", ha="center", va="center",
                fontsize=6.5, color=edgecolor, fontweight="bold", zorder=2)

# 时间刻度
ax_dsl.set_xticks(np.arange(0, 11, 1))
ax_dsl.set_xticklabels([f"{i}s" for i in range(11)], fontsize=6.5)
ax_dsl.tick_params(axis="x", length=0)

# ── 下半: Mel 频谱热力图 (纯 matplotlib, 不依赖 librosa/numba) ──
ax_spec = fig.add_subplot(gs[1])

img = ax_spec.imshow(
    mel_spec_db, aspect="auto", origin="lower",
    extent=[0, duration, 0, 8000],
    cmap="magma", vmin=-60, vmax=0, interpolation="bilinear",
)

y_ticks = np.arange(0, 8001, 2000)
ax_spec.set_yticks(y_ticks)
ax_spec.set_yticklabels([f"{int(t/1000)}k" for t in y_ticks])
ax_spec.set_xlim(0, duration)

# 时间分割线 (白色虚线)
for t_seg in [4.0, 6.0]:
    ax_spec.axvline(x=t_seg, color="white", linestyle="--",
                    linewidth=1.5, alpha=0.9, zorder=5)

# 轴标签
ax_spec.set_xlabel("Time (s)", fontweight="bold")
ax_spec.set_ylabel("Frequency (Hz)", fontweight="bold")

# 颜色条
cbar = fig.colorbar(img, ax=ax_spec, pad=0.015, shrink=0.9)
cbar.set_label("Magnitude (dB)", fontsize=7, labelpad=3)
cbar.ax.tick_params(labelsize=6)

# 标题
fig.suptitle("Mel-Spectrogram Case Study: Temporal Structural Control via Prompt DSL",
             fontsize=9, fontweight="bold", x=0.02, ha="left", y=0.98)

# ═══════════════════════════════════════════════════════════
# 3. 保存
# ═══════════════════════════════════════════════════════════
out_dir = "C:/Deepseek/outputs/melspec_case_study"
os.makedirs(out_dir, exist_ok=True)

for fmt in ["pdf", "png"]:
    path = f"{out_dir}/melspec_case_study.{fmt}"
    kwargs = {"bbox_inches": "tight", "pad_inches": 0.08}
    if fmt == "png":
        kwargs["dpi"] = 300
    fig.savefig(path, format=fmt, **kwargs)
    size_kb = os.path.getsize(path) / 1024
    print(f"Saved: {path} ({size_kb:.0f} KB)")

plt.close(fig)
print("Mel-Spectrogram Case Study 渲染完成")
