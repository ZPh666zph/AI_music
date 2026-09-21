# -*- coding: utf-8 -*-
"""
eval_moe_separation.py — 8 专家流派隔离与表征评测脚本
================================================
对应论文 Proposition 2（Style Separability under Explicit Routing，Eq. (9) 的
KL 下界）与 Fig. 4 / Fig. 6（Router 热力图 & 音色分布）。

功能
----
1. 载入 moe_8expert.pt（缺省回退 moe_5expert.pt），按风格喂 DSL 样本前向，
   收集每个专家输出层的特征向量（ExpertFFN 输出，d=64）。
2. 对每个风格估计均值 μ_k 与协方差 Σ_k（pooled + ridge），计算
   - 成对**对称 KL 散度**矩阵 D_sym(p||q) = ½[D_KL(p||q) + D_KL(q||p)]
   - 成对**余弦相似度**矩阵
3. 自动绘制两张出版级图（PDF + PNG，矢量可编辑）：
   - 图 1：8×8 路由混淆矩阵热力图（Confusion Matrix）
   - 图 2：特征距离（对称 KL / cosine dist）vs 声学物理特征差
     （|Δ频谱质心| Hz 与 |ΔRMS| dB）散点 + 线性拟合 + Pearson r/p（扩展论文 Fig. 9）

环境说明：本机 torch 2.7 与 numpy 1.26 存在 DLL 冲突，torch.from_numpy 不可用；
          所有 torch→numpy 转换经 .tolist() 中转，numpy/matplotlib 只做统计与绘图。
          声学特征优先读真实 wav（librosa），否则用每风格 anchor 表（论文 Fig. 5 数据）。

用法
----
  python eval_moe_separation.py
  python eval_moe_separation.py --ckpt outputs/checkpoints/moe_8expert.pt \
        --samples-per-style 32 --fig-dir outputs/figures
"""
import os
# 本机 Anaconda 同时 link 了 torch 与 librosa/matplotlib 的两份 OpenMP 运行时，
# 官方建议的 workaround（若不设置，进程会在 import 阶段直接 abort）。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import copy
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from model.train_moe import TokenEncoder, ExpertFFN, tokenize, VOCAB_SIZE, STYLE_MAP
from model.moe_expansion import (
    ExtensibleStyleRouter, OLD_STYLES, NEW_STYLES, ALL_STYLES, STYLE_TO_ID,
    synthesize_new_style_dsl,
)

warnings.filterwarnings("ignore")

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

torch.manual_seed(0)
np.random.seed(0)

# ─────────────────────────────────────────────────────────────
# 每风格声学物理特征 anchor 表（Hz 频谱质心 / dB RMS / dB 动态范围）
# 前 3 行取自论文 Fig. 5 实测（Erhu 1689 / Dizi 2132 / Suona 2720），
# 其余 5 个按乐器声学文献典型值锚定；提供 --audio-map 时会被真实测量覆盖。
# ─────────────────────────────────────────────────────────────
ACOUSTIC_ANCHORS = {
    "string":          (1689.0, -20.0, 12.0),   # 论文 Fig. 5: Erhu
    "wind":            (2132.0, -18.0, 14.0),   # 论文 Fig. 5: Dizi  +26% vs Erhu
    "brass":           (2720.0, -14.0, 24.0),   # 论文 Fig. 5: Suona +28% vs Dizi
    "electronic":      (1900.0, -10.0, 8.0),
    "piano":           (1050.0, -22.0, 30.0),   # 钢琴质心偏低、动态最大
    "classical_piano": (950.0,  -24.0, 34.0),
    "jazz_sax":        (1600.0, -16.0, 18.0),
    "rock_guitar":     (2200.0, -9.0,  15.0),
    "edm_synth":       (2600.0, -8.0,  9.0),
    "acoustic_bass":   (520.0,  -20.0, 16.0),
}


# ─────────────────────────────────────────────────────────────
# 模型装载（与 moe_expansion.MoE8Model 同构）
# ─────────────────────────────────────────────────────────────
class _EvalModel(nn.Module):
    """与 MoE8Model 同构的容器（保证 forward 签名一致）。"""

    def __init__(self, n_experts, d_model=64, d_ff=256):
        super().__init__()
        self.token_encoder = TokenEncoder()
        self.router = ExtensibleStyleRouter(n_experts)
        self.experts = nn.ModuleList([ExpertFFN(d_model, d_ff)
                                      for _ in range(n_experts)])
        self.head = nn.Linear(d_model, 1)

    def forward(self, token_ids, style_ids, return_features=False):
        tok = self.token_encoder(token_ids)
        tok = tok.mean(dim=1)[:, :64]
        logits_r = self.router.forward_logits(style_ids)
        w = F.softmax(logits_r, dim=-1)
        expert_outs = [expert(tok) for expert in self.experts]
        mixed = sum(w[:, i:i + 1] * expert_outs[i]
                    for i in range(len(self.experts)))
        out = self.head(mixed).squeeze(-1)
        if return_features:
            return out, w, logits_r, tok, mixed
        return out, w, logits_r


def load_model(ckpt_path: str):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    # 专家数 = experts.<i>.net.0.weight 的最大 i + 1
    n_experts = 0
    for k in sd:
        if k.startswith("experts.") and k.endswith(".net.0.weight"):
            i = int(k.split(".")[1])
            n_experts = max(n_experts, i + 1)
    assert n_experts > 0, "checkpoint 无 experts 参数"

    model = _EvalModel(n_experts)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, n_experts


# ─────────────────────────────────────────────────────────────
# 特征收集：每个风格 → 各专家输出特征矩阵
# ─────────────────────────────────────────────────────────────
def collect_features(model, n_experts, samples_per_style: int):
    """返回 features[style_idx][expert_idx] = FloatTensor [N, d]，
    和 router_weight_mean[style_idx] = [n_experts]。"""
    styles = ALL_STYLES[:n_experts]

    def _gen(style: str, k: int):
        if style in NEW_STYLES:
            return synthesize_new_style_dsl(style, k)["tokens"]
        # 旧风格用固定 DSL（与 modern_styles_train.jsonl 同语法）
        bpm = {"string": 70, "wind": 90, "brass": 110,
               "electronic": 140, "piano": 80}[style]
        return (f"[GLOBAL:STYLE:{style}]\n[GLOBAL:EXPERT:{style}]\n"
                f"[GLOBAL:TEMPO:{bpm}]\n[GLOBAL:KEY:C_major]\n[GLOBAL:TIME:4/4]\n\n"
                f"[SEC:intro|BAR:1-8]\n  [STEM:{style}]\n"
                f"    [DYN:mp|ART:legato|TEX:medium|TEMPO:{bpm}]\n  [/STEM]\n")

    features = [[[] for _ in range(n_experts)] for _ in range(n_experts)]
    wmeans = [None] * n_experts

    with torch.no_grad():
        for si, style in enumerate(styles):
            wsum = torch.zeros(n_experts)
            for k in range(samples_per_style):
                tok = tokenize(_gen(style, k)).unsqueeze(0)
                sid = torch.tensor([STYLE_TO_ID[style]], dtype=torch.long)
                out, w, rl, tok_feat, mixed = model.forward(
                    tok, sid, return_features=True)
                # 重新计算各专家输出（forward 内已算，但这里显式求，确保特征口径）
                expert_outs = [expert(tok_feat) for expert in model.experts]
                for ei in range(n_experts):
                    features[si][ei].append(expert_outs[ei].squeeze(0).clone())
                wsum += w.squeeze(0)
            wmeans[si] = wsum / samples_per_style

    F = [[torch.stack(features[si][ei]).float()
          for ei in range(n_experts)] for si in range(n_experts)]
    return F, wmeans, styles


# ─────────────────────────────────────────────────────────────
# 统计：μ / Σ / 对称 KL / 余弦（纯 torch，避免 numpy bridge）
# ─────────────────────────────────────────────────────────────
RIDGE = 1e-4   # 协方差正则，保证 d=64 时 Σ 可逆


def gaussian_stats(rows: torch.Tensor):
    mu = rows.mean(dim=0)                       # [d]
    X = rows - mu
    n = rows.shape[0]
    cov = (X.t() @ X) / max(n - 1, 1)           # [d, d]
    cov = cov + RIDGE * torch.eye(cov.shape[0])
    return mu, cov, n


def gauss_kl(mu1, cov1, mu2, cov2, n1, n2) -> float:
    """KL(N1 || N2) 的高斯闭式（d 维）。"""
    d = mu1.shape[0]
    cov2_inv = torch.linalg.inv(cov2)
    maha = (mu1 - mu2).unsqueeze(0) @ cov2_inv @ (mu1 - mu2).unsqueeze(1)
    tr = torch.trace(cov2_inv @ cov1)
    logdet = torch.logdet(cov2) - torch.logdet(cov1)
    kl = 0.5 * (tr + maha.squeeze() - d + logdet)
    return float(kl.item())


def symmetric_kl(mu1, cov1, mu2, cov2, n1, n2) -> float:
    k12 = gauss_kl(mu1, cov1, mu2, cov2, n1, n2)
    k21 = gauss_kl(mu2, cov2, mu1, cov1, n2, n1)
    return 0.5 * (k12 + k21)


def cosine_sim(mu1: torch.Tensor, mu2: torch.Tensor) -> float:
    return float(F.cosine_similarity(mu1.unsqueeze(0), mu2.unsqueeze(0)).item())


def compute_metrics(F, n_experts):
    """返回 {mu[], cov[], n[]} 与 KL 矩阵、cos 矩阵（上三角元素用于散点）。"""
    mus, covs, ns = [], [], []
    for si in range(n_experts):
        # 用该风格"主专家"（路由权重最大的专家）的输出层特征代表该风格
        rows = F[si][si]
        mu, cov, n = gaussian_stats(rows)
        mus.append(mu); covs.append(cov); ns.append(n)

    kl_mat = np.zeros((n_experts, n_experts))
    cos_mat = np.zeros((n_experts, n_experts))
    for i in range(n_experts):
        for j in range(n_experts):
            kl_mat[i, j] = symmetric_kl(mus[i], covs[i], mus[j], covs[j], ns[i], ns[j])
            cos_mat[i, j] = cosine_sim(mus[i], mus[j])
    return {"mu": mus, "cov": covs, "n": ns,
            "kl": kl_mat, "cos": cos_mat}


# ─────────────────────────────────────────────────────────────
# 声学物理特征：真实 wav 优先，否则 anchor 表
# ─────────────────────────────────────────────────────────────
def acoustic_features(styles, audio_map: dict):
    feats = {}
    for s in styles:
        if s in audio_map and Path(audio_map[s]).exists():
            feats[s] = _measure_wav(audio_map[s])
        else:
            feats[s] = ACOUSTIC_ANCHORS.get(s, (1000.0, -20.0, 12.0))
    return feats


def _measure_wav(path: str):
    try:
        import librosa
        y, sr = librosa.load(path, sr=22050, mono=True)
        S = np.abs(librosa.stft(y, n_fft=1024, hop_length=512))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
        centroid = float(np.sum(freqs[:, None] * S) / (np.sum(S) + 1e-8))
        rms = 20.0 * np.log10(float(np.sqrt(np.mean(y ** 2)) + 1e-8))
        dyn = float(np.percentile(
            20.0 * np.log10(np.abs(y) + 1e-8), 95)
            - np.percentile(20.0 * np.log10(np.abs(y) + 1e-8), 5))
        return (centroid, rms, dyn)
    except Exception as e:
        print(f"  [warn] 读取音频失败 {path}: {e}，回退 anchor")
        return ACOUSTIC_ANCHORS.get(Path(path).stem, (1000.0, -20.0, 12.0))


# ─────────────────────────────────────────────────────────────
# 图 1：8×8 路由混淆矩阵热力图
# ─────────────────────────────────────────────────────────────
def plot_confusion(wmeans, styles, out_dir: Path):
    n = len(styles)
    C = np.array([w.tolist() for w in wmeans])     # [n, n] 行=真实风格，列=预测专家
    # 归一化行和为 1（softmax 已保证，这里显式防止数值飘）
    C = C / (C.sum(axis=1, keepdims=True) + 1e-12)

    fig, ax = plt.subplots(figsize=(7.6, 6.8))
    im = ax.imshow(C, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(styles, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(styles, fontsize=9)
    ax.set_xlabel("Predicted expert (argmax of router)", fontsize=11)
    ax.set_ylabel("True style", fontsize=11)
    ax.set_title("8×8 Style–Expert Routing Confusion Matrix", fontsize=12)
    for i in range(n):
        for j in range(n):
            v = C[i, j]
            ax.text(j, i, f"{v:.2f}",
                    ha="center", va="center",
                    color="white" if v > 0.6 else "black", fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("P(expert j | style i)", fontsize=10)
    diag = float(np.diag(C).mean())
    ax.set_xlabel(f"Predicted expert (mean diagonal {diag:.2f})", fontsize=11)
    fig.tight_layout()
    out_pdf = out_dir / "fig1_routing_confusion_matrix.pdf"
    out_png = out_dir / "fig1_routing_confusion_matrix.png"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=150)
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  [图 1] {out_pdf}")
    return C


# ─────────────────────────────────────────────────────────────
# 图 2：特征距离 vs 声学特征差 散点拟合
# ─────────────────────────────────────────────────────────────
def _upper_tri(mat):
    n = mat.shape[0]
    rows, cols = np.triu_indices(n, k=1)
    return mat[rows, cols]


def _pearson(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    # Pearson r 与其双尾 p（t 分布近似，纯 numpy 实现避免依赖环境里损坏的 scipy）
    r = np.corrcoef(x, y)[0, 1]
    if np.isnan(r) or abs(r) >= 1.0 - 1e-12:
        return 0.0 if np.isnan(r) else float(r), 1.0 if abs(r) >= 1.0 - 1e-12 else 1.0
    n = len(x)
    t = r * np.sqrt((n - 2) / max(1e-12, 1 - r * r))
    p = 2.0 * _t_sf(abs(t), n - 2)
    return float(r), float(p)


def _t_sf(t: float, df: int) -> float:
    """Student-t 双尾 survival function 的数值近似（正则化不完全 beta）。"""
    from math import lgamma, exp
    df = max(df, 1)
    x = df / (df + t * t)
    # 用数值稳定公式：sf = 0.5 * I_x(df/2, 1/2)
    a, b = df / 2.0, 0.5
    ib = _betainc_reg(a, b, x)
    return 0.5 * ib


def _betainc_reg(a: float, b: float, x: float) -> float:
    """正则化不完全 beta I_x(a, b)，连分式 + 反射公式（Lentz，数值配方 betai）。"""
    import math
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _betainc_reg(b, a, 1.0 - x)
    logk = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
            + a * math.log(x) + b * math.log(1.0 - x))
    f = _betacf(a, b, x)
    return math.exp(logk) * f / a if a > 0 else 0.0


def _betacf(a: float, b: float, x: float, itmax: int = 200, eps: float = 1e-12) -> float:
    """不完全 beta 的连分式（修正 Lentz）。"""
    FPMIN = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def plot_distance_vs_acoustics(kl_mat, cos_mat, styles, feats, out_dir: Path):
    n = len(styles)
    idx = _upper_tri(np.zeros((n, n)))          # 上三角 (i<j) 对
    pairs = np.array(np.triu_indices(n, k=1)).T  # [P, 2]

    kl = _upper_tri(kl_mat)
    cos = _upper_tri(cos_mat)
    d_centroid = np.array([abs(feats[styles[i]][0] - feats[styles[j]][0])
                           for i, j in pairs], dtype=float)
    d_rms = np.array([abs(feats[styles[i]][1] - feats[styles[j]][1])
                      for i, j in pairs], dtype=float)
    cos_dist = 1.0 - cos

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 4.8))
    scats = []

    # (a) 对称 KL vs |Δ频谱质心|
    ax = axes[0]
    r, p = _pearson(np.log1p(kl), d_centroid)
    ax.scatter(np.log1p(kl), d_centroid, s=44, c="#1f77b4", alpha=0.8)
    m, b = np.polyfit(np.log1p(kl), d_centroid, 1)
    xs = np.linspace(np.log1p(kl).min(), np.log1p(kl).max(), 50)
    ax.plot(xs, m * xs + b, color="#d62728", lw=1.6)
    ax.set_xlabel("Symmetric KL divergence, log(1+KL)")
    ax.set_ylabel("|Δ spectral centroid| (Hz)")
    ax.set_title(f"(a) KL vs centroid  r={r:.2f}, p={p:.2e}")

    # (b) cosine distance vs |Δ频谱质心|
    ax = axes[1]
    r, p = _pearson(cos_dist, d_centroid)
    ax.scatter(cos_dist, d_centroid, s=44, c="#2ca02c", alpha=0.8)
    m, b = np.polyfit(cos_dist, d_centroid, 1)
    xs = np.linspace(cos_dist.min(), cos_dist.max(), 50)
    ax.plot(xs, m * xs + b, color="#d62728", lw=1.6)
    ax.set_xlabel("Cosine distance, 1−cos(μ_i, μ_j)")
    ax.set_ylabel("|Δ spectral centroid| (Hz)")
    ax.set_title(f"(b) Cosine dist vs centroid  r={r:.2f}, p={p:.2e}")

    # (c) 对称 KL vs |ΔRMS|
    ax = axes[2]
    r, p = _pearson(np.log1p(kl), d_rms)
    ax.scatter(np.log1p(kl), d_rms, s=44, c="#9467bd", alpha=0.8)
    m, b = np.polyfit(np.log1p(kl), d_rms, 1)
    xs = np.linspace(np.log1p(kl).min(), np.log1p(kl).max(), 50)
    ax.plot(xs, m * xs + b, color="#d62728", lw=1.6)
    ax.set_xlabel("Symmetric KL divergence, log(1+KL)")
    ax.set_ylabel("|Δ RMS energy| (dB)")
    ax.set_title(f"(c) KL vs RMS  r={r:.2f}, p={p:.2e}")

    # 共享右下角说明
    fig.suptitle("Feature-space separation vs acoustic physical distance "
                 f"({n} styles, {n * (n - 1) // 2} pairs)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out_pdf = out_dir / "fig2_feature_distance_vs_acoustics.pdf"
    out_png = out_dir / "fig2_feature_distance_vs_acoustics.png"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=150)
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  [图 2] {out_pdf}")
    return kl, cos, d_centroid, d_rms


# ─────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="8 专家 MoE 分离度评测")
    ap.add_argument("--ckpt", default="outputs/checkpoints/moe_8expert.pt")
    ap.add_argument("--samples-per-style", type=int, default=32)
    ap.add_argument("--fig-dir", default="outputs/figures")
    ap.add_argument("--audio-map", default=None,
                    help="JSON: {style: wav_path, ...}，缺省用 anchor 表")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        fallback = Path("outputs/checkpoints/moe_5expert.pt")
        if fallback.exists():
            print(f"[warn] 未找到 {ckpt}，回退 {fallback}")
            ckpt = fallback
        else:
            print("错误: 找不到任何 MoE checkpoint（先运行 moe_expansion.py）")
            return 2

    print("=" * 62)
    print("MoE 流派隔离与表征评测")
    print(f"  ckpt: {ckpt}")
    print("=" * 62)

    model, n_experts = load_model(str(ckpt))
    styles = ALL_STYLES[:n_experts]
    print(f"  专家数: {n_experts}  风格: {styles}")
    print(f"  每风格采样: {args.samples_per_style}")

    F, wmeans, _ = collect_features(model, n_experts, args.samples_per_style)
    stats = compute_metrics(F, n_experts)
    kl_mat, cos_mat = stats["kl"], stats["cos"]

    audio_map = {}
    if args.audio_map:
        try:
            audio_map = json.loads(Path(args.audio_map).read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[warn] audio-map 解析失败: {e}")
    feats = acoustic_features(styles, audio_map)

    print("\n  对称 KL 矩阵（对角≈0，越大越分离）:")
    print("    " + " ".join(f"{s[:6]:>8}" for s in styles))
    for i, s in enumerate(styles):
        print(f"  {s[:6]:>6} " + " ".join(f"{kl_mat[i, j]:8.2f}" for j in range(n_experts)))
    print("\n  余弦相似度矩阵（对角=1）:")
    print("    " + " ".join(f"{s[:6]:>8}" for s in styles))
    for i, s in enumerate(styles):
        print(f"  {s[:6]:>6} " + " ".join(f"{cos_mat[i, j]:8.3f}" for j in range(n_experts)))

    out_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    C = plot_confusion(wmeans, styles, out_dir)
    diagonal = float(np.diag(C).mean())
    kl, cos, d_c, d_r = plot_distance_vs_acoustics(kl_mat, cos_mat, styles,
                                                   feats, out_dir)

    # 汇总指标（供写论文引用）
    off_kl = kl_mat[np.triu_indices(n_experts, k=1)].mean()
    off_cos = 1.0 - cos_mat[np.triu_indices(n_experts, k=1)].mean()
    print("\n  ================= 汇总 =================")
    print(f"  路由对角线均值:                 {diagonal:.4f}  (>0.8 期望)")
    print(f"  离对角对称 KL 均值:             {off_kl:.3f}")
    print(f"  离对角 cosine distance 均值:    {off_cos:.3f}")
    print(f"  σ_centroid (8 风格):            {np.std([feats[s][0] for s in styles]):.1f} Hz")
    print(f"  σ_RMS     (8 风格):            {np.std([feats[s][1] for s in styles]):.2f} dB")
    print(f"  图片目录: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
