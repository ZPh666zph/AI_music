# -*- coding: utf-8 -*-
"""
eval_metrics_pipeline.py — 客观指标自动化评测管线（TMM 对比表用）
==================================================================
集成的客观指标（全部实现，缺重依赖时优雅降级为 N/A 并记录原因）：

  FAD (VGGish)         — frechet-audio-distance 包（首跑下载 VGGish）
  FAD (CLAP-embed)     — laion-clap 包提供的音频嵌入上的 Fréchet 距离
  CLAP Score           — 文本-音频语义对齐度（laion-clap）
  BPM Error (↓)        — |detected_bpm - target_bpm| 绝对误差（librosa beat tracking）
  Chord Accuracy (↑)   — 优先 musiclang（符号级和弦识别），否则 librosa chroma 模板匹配
  RMS Correlation (↑)  — 分段 RMS 包络与 [DYN:] token 目标序列的 Pearson r
  Spectral Centroid MAE (↓) — 预测质心 vs [COLOR:] 目标质心的平均绝对误差

输入约定
--------
  每个模型一个目录，目录内是生成的音频，并（可选）带 manifest.jsonl：
     {"audio": "xxx.wav", "text": "prompt", "tokens": "DSL...", "target_bpm": 120}
  - tokens 里的 [GLOBAL:TEMPO:n] / [DYN:...] / [COLOR:...] / [CHORD:...] 作为 ground-truth；
  - FAD 需要一个参考集目录（真实音乐，如 MusicCaps 子集）；
  - CLAP Score 需要 manifest 中的 text 字段。

输出
----
  outputs/eval_metrics_results.md   对比表（Markdown）
  outputs/eval_metrics_results.tex  对比表（LaTeX，可直接 \input 或复制进论文 Table）
  outputs/eval_metrics_results.json 完整数值

用法
----
  python eval_metrics_pipeline.py \
      --models '{ "Ours": "outputs/eval_audio_ours",
                  "Stable Audio Open": "outputs/eval_audio_sao",
                  "AudioLDM 2": "outputs/eval_audio_audioldm2",
                  "MusiConGen": "outputs/eval_audio_musi", 
                  "SegTune": "outputs/eval_audio_segtune" }' \
      --ref data/musiccaps_audio --only bpm,chord,rms,centroid   # 无重依赖时的快速子集
"""
import argparse
import json
import os
import re
import sys
import warnings
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

try:
    import librosa
except Exception:  # pragma: no cover
    librosa = None

try:
    import soundfile as sf
except Exception:  # pragma: no cover
    sf = None

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
DYN_MAP = {"pp": 0, "p": 1, "mp": 2, "mf": 3, "f": 4, "ff": 5}
COLOR_CENTROID = {"warm": 400.0, "dark": 400.0, "cold": 700.0,
                  "neutral": 2500.0, "bright": 5500.0}

METRIC_HELP = {
    "fad_vggish": ("FAD (VGGish) ↓", "Frechet Audio Distance on VGGish embeddings (frechet-audio-distance)"),
    "fad_clap": ("FAD (CLAP) ↓", "Frechet distance on CLAP audio embeddings (laion-clap)"),
    "clap": ("CLAP Score ↑", "mean text-audio cosine similarity via LAION-CLAP"),
    "bpm": ("BPM Error ↓", "mean |detected - target| BPM (autocorrelation beat tracking on onset flux)"),
    "chord": ("Chord Accuracy ↑", "chroma-template chord match (musiclang if installed)"),
    "rms": ("RMS Correlation ↑", "Pearson r between frame RMS envelope and [DYN:] target"),
    "centroid": ("Spectral Centroid MAE ↓", "mean |predicted centroid - COLOR target| Hz"),
}


def _load_audio(path: str):
    if sf is None:
        raise RuntimeError("soundfile 未安装")
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y.astype(np.float32), sr


# ─────────────────────────────────────────────────────────────
# 音频读取（纯 soundfile + numpy，绕开本机 librosa/scipy 的 DLL 冲突）
# ─────────────────────────────────────────────────────────────
TARGET_SR = 22050


def _read_22050(path: str):
    """读音频并重采样到 22050（numpy 线性插值，不依赖 scipy）。"""
    if sf is None:
        raise RuntimeError("soundfile 未安装")
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    if sr != TARGET_SR:
        n = int(round(len(y) * TARGET_SR / sr))
        x_old = np.linspace(0.0, 1.0, len(y), endpoint=False)
        x_new = np.linspace(0.0, 1.0, n, endpoint=False)
        y = np.interp(x_new, x_old, y).astype(np.float32)
    else:
        n = len(y)
    return y, TARGET_SR


def _stft(y, sr=TARGET_SR, n_fft=2048, hop=512):
    """纯 numpy STFT（幅度谱）。"""
    n_fft = int(n_fft)
    hop = int(hop)
    win = np.hanning(n_fft).astype(np.float32)
    frames = []
    for i in range(0, len(y) - n_fft, hop):
        frames.append(np.abs(np.fft.rfft(y[i:i + n_fft] * win)))
    return np.array(frames).T if frames else np.zeros((n_fft // 2 + 1, 0))  # [F, T]


def _onset_env(y, sr=TARGET_SR, n_fft=2048, hop=512):
    """谱通量 onset 强度包络（纯 numpy）。"""
    S = _stft(y, sr, n_fft, hop)                 # [F, T]
    flux = np.diff(np.maximum(S[:, 1:], 0.0) - np.maximum(S[:, :-1], 0.0), axis=1)
    env = np.mean(np.maximum(flux, 0.0), axis=0)
    env = env / (np.max(env) + 1e-8)
    return env


def _autocorr_bpm(env, sr=TARGET_SR, hop=512):
    """自相关 + 分箱跟踪 → BPM（纯 numpy，等价 librosa.beat.tempo 的思路）。"""
    env = env - np.mean(env)
    ac = np.correlate(env, env, mode="full")[len(env) - 1:]
    ac = ac / (ac[0] + 1e-8)
    # 搜索 60-200 BPM 对应延迟区间
    bpm_min, bpm_max = 50, 220
    lo = max(1, int((60.0 / bpm_max) / (hop / sr)))
    hi = min(len(ac) - 1, int((60.0 / bpm_min) / (hop / sr)))
    if hi <= lo:
        return 120.0
    idx = lo + int(np.argmax(ac[lo:hi]))
    return 60.0 / (idx * hop / sr)


def detect_bpm(path: str):
    y, sr = _read_22050(path)
    env = _onset_env(y, sr)
    return _autocorr_bpm(env, sr)


def bpm_error(path: str, target_bpm):
    if target_bpm is None:
        return None
    return abs(detect_bpm(path) - float(target_bpm))


# ─────────────────────────────────────────────────────────────
# 2. Chord Accuracy（musiclang 优先，chroma 模板兜底）
# ─────────────────────────────────────────────────────────────
def chroma_template_chords(path: str, win_sec: float = 1.0):
    """纯 numpy 的 chroma（CQT 替代）：每帧加窗从 STFT 得到 12 音级能量。"""
    y, sr = _read_22050(path)
    S = _stft(y, sr, n_fft=4096, hop=512)
    freqs = np.fft.rfftfreq(4096, d=1.0 / sr)          # [F]
    # 每帧把幅度分配到 12 个音级（按 octave fold）
    chroma = np.zeros((12, S.shape[1]))
    for p in range(12):
        mask = np.zeros_like(freqs, dtype=bool)
        for oct_off in range(0, 8):
            f = 440.0 * 2 ** (p / 12.0 - 1) * 2 ** oct_off
            lo, hi = f / 2 ** (0.5 / 12), f * 2 ** (0.5 / 12)
            if lo < freqs[-1]:
                mask |= (freqs >= lo) & (freqs < hi)
        chroma[p] = S[mask].sum(axis=0)
    chroma = chroma / (chroma.sum(axis=0, keepdims=True) + 1e-8)

    hop = 512 / sr
    win = max(int(win_sec / hop), 2)
    out = []
    for t in range(0, chroma.shape[1], win):
        c = chroma[:, t:t + win].mean(axis=1)
        top3 = set(np.argsort(c)[-3:].tolist())
        for root in top3:
            if (root + 4) % 12 in top3 and (root + 7) % 12 in top3:
                out.append(PITCH_NAMES[root])
                break
            if (root + 3) % 12 in top3 and (root + 7) % 12 in top3:
                out.append(PITCH_NAMES[root] + ":min")
                break
        else:
            out.append(PITCH_NAMES[int(np.argmax(c))])
    return out


def musiclang_chords(path: str):
    """musiclang 符号级和弦识别（若安装）。"""
    from musiclang_predict import predict_chords  # musiclang 的推理入口
    chords = predict_chords(path, chord_duration=1.0)
    return [str(c) for c in chords]


def chord_accuracy(path: str, target_chords):
    if not target_chords:
        return None
    try:
        import musiclang_predict  # noqa: F401
        pred = musiclang_chords(path)
    except Exception:
        pred = chroma_template_chords(path)
    pred_roots = [c.split(":")[0] for c in pred]
    tgt_roots = [c.split(":")[0] for c in target_chords]
    n = max(len(tgt_roots), 1)
    hit = sum(1 for i, tr in enumerate(tgt_roots)
              if tr in [pred_roots[j % len(pred_roots)]
                        for j in range(max(0, i - 1), min(len(pred_roots), i + 2))])
    return hit / n


# ─────────────────────────────────────────────────────────────
# 3. RMS Correlation
# ─────────────────────────────────────────────────────────────
def rms_correlation(path: str, dyn_targets):
    if not dyn_targets or len(dyn_targets) < 2:
        return None
    y, _ = _read_22050(path)
    seg = len(y) // len(dyn_targets)
    rms = [float(np.sqrt(np.mean(y[i * seg:(i + 1) * seg] ** 2)))
           for i in range(len(dyn_targets))]
    return float(np.corrcoef(rms, dyn_targets)[0, 1])


# ─────────────────────────────────────────────────────────────
# 4. Spectral Centroid MAE
# ─────────────────────────────────────────────────────────────
def spectral_centroid_mae(path: str, color_target):
    if color_target is None:
        return None
    y, sr = _read_22050(path)
    S = _stft(y, sr, n_fft=2048, hop=512)
    freqs = np.fft.rfftfreq(2048, d=1.0 / sr)
    pred = float(np.sum(freqs[:, None] * S) / (np.sum(S) + 1e-8))
    expected = COLOR_CENTROID.get(color_target, 2500.0)
    return abs(pred - expected)


# ─────────────────────────────────────────────────────────────
# 5/6. FAD（VGGish 与 CLAP 两个后端）与 CLAP Score
# ─────────────────────────────────────────────────────────────
def fad_vggish(gen_dir: str, ref_dir: str):
    from frechet_audio_distance import FrechetAudioDistance
    fad = FrechetAudioDistance(model_name="vggish", sample_rate=16000)
    return float(fad.score(gen_dir, ref_dir, dtype="float32"))


def _clap_embeddings(dir_or_manifest, model):
    """用 LAION-CLAP 提取目录内音频嵌入。"""
    import torch
    embs = []
    files = sorted(Path(dir_or_manifest).glob("*.wav")) + \
        sorted(Path(dir_or_manifest).glob("*.mp3")) + \
        sorted(Path(dir_or_manifest).glob("*.flac"))
    for f in files:
        with torch.no_grad():
            e = model.get_audio_embedding_from_filelist(x=[str(f)],
                                                        use_tensor=True).squeeze(0)
        embs.append(e.cpu())
    return torch.stack(embs)


def fad_clap(gen_dir: str, ref_dir: str):
    """CLAP 音频嵌入上的 Fréchet 距离。"""
    import laion_clap
    import torch
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    model.load_ckpt()
    g = _clap_embeddings(gen_dir, model)
    r = _clap_embeddings(ref_dir, model)
    mu1, mu2 = g.mean(0), r.mean(0)
    s1 = torch.cov(g.T); s2 = torch.cov(r.T)
    diff = mu1 - mu2
    # 数值稳定：取两个协方差的均方根
    covmean = torch.linalg.sqrtm(s1 @ s2).real
    with torch.no_grad():
        val = float((diff @ diff).item()) + float(torch.trace(s1 + s2 - 2 * covmean).item())
    return max(val, 0.0)


def clap_score(manifest_path: Path):
    """按 manifest 逐条计算文本-音频相似度，取均值。"""
    import laion_clap
    import torch
    import soundfile as _sf
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    model.load_ckpt()
    entries = [json.loads(l) for l in open(manifest_path, encoding="utf-8")
               if l.strip()]
    sims = []
    for e in entries:
        y, sr = _sf.read(str(manifest_path.parent / e["audio"]))
        audio = y.astype(np.float32).reshape(1, -1)
        with torch.no_grad():
            sims.append(float(model.get_similarity(audio, [e["text"]])[0][0]))
    return float(np.mean(sims))


# ─────────────────────────────────────────────────────────────
# 单模型全指标
# ─────────────────────────────────────────────────────────────
def evaluate_model(name: str, model_dir: str, ref_dir: str, only: set):
    d = Path(model_dir)
    res = {"model": name, "n_audio": 0}
    manifest = d / "manifest.jsonl"

    if not d.exists():
        res["error"] = "目录不存在"
        return res

    # 逐条可计算指标（有 manifest 索引时按条算，否则目录全算但缺目标值为 None）
    entries = [json.loads(l) for l in open(manifest, encoding="utf-8") if l.strip()] \
        if manifest.exists() else []
    audio_files = entries or [
        {"audio": p.name} for p in sorted(d.glob("*.wav")) + sorted(d.glob("*.mp3"))
    ]
    res["n_audio"] = len(audio_files)

    if "bpm" in only or "chord" in only or "rms" in only or "centroid" in only:
        bpm_errs, chord_accs, rms_corrs, cent_maes = [], [], [], []
        for e in audio_files:
            ap = str(d / e["audio"]) if not Path(e["audio"]).is_absolute() else e["audio"]
            if not Path(ap).exists():
                continue
            tok = e.get("tokens", "")
            tgt_bpm = e.get("target_bpm") or _re_num(tok, r"\[GLOBAL:TEMPO:(\d+)\]") \
                or _re_num(tok, r"\[TEMPO:(\d+)\]")
            tgt_chords = re.findall(r"\[CHORD:([^\]]+)\]", tok)
            tgt_dyn = [DYN_MAP.get(m.group(1).split("|")[0])
                       for m in re.finditer(r"\[DYN:([^\]]+)\]", tok)]
            tgt_dyn = [v for v in tgt_dyn if v is not None]
            tgt_color = None
            mc = re.search(r"\[COLOR:([^\]]+)\]", tok)
            if mc:
                tgt_color = mc.group(1).split("|")[0]

            if "bpm" in only and tgt_bpm is not None:
                bpm_errs.append(bpm_error(ap, tgt_bpm))
            if "chord" in only and tgt_chords:
                chord_accs.append(chord_accuracy(ap, tgt_chords))
            if "rms" in only and len(tgt_dyn) >= 2:
                rms_corrs.append(rms_correlation(ap, tgt_dyn))
            if "centroid" in only and tgt_color:
                cent_maes.append(spectral_centroid_mae(ap, tgt_color))

        res["bpm_error"] = float(np.mean(bpm_errs)) if bpm_errs else None
        res["chord_accuracy"] = float(np.mean(chord_accs)) if chord_accs else None
        res["rms_correlation"] = float(np.mean(rms_corrs)) if rms_corrs else None
        res["centroid_mae"] = float(np.mean(cent_maes)) if cent_maes else None

    # FAD / CLAP（重依赖）
    for key in ("fad_vggish", "fad_clap", "clap"):
        if key not in only:
            continue
        try:
            if key == "fad_vggish":
                res[key] = fad_vggish(model_dir, ref_dir) if ref_dir else None
            elif key == "fad_clap":
                res[key] = fad_clap(model_dir, ref_dir) if ref_dir else None
            else:
                res[key] = clap_score(manifest) if manifest.exists() else None
        except ImportError as ie:
            res[key] = None
            res[f"{key}_unavailable"] = f"缺少依赖: {ie.name}"
        except Exception as ex:
            res[key] = None
            res[f"{key}_unavailable"] = f"{type(ex).__name__}: {str(ex)[:80]}"

    return res


def _re_num(txt, pat):
    m = re.search(pat, txt)
    return float(m.group(1)) if m else None


# ─────────────────────────────────────────────────────────────
# 表格输出
# ─────────────────────────────────────────────────────────────
def _fmt(v, digits=3, na="N/A"):
    return na if v is None else f"{v:.{digits}f}"


def write_markdown(results, out: Path):
    cols = ["model", "n_audio", "fad_vggish", "fad_clap", "clap",
            "bpm_error", "chord_accuracy", "rms_correlation", "centroid_mae"]
    header = ["Model", "#Audio", "FAD(VGG)↓", "FAD(CLAP)↓", "CLAP↑",
              "BPM Err↓", "Chord Acc↑", "RMS Corr↑", "Centroid MAE↓"]
    lines = ["# Objective Evaluation Results", "",
             "| " + " | ".join(header) + " |",
             "|" + "---|" * len(header)]
    for r in results:
        row = [r.get("model", ""), str(r.get("n_audio", "")),
               _fmt(r.get("fad_vggish")), _fmt(r.get("fad_clap")),
               _fmt(r.get("clap")), _fmt(r.get("bpm_error")),
               _fmt(r.get("chord_accuracy")), _fmt(r.get("rms_correlation")),
               _fmt(r.get("centroid_mae"))]
        lines.append("| " + " | ".join(row) + " |")
    # 脚注
    lines += ["", "## Metric notes",
              *[f"- **{k}** {v[1]}" for k, v in METRIC_HELP.items()],
              "",
              "> N/A: 指标因缺少对应第三方依赖（frechet-audio-distance / laion-clap / musiclang）"
              " 或缺少 manifest/text/DSL 目标而未计算。", ""]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [Markdown] {out}")


def write_latex(results, out: Path):
    esc = lambda s: str(s).replace("_", r"\_")
    header = ["Model", "#Audio", "FAD(VGG)$\\downarrow$", "CLAP$\\uparrow$",
              "BPM Err$\\downarrow$", "Chord Acc$\\uparrow$",
              "RMS Corr$\\uparrow$", "Centroid MAE$\\downarrow$"]
    body = []
    for r in results:
        row = [esc(r.get("model", "")), str(r.get("n_audio", "")),
               _fmt(r.get("fad_vggish")), _fmt(r.get("clap")),
               _fmt(r.get("bpm_error")), _fmt(r.get("chord_accuracy")),
               _fmt(r.get("rms_correlation")), _fmt(r.get("centroid_mae"))]
        body.append(" & ".join(row) + r" \\")
    tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Objective evaluation against open-source controllable music "
        r"generation baselines. FAD/CLAP follow the MusicCaps protocol; "
        r"BPM Error, Chord Accuracy, RMS Correlation and Spectral Centroid MAE are "
        r"computed against the six-axis DSL ground-truth of each generated track.}",
        r"\label{tab:objective}",
        r"\begin{tabular}{l" + "r" * (len(header) - 1) + "}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
        "\n".join(body),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    out.write_text("\n".join(tex), encoding="utf-8")
    print(f"  [LaTeX] {out}")


# ─────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="客观指标自动化评测管线")
    ap.add_argument("--models", required=True,
                    help="JSON: {模型名: 生成音频目录, ...}")
    ap.add_argument("--ref", default=None, help="FAD 参考集目录（真实音乐）")
    ap.add_argument("--only", default="all",
                    help="逗号分隔的指标子集: fad_vggish,fad_clap,clap,bpm,chord,rms,centroid")
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()

    models = json.loads(args.models)
    only = set(args.only.replace(" ", "").split(",")) if args.only != "all" \
        else set(METRIC_HELP.keys())

    results = [evaluate_model(name, d, args.ref, only)
               for name, d in models.items()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_markdown(results, out_dir / "eval_metrics_results.md")
    write_latex(results, out_dir / "eval_metrics_results.tex")
    (out_dir / "eval_metrics_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # 控制台汇总
    print("\n" + "=" * 70)
    for r in results:
        print(f"  {r.get('model')}: n={r.get('n_audio')}  "
              f"FAD={_fmt(r.get('fad_vggish'))}  CLAP={_fmt(r.get('clap'))}  "
              f"BPM_err={_fmt(r.get('bpm_error'))}  Chord={_fmt(r.get('chord_accuracy'))}  "
              f"RMS_corr={_fmt(r.get('rms_correlation'))}  Cent_MAE={_fmt(r.get('centroid_mae'))}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
