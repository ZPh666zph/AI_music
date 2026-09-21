# -*- coding: utf-8 -*-
"""
eval_unified_fad_clap.py — 统一计算 FAD (VGGish) 与 CLAP Score，带自动重采样。

为什么必须重采样：
    - FAD 的 VGGish 特征提取输入为 16 kHz；
    - CLAP 的音频塔输入为 48 kHz；
    - MusicGen 输出 32 kHz、AudioLDM 2 输出 16 kHz、Stable Audio Open 输出 44.1 kHz，
      若不做统一重采样，指标要么报错，要么在不同声学域上计算，导致对比不公平。

重采样方案：
    - 主路径：torchaudio.functional.resample（Kaiser 窗，标准无争议实现）
    - 兜底：numpy 线性插值（规避本机 scipy/librosa 的 DLL 冲突）

依赖（懒加载，缺什么补什么）：
    pip install soundfile torch torchaudio
    pip install frechet-audio-distance   # FAD (VGGish)
    pip install laion-clap               # CLAP Score

用法：
  # 1) 单模型
  python eval_unified_fad_clap.py \
      --gen_dirs ./results/musicgen \
      --ref_dir ./data/musiccaps_audio \
      --prompts baseline_prompts.json

  # 2) 多模型一次算完（每个目录内 wav 按文件名排序后与 prompts 一一对应）
  python eval_unified_fad_clap.py \
      --gen_dirs ./results/musicgen ./results/audioldm2 ./results/sao \
                 ./results/musicongen ./results/segtune ./results/ours \
      --ref_dir ./data/musiccaps_audio \
      --prompts baseline_prompts.json

  # 3) 用每个模型目录里的 manifest.jsonl 配对 CLAP 文本（audio + text/prompt 字段）
  python eval_unified_fad_clap.py \
      --gen_dirs ./results/musicgen ./results/ours \
      --ref_dir ./data/musiccaps_audio \
      --use_manifest

输出：
    ./results/results_summary.json —— 每个模型的 FAD / CLAP_mean / CLAP_std / n_samples
    控制台同时打印可直接粘贴进论文的表格行。
"""
import argparse
import json
import os
import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import warnings

import numpy as np
import soundfile as sf

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import torch
        import torchaudio.functional as AF
        # torch 的 numpy 桥接在部分 Windows 环境会因 DLL 冲突损坏（torch.from_numpy /
        # tensor.numpy() 直接报 "Numpy is not available"）。先探测，坏了就走 numpy 兜底。
        _probe = torch.zeros(1)
        _probe.numpy()
        torch.from_numpy(np.zeros(1, dtype=np.float32))
    HAS_TORCH = True
except Exception:
    torch = None
    AF = None
    HAS_TORCH = False

FAD_SR = 16000
# 缓存目录：由 --cache_dir 指定，避免不同评测口径共用缓存（曾导致口径污染）
CACHE_DIR = Path('results/_eval_cache')    # VGGish 输入
CLAP_SR = 48000   # LAION-CLAP 音频塔输入


# ─────────────────────────────────────────────────────────────
# 音频读取与重采样
# ─────────────────────────────────────────────────────────────
def load_mono(path: str):
    """读 wav → mono float32 (numpy) + 原始采样率。"""
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y.astype(np.float32), int(sr)


def _resample_torch(y: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    t = torch.from_numpy(y).unsqueeze(0)  # (1, T)
    out = AF.resample(t, sr, target_sr)   # Kaiser-window resampling
    return out[0].numpy().astype(np.float32)


def _resample_numpy(y: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    n = int(round(len(y) * target_sr / sr))
    x_old = np.linspace(0.0, 1.0, len(y), endpoint=False)
    x_new = np.linspace(0.0, 1.0, n, endpoint=False)
    return np.interp(x_new, x_old, y).astype(np.float32)


_FALLBACK_WARNED = False


def resample_to(y: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    global _FALLBACK_WARNED
    if sr == target_sr:
        return y
    if HAS_TORCH:
        return _resample_torch(y, sr, target_sr)
    if not _FALLBACK_WARNED:
        print("  [resample] torchaudio 不可用，使用 numpy 线性插值兜底（仅提示一次）")
        _FALLBACK_WARNED = True
    return _resample_numpy(y, sr, target_sr)


def normalize_dir(in_dir: str, out_dir: str, target_sr: int) -> str:
    """把目录内所有 wav 重采样为 target_sr 单声道，写入 out_dir。

    已存在且非空的目标文件会跳过（断点续跑：避免每次重跑 8.8G 参考集）。
    """
    os.makedirs(out_dir, exist_ok=True)
    wavs = sorted(Path(in_dir).glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"{in_dir} 中没有找到任何 .wav")
    done = 0
    for p in wavs:
        dst = Path(out_dir) / p.name
        if dst.exists() and dst.stat().st_size > 0:
            done += 1
            continue
        y, sr = load_mono(str(p))
        y = resample_to(y, sr, target_sr)
        sf.write(str(dst), y, target_sr)
    if done:
        print(f"  [resample] 跳过已缓存 {done}/{len(wavs)}（{out_dir}）")
    return out_dir



def _vggish_worker_subset(wav_paths, cache_key_part=None):
    """每个子进程：自建 CPU VGGish，处理一组 wav，返回 [N,128] 嵌入（N=len(paths)）。"""
    import sys as _sys
    import numpy as _np
    import soundfile as _sf
    import torch as _torch
    _vggish_root = _sys.path
    _hub_root = __import__('os').path.expanduser(
        r'~/.cache/torch/hub/harritaylor_torchvggish_master')
    if _hub_root not in _sys.path:
        _sys.path.insert(0, _hub_root)
    from torchvggish.vggish import VGGish
    model = VGGish(urls={
        'vggish': 'https://github.com/harritaylor/torchvggish/'
                  'releases/download/v0.1/vggish-10086976.pth',
        'pca': 'https://github.com/harritaylor/torchvggish/'
               'releases/download/v0.1/vggish_pca_params-970ea276.pth'},
        pretrained=True, preprocess=True, postprocess=False,
        progress=False, device='cpu')
    model.eval()
    embs = []
    with _torch.no_grad():
        for p in wav_paths:
            y, sr = _sf.read(str(p), dtype='float32')
            if y.ndim > 1:
                y = y.mean(axis=1)
            if sr != FAD_SR:
                n = int(round(len(y) * FAD_SR / sr))
                y = _np.interp(_np.linspace(0, 1, n, endpoint=False),
                               _np.linspace(0, 1, len(y), endpoint=False), y)
            out = model(y, fs=FAD_SR)
            out = out.reshape(out.shape[0], -1).mean(axis=0, keepdim=True)
            embs.append(out.cpu().numpy())
    return _np.concatenate(embs, axis=0)


# ─────────────────────────────────────────────────────────────
# 指标计算（重依赖全部懒加载，避免破坏 resample 逻辑）
# ─────────────────────────────────────────────────────────────
def compute_fad(gen_dir_16k: str, ref_dir_16k: str) -> float:
    # 手工 VGGish FAD：嵌入计算交给模块级 _vggish_worker_subset（多进程，见下）。
    import numpy as np

    def embed_dir_cached(d, cache_key):
        # VGGish 嵌入缓存（npz）＋ 多进程并行（i7 16 核，避免单核 numpy 拖 3-4 小时）
        cache_path = CACHE_DIR / f"{cache_key}_vggish.npz"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            return np.load(str(cache_path))["emb"]
        import multiprocessing as _mp
        from concurrent.futures import ProcessPoolExecutor as _PPE
        import os as _os2
        wavs = sorted(Path(d).glob("*.wav"))
        # FAD_WORKERS=1 -> 单进程（某些环境下子进程 torch/numpy 桥接不可用时的稳妥路径）
        n_workers = int(_os2.environ.get("FAD_WORKERS", "0")) or min(6, _mp.cpu_count())
        if n_workers == 1:
            arr = _vggish_worker_subset(wavs)
            np.savez(str(cache_path), emb=arr)
            return arr
        # 关键：必须用 spawn（fork 后子进程的 torch/numpy 桥接会崩 -> "Numpy is not available"）
        _ctx = _mp.get_context("spawn")
        chunk = 8
        parts = [wavs[i:i+chunk] for i in range(0, len(wavs), chunk)]
        arrs = []
        with _PPE(max_workers=n_workers, mp_context=_ctx) as ex:
            for arr in ex.map(_vggish_worker_subset, parts):
                arrs.append(arr)
        arr = np.concatenate(arrs, axis=0)
        np.savez(str(cache_path), emb=arr)
        return arr

    ref_key = "ref_" + Path(ref_dir_16k).name
    e1 = embed_dir_cached(gen_dir_16k, Path(gen_dir_16k).name)
    e2 = embed_dir_cached(ref_dir_16k, ref_key)
    mu1 = e1.mean(axis=0)
    s1 = np.cov(e1, rowvar=False)
    mu2 = e2.mean(axis=0)
    s2 = np.cov(e2, rowvar=False)
    diff = mu1 - mu2
    ssum = s1 + s2
    frechet = float(diff @ diff + np.trace(ssum)
                    - 2 * np.sqrt(np.maximum(np.linalg.det(ssum), 0.0) + 1e-12))
    return frechet


def compute_clap(audio_paths, texts):
    """返回 (mean, std) CLAP Score。audio_paths 必须已是 48 kHz 单声道。"""
    try:
        import laion_clap
    except ImportError:
        raise ImportError("请先安装: pip install laion-clap")

    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()
    # GPU 优先；分批计算避免 8GB 卡整批 OOM
    import torch as _torch
    if _torch.cuda.is_available():
        model.model.to("cuda")
    model.model.eval()

    # 缓存：text 嵌入 4 模型共享；audio 嵌入按目录缓存（中断可续）
    cache_dir = CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    text_cache = cache_dir / "clap_text.npz"
    if text_cache.exists():
        text_embs = np.load(str(text_cache))["emb"]
    else:
        B = 64
        parts = []
        for i in range(0, len(texts), B):
            parts.append(np.asarray(model.get_text_embedding(texts[i:i+B])))
        text_embs = np.concatenate(parts, axis=0)
        np.savez(str(text_cache), emb=text_embs)

    audio_cache = cache_dir / (Path(audio_paths[0]).parent.name + "_clap_audio.npz")
    if audio_cache.exists():
        audio_embs = np.load(str(audio_cache))["emb"]
    else:
        B = 64
        parts = []
        for i in range(0, len(audio_paths), B):
            parts.append(np.asarray(model.get_audio_embedding_from_filelist(
                x=audio_paths[i:i+B], use_tensor=False)))
        audio_embs = np.concatenate(parts, axis=0)
        np.savez(str(audio_cache), emb=audio_embs)

    # 逐条 cosine similarity
    # 长度对齐：某些模型可能少生成 1 条（如 668 vs prompts 669），取 min 并提示。
    n = min(text_embs.shape[0], audio_embs.shape[0])
    if text_embs.shape[0] != audio_embs.shape[0]:
        print(f"  [CLAP] 警告: text {text_embs.shape[0]} vs audio {audio_embs.shape[0]}，取 {n} 条配对")
    text_embs = text_embs[:n]
    audio_embs = audio_embs[:n]
    text_embs = text_embs / (np.linalg.norm(text_embs, axis=-1, keepdims=True) + 1e-8)
    audio_embs = audio_embs / (np.linalg.norm(audio_embs, axis=-1, keepdims=True) + 1e-8)
    sims = np.sum(text_embs * audio_embs, axis=-1)
    return float(sims.mean()), float(sims.std()), int(n)


# ─────────────────────────────────────────────────────────────
# CLAP 文本配对
# ─────────────────────────────────────────────────────────────
def load_prompts(path: str):
    """读 baseline_prompts.json（list）或 jsonl，返回文本列表。"""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        records = json.loads(text)
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    prompts = []
    for rec in records:
        prompts.append(rec.get("text") or rec.get("baseline_prompt") or rec.get("prompt") or "")
    prompts = [x for x in prompts if x]
    if not prompts:
        raise ValueError(f"{path} 中没有可用的 text/baseline_prompt/prompt 字段")
    return prompts


def load_manifest_pairs(gen_dir: str):
    """读 gen_dir 内的 manifest.jsonl，返回 (audio_paths, texts)。"""
    manifest = Path(gen_dir) / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"{gen_dir} 下没有 manifest.jsonl（--use_manifest 需要）")
    paths, texts = [], []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        fname = d.get("audio") or d.get("file") or d.get("path")
        text = d.get("text") or d.get("prompt") or d.get("baseline_prompt")
        if not fname or not text:
            continue
        full = Path(gen_dir) / fname
        if full.exists():
            paths.append(str(full))
            texts.append(text)
    return paths, texts


def pair_by_order(gen_dir_48k: str, prompts):
    """无 manifest 时：wav 按文件名排序，与 prompts 按序一一配对。"""
    wavs = sorted(Path(gen_dir_48k).glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"{gen_dir_48k} 中没有 .wav")
    if len(wavs) != len(prompts):
        print(f"  [pair] 警告: {gen_dir_48k} 有 {len(wavs)} 个 wav，"
              f"但 prompts 有 {len(prompts)} 条，取 min 配对（请检查是否漏生成）")
    n = min(len(wavs), len(prompts))
    return [str(w) for w in wavs[:n]], prompts[:n]


# ─────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────
def main():
    global CACHE_DIR
    ap = argparse.ArgumentParser(description="统一 FAD + CLAP 评测（自动重采样）")
    ap.add_argument("--gen_dirs", nargs="+", required=True,
                    help="一个或多个生成音频目录（./results/musicgen ./results/ours ...）")
    ap.add_argument("--ref_dir", required=True, help="FAD 参考集目录（真实音乐，如 MusicCaps 子集）")
    ap.add_argument("--prompts", default="baseline_prompts.json",
                    help="baseline_prompts.json（无 --use_manifest 时按序配对 CLAP 文本）")
    ap.add_argument("--use_manifest", action="store_true",
                    help="使用每个 gen_dir 内的 manifest.jsonl 配对 CLAP 文本")
    ap.add_argument("--cache_dir", default="./results/_eval_cache",
                    help="重采样缓存目录（16k/48k 副本）")
    ap.add_argument("--out", default="./results/results_summary.json",
                    help="结果 JSON 输出路径")
    ap.add_argument("--fad_only", action="store_true",
                    help="只算 FAD，跳过 CLAP（laion_clap 在本机过慢时用）")
    args = ap.parse_args()
    CACHE_DIR = Path(args.cache_dir)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[eval] cache_dir = {CACHE_DIR}", flush=True)

    # 参考集只重采样一次（FAD 用 16k）
    ref16 = os.path.join(args.cache_dir, "ref_16k")
    print(f"重采样参考集 -> 16 kHz: {args.ref_dir}")
    normalize_dir(args.ref_dir, ref16, FAD_SR)

    prompts = None
    if not args.use_manifest:
        prompts = load_prompts(args.prompts)
        print(f"读取 CLAP prompts: {len(prompts)} 条（{args.prompts}）")

    results = {}
    for gen_dir in args.gen_dirs:
        name = Path(gen_dir).name
        print(f"\n===== 评测 {gen_dir} =====")

        gen16 = normalize_dir(gen_dir, os.path.join(args.cache_dir, name + "_16k"), FAD_SR)
        gen48 = normalize_dir(gen_dir, os.path.join(args.cache_dir, name + "_48k"), CLAP_SR)

        fad = compute_fad(gen16, ref16)
        print(f"  FAD (VGGish)   = {fad:.4f}")

        if args.use_manifest:
            audio_paths, texts = load_manifest_pairs(gen_dir)
            # 对 manifest 里指定的文件单独做 48k 对齐（复用缓存目录）
            audio_paths_48k = []
            for p in audio_paths:
                y, sr = load_mono(p)
                y = resample_to(y, sr, CLAP_SR)
                dst = os.path.join(gen48, Path(p).name)
                sf.write(dst, y, CLAP_SR)
                audio_paths_48k.append(dst)
        else:
            audio_paths_48k, texts = pair_by_order(gen48, prompts)

        clap_mean = clap_std = None
        if not args.fad_only:
            clap_mean, clap_std, clap_n = compute_clap(audio_paths_48k, texts)
            print(f"  CLAP Score     = {clap_mean:.4f} ± {clap_std:.4f}  (n={clap_n})")

        results[name] = {
            "dir": gen_dir,
            "FAD": fad,
            "CLAP_mean": clap_mean,
            "CLAP_std": clap_std,
            "n_samples": len(texts),
            "n_gen": len(audio_paths_48k),
        }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n===== 汇总（论文表格行） =====")
    print(f"{'Model':20s} {'FAD↓':>8s} {'CLAP↑':>8s} {'±std':>6s} {'n':>4s}")
    for name, r in results.items():
        print(f"{name:20s} {r['FAD']:8.4f} {r['CLAP_mean']:8.4f} {r['CLAP_std']:6.4f} {r['n_samples']:4d}")
    print(f"\n结果已写入: {args.out}")


if __name__ == "__main__":
    main()
