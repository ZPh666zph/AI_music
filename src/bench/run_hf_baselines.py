# -*- coding: utf-8 -*-
"""
run_hf_baselines.py — 依次运行 3 个 HuggingFace Baseline，音频保存到 ./results/<model>/。

模型与统一时长：
    MusicGen          facebook/musicgen-medium         transformers  | 500 tokens = 10 s @ 50 Hz
    AudioLDM 2        cvssp/audioldm2                  diffusers     | audio_length_in_s = 10.0
    Stable Audio Open stabilityai/stable-audio-open-1.0 diffusers    | audio_end_in_s   = 10.0

工程保证：
    - 3 个模型统一生成 10 秒（可用 --seconds 覆盖），写盘前强制 trim/pad 到精确时长；
    - 输出文件名与测试集 ID 严格对齐：001.wav, 002.wav, ...（ID 取自 baseline_prompts.json
      每条记录的 "id" 字段，缺省时用行号，从 0 开始按 %03d 补零）；
    - 顺序加载模型：MusicGen 全部跑完 -> del + empty_cache -> AudioLDM 2 -> ... 防止 OOM；
    - 单条生成失败只打印 error 并跳过，写入 <out_dir>/failed.jsonl，不会中断整个脚本；
    - tqdm 进度条；tqdm 缺失时自动降级为普通循环。

用法：
    python run_hf_baselines.py                          # 依次跑全部 3 个模型
    python run_hf_baselines.py --models musicgen        # 只跑 MusicGen
    python run_hf_baselines.py --models audioldm2 sao   # 只跑后两个
    python run_hf_baselines.py --seconds 30             # 统一 30 s（注意显存）
    python run_hf_baselines.py --prompts baseline_prompts.json --out_root ./results
"""
import argparse
import gc
import json
import os
import sys
import warnings
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import soundfile as sf
import torch

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable


# ─────────────────────────────────────────────────────────────
# 环境自检
# ─────────────────────────────────────────────────────────────
def _torch_numpy_bridge_ok() -> bool:
    """torch 的 numpy 桥接在部分 Windows 环境会因 DLL 冲突损坏。"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = torch.zeros(1)
            t.numpy()
            torch.from_numpy(np.zeros(1, dtype=np.float32))
        return True
    except Exception:
        return False


TORCH_NUMPY_OK = _torch_numpy_bridge_ok()


def _print_env_warning():
    print(f"  torch {torch.__version__}  numpy {np.__version__}  CUDA {torch.cuda.is_available()}")
    if not TORCH_NUMPY_OK:
        print("  [警告] 检测到 torch<->numpy 桥接损坏（常见于 Windows numpy/torch DLL 冲突）。")
        print("         MusicGen 输出将用 tolist() 兜底写 WAV；但 diffusers 管线")
        print("         （AudioLDM 2 / Stable Audio Open）内部输出转换可能失败。")
        print("         若后两个模型全部报错，请在干净 conda 环境中运行本脚本。")
    if not torch.cuda.is_available():
        print("  [警告] 未检测到 CUDA，diffusion 模型在 CPU 上会非常慢，建议 GPU 运行。")


# ─────────────────────────────────────────────────────────────
# Prompt 读取与文件名对齐
# ─────────────────────────────────────────────────────────────
def load_prompts(path: str):
    """读 baseline_prompts.json（list）或 .jsonl，返回 [(idx, prompt), ...]。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"找不到 prompt 文件: {path}")
    text = p.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        records = json.loads(text)
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]

    pairs = []
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            continue
        prompt = (rec.get("baseline_prompt") or rec.get("text")
                  or rec.get("prompt") or "").strip()
        if not prompt:
            print(f"  [警告] 第 {i} 条记录没有 baseline_prompt/text/prompt，跳过")
            continue
        idx = rec.get("id", i)
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            idx = i
        pairs.append((idx, prompt))
    if not pairs:
        raise ValueError(f"{path} 中没有可用 prompt")
    return pairs


def fmt_filename(idx: int) -> str:
    """测试集 ID -> 零填充文件名（0 -> 001.wav）。"""
    return f"{max(int(idx), 0):03d}.wav"


# ─────────────────────────────────────────────────────────────
# 音频统一化：to numpy / mono / 精确时长
# ─────────────────────────────────────────────────────────────
def _to_numpy(audio) -> np.ndarray:
    if isinstance(audio, torch.Tensor):
        a = audio.detach().cpu().float()
        while a.ndim > 1:
            a = a[0]                       # (B, C, T) -> (C, T) -> (T,)
        if TORCH_NUMPY_OK:
            return a.numpy().astype(np.float32)
        return np.asarray(a.tolist(), dtype=np.float32)
    a = np.asarray(audio, dtype=np.float32)
    while a.ndim > 1 and a.shape[0] in (1, 2):
        # (1, T) 去掉通道维；(2, T) 双声道取均值 -> mono
        a = a.mean(axis=0) if a.shape[0] == 2 else a[0]
    return a


def _force_duration(a: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    target = int(round(sr * seconds))
    if len(a) > target:
        return a[:target]
    if len(a) < target:
        return np.pad(a, (0, target - len(a)))
    return a


def save_audio(audio, sr: int, out_path: Path, seconds: float):
    a = _to_numpy(audio)
    a = _force_duration(a, int(sr), float(seconds))
    sf.write(str(out_path), a, int(sr))


# ─────────────────────────────────────────────────────────────
# 三个模型的 load / generate
# ─────────────────────────────────────────────────────────────
def load_musicgen(device: str):
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    # 优先用本地缓存/本地目录，避免依赖网络下载
    model_id = "models/musicgen-medium" if Path("models/musicgen-medium").exists() else "facebook/musicgen-medium"
    model = MusicgenForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype, local_files_only=False).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def gen_musicgen(loaded, prompt: str, seconds: float, seed: int):
    model, processor = loaded
    torch.manual_seed(seed)
    inputs = processor(text=[prompt], padding=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        audio = model.generate(
            **inputs,
            max_new_tokens=int(round(seconds * 50)),   # 50 Hz frame rate
            do_sample=True,
            guidance_scale=3.0,
        )
    sr = int(model.config.audio_encoder.sampling_rate)  # 32000
    return audio, sr


def load_audioldm2(device: str):
    from diffusers import AudioLDM2Pipeline
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model_id = "models/audioldm2" if Path("models/audioldm2").exists() else "cvssp/audioldm2"
    pipe = AudioLDM2Pipeline.from_pretrained(model_id, torch_dtype=dtype)
    # 8GB 显存不够全量上卡，用 CPU offload（顺序推理，offload 切换开销可接受）
    pipe.enable_model_cpu_offload()
    return pipe


def gen_audioldm2(pipe, prompt: str, seconds: float, seed: int, steps: int = 200):
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    out = pipe(
        prompt,
        num_inference_steps=steps,
        audio_length_in_s=float(seconds),
        num_waveforms_per_prompt=1,
        generator=generator,
    )
    return out.audios[0], 16000


def load_sao(device: str):
    from diffusers import StableAudioPipeline
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model_id = "models/sao" if Path("models/sao").exists() else "stabilityai/stable-audio-open-1.0"
    pipe = StableAudioPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.enable_model_cpu_offload()
    return pipe


def gen_sao(pipe, prompt: str, seconds: float, seed: int, steps: int = 100):
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    out = pipe(
        prompt,
        negative_prompt="low quality, muffled",
        num_inference_steps=steps,
        audio_end_in_s=float(seconds),
        num_waveforms_per_prompt=1,
        generator=generator,
    )
    return out.audios[0], 44100


MODELS = {
    "musicgen": {
        "name": "MusicGen",
        "dir": "musicgen",
        "load": load_musicgen,
        "gen": gen_musicgen,
        "kwargs": {},
    },
    "audioldm2": {
        "name": "AudioLDM 2",
        "dir": "audioldm2",
        "load": load_audioldm2,
        "gen": gen_audioldm2,
        "kwargs": {"steps": 200},
    },
    "sao": {
        "name": "Stable Audio Open",
        "dir": "sao",
        "load": load_sao,
        "gen": gen_sao,
        "kwargs": {"steps": 100},
    },
}


# ─────────────────────────────────────────────────────────────
# 单个模型的完整流程
# ─────────────────────────────────────────────────────────────
def run_model(key: str, cfg: dict, pairs, out_root: Path, seconds: float,
              seed: int, device: str):
    name = cfg["name"]
    out_dir = out_root / cfg["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}\n加载 {name} ({key}) ...")
    loaded = None
    try:
        loaded = cfg["load"](device)
    except Exception as e:
        print(f"[fatal] {name} 加载失败，跳过该模型: {e}")
        return

    failed = []
    ok = 0
    try:
        pbar = tqdm(pairs, desc=name, unit="clip", ncols=100)
        for idx, prompt in pbar:
            fname = fmt_filename(idx)
            out_path = out_dir / fname
            # 已存在则跳过（断点续跑，避免重复生成）
            if out_path.exists():
                ok += 1
                pbar.set_postfix_str(f"skip {fname}")
                continue
            try:
                seed_i = int(seed) + int(idx)
                audio, sr = cfg["gen"](loaded, prompt, seconds, seed_i, **cfg["kwargs"])
                save_audio(audio, sr, out_path, seconds)
                ok += 1
                pbar.set_postfix_str(fname)
            except Exception as e:
                failed.append({"id": idx, "error": str(e)})
                pbar.set_postfix_str(f"ERR id={idx}")
                print(f"\n[error] {name} id={idx} -> {fname}: {e}")
    finally:
        del loaded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{name} 完成: 成功 {ok}，失败 {len(failed)}，输出目录 {out_dir}")
    if failed:
        fail_path = out_dir / "failed.jsonl"
        with open(fail_path, "w", encoding="utf-8") as f:
            for item in failed:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"失败清单已写入: {fail_path}")


def main():
    ap = argparse.ArgumentParser(description="HF 三剑客 Baseline 顺序推理")
    ap.add_argument("--prompts", default="baseline_prompts.json")
    ap.add_argument("--out_root", default="./results")
    ap.add_argument("--models", nargs="+",
                    choices=list(MODELS.keys()),
                    default=list(MODELS.keys()),
                    help="要运行的模型，默认全部: musicgen audioldm2 sao")
    ap.add_argument("--seconds", type=float, default=10.0,
                    help="统一生成时长（秒），默认 10")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("===== HF Baseline 顺序推理 =====")
    _print_env_warning()
    print(f"统一时长: {args.seconds}s | 设备: {args.device} | seed: {args.seed}")
    print(f"输出根目录: {args.out_root}")

    pairs = load_prompts(args.prompts)
    print(f"读取 prompts: {len(pairs)} 条（{args.prompts}）")

    for key in args.models:
        run_model(key, MODELS[key], pairs, Path(args.out_root),
                  args.seconds, args.seed, args.device)

    print("\n全部完成。")


if __name__ == "__main__":
    main()
