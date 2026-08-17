#!/usr/bin/env python3
"""
Step 1: Zero-Shot Validation — 音乐 Token 零样本响应度测试
===========================================================
基于报告《音乐生成提示词工程框架》第 12 章「行动建议」Step 1

验证目标:
  实验 A: 和弦 token 的响应度（预期：有响应）
  实验 B: dynamics / texture / timbre 纯文本描述的响应度（预期：微弱或无）

运行: python step1_zero_shot_validation.py
输出: outputs/ 目录下生成音频文件 + outputs/results.json 记录主观评分

⚠️ 由于 RTX 5060 (Blackwell sm_120) 与 PyTorch cu126 不兼容，使用 CPU 模式
   生成速度较慢（~5分钟/5秒音频），建议分次运行或减少生成次数
"""

import os
import json
import time
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = ''  # CPU 模式
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import torch
import soundfile as sf
from transformers import MusicgenForConditionalGeneration, AutoProcessor

# ===== 配置 =====
OUTPUT_DIR = "C:/Deepseek/outputs"
MODEL_NAME = "facebook/musicgen-medium"
SAMPLE_RATE = 32000
MAX_NEW_TOKENS = 256  # ≈ 5 秒音频
RESULTS_FILE = os.path.join(OUTPUT_DIR, "results.json")

# ===== 实验 A: 和弦 token 响应度 =====
# 同一风格提示词，不同和弦进行
CHORD_EXPERIMENTS = [
    {
        "id": "chord_01_major",
        "prompt": "calm piano piece, C G Am F chord progression, 80 bpm, 4/4",
        "dimension": "harmony",
        "description": "经典 I-V-vi-IV 进行（明亮、流行）"
    },
    {
        "id": "chord_02_minor",
        "prompt": "calm piano piece, Am F C G chord progression, 80 bpm, 4/4",
        "dimension": "harmony",
        "description": "vi-IV-I-V 进行（从忧郁到明亮）"
    },
    {
        "id": "chord_03_jazz",
        "prompt": "calm piano piece, Dm7 G7 Cmaj7 C6 chord progression, 80 bpm, 4/4",
        "dimension": "harmony",
        "description": "ii-V-I 爵士和声（丰富色彩）"
    },
    {
        "id": "chord_baseline",
        "prompt": "calm piano piece, 80 bpm, 4/4",
        "dimension": "harmony_baseline",
        "description": "无和弦指定的基线（对照）"
    },
]

# ===== 实验 B: 动态 / 织体 / 音色 纯文本响应度 =====
TEXTURE_DYNAMICS_EXPERIMENTS = [
    # --- dynamics (动态) ---
    {
        "id": "dyn_01_soft",
        "prompt": "piano solo, very soft volume pianissimo, gentle touch, delicate, 80 bpm",
        "dimension": "dynamics",
        "description": "极弱 (pp) — 模型是否生成轻柔音乐"
    },
    {
        "id": "dyn_02_loud",
        "prompt": "piano solo, very loud fortissimo, powerful striking, aggressive, 80 bpm",
        "dimension": "dynamics",
        "description": "极强 (ff) — 模型是否生成强力度音乐"
    },
    {
        "id": "dyn_03_crescendo",
        "prompt": "piano starting very soft, gradually building intensity and volume to a powerful climax, 80 bpm",
        "dimension": "dynamics",
        "description": "渐强 — 模型是否体现动态变化"
    },
    # --- texture (织体) ---
    {
        "id": "tex_01_sparse",
        "prompt": "sparse texture, solo single-note melody line, minimal arrangement, monophonic, 80 bpm",
        "dimension": "texture",
        "description": "稀疏织体 — 是否只用极少音符"
    },
    {
        "id": "tex_02_dense",
        "prompt": "dense orchestral texture, full symphony, many instruments playing simultaneously, polyphonic, 80 bpm",
        "dimension": "texture",
        "description": "密集织体 — 是否多层乐器同时演奏"
    },
    # --- timbre (音色) ---
    {
        "id": "tim_01_warm",
        "prompt": "warm analog synthesizer pad sound, soft filter, vintage tone, 80 bpm",
        "dimension": "timbre",
        "description": "暖音色 — 是否接近模拟合成器质感"
    },
    {
        "id": "tim_02_bright",
        "prompt": "bright digital piano sound, sharp attack, crystal clear highs, 80 bpm",
        "dimension": "timbre",
        "description": "明亮音色 — 是否接近数字钢琴质感"
    },
    # --- articulation (奏法) ---
    {
        "id": "art_01_staccato",
        "prompt": "piano staccato, short detached notes, bouncy rhythm, 120 bpm",
        "dimension": "articulation",
        "description": "断奏 — 音符是否短促分离"
    },
    {
        "id": "art_02_legato",
        "prompt": "piano legato, smoothly connected notes, flowing melody, 80 bpm",
        "dimension": "articulation",
        "description": "连奏 — 音符是否平滑连接"
    },
    # --- baseline (对照) ---
    {
        "id": "baseline_piano",
        "prompt": "piano piece, 80 bpm",
        "dimension": "baseline",
        "description": "纯基线对照（无任何额外描述）"
    },
]

ALL_EXPERIMENTS = CHORD_EXPERIMENTS + TEXTURE_DYNAMICS_EXPERIMENTS


def load_model():
    """加载 MusicGen medium 模型（CPU 模式）"""
    print(f"Loading {MODEL_NAME}...")
    model = MusicgenForConditionalGeneration.from_pretrained(MODEL_NAME)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
    return model, processor


def generate_audio(model, processor, prompt: str, output_path: str) -> float:
    """
    生成音频并保存
    返回: 生成耗时（秒）
    """
    start = time.time()
    inputs = processor(text=[prompt], padding=True, return_tensors="pt")
    with torch.no_grad():
        audio = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    elapsed = time.time() - start
    duration = audio.shape[2] / SAMPLE_RATE
    sf.write(output_path, audio[0, 0].numpy(), SAMPLE_RATE)
    return elapsed, duration


def run_experiments(model, processor, experiments, label: str, skip_existing: bool = True):
    """
    运行一组实验，返回结果列表
    """
    results = []
    total = len(experiments)
    for i, exp in enumerate(experiments):
        out_file = os.path.join(OUTPUT_DIR, f"{exp['id']}.wav")
        
        if skip_existing and os.path.exists(out_file):
            print(f"[{i+1}/{total}] SKIP (exists): {exp['id']}")
            results.append({**exp, "status": "skipped", "output": out_file})
            continue
            
        print(f"[{i+1}/{total}] Generating: {exp['id']} — {exp['description']}")
        print(f"         Prompt: {exp['prompt'][:100]}...")
        
        try:
            elapsed, duration = generate_audio(model, processor, exp['prompt'], out_file)
            print(f"         Done: {duration:.1f}s audio in {elapsed:.1f}s")
            results.append({
                **exp,
                "status": "ok",
                "output": out_file,
                "duration_s": round(duration, 1),
                "generation_time_s": round(elapsed, 1),
            })
        except Exception as e:
            print(f"         ERROR: {e}")
            results.append({**exp, "status": "error", "error": str(e)})
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Step 1: Zero-Shot Music Token Validation")
    parser.add_argument("--skip-chords", action="store_true", help="跳过和弦实验")
    parser.add_argument("--skip-texture", action="store_true", help="跳过织体/动态/音色实验")
    parser.add_argument("--force", action="store_true", help="强制重新生成所有音频")
    args = parser.parse_args()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    model, processor = load_model()
    
    all_results = []
    
    # 实验 A: 和弦响应
    if not args.skip_chords:
        print("\n" + "=" * 60)
        print("EXPERIMENT A: Chord Token Response (和弦 token 响应度)")
        print("=" * 60)
        chord_results = run_experiments(
            model, processor, CHORD_EXPERIMENTS, "chord", 
            skip_existing=not args.force
        )
        all_results.extend(chord_results)
    
    # 实验 B: 织体/动态/音色
    if not args.skip_texture:
        print("\n" + "=" * 60)
        print("EXPERIMENT B: Dynamics / Texture / Timbre (纯文本描述响应度)")
        print("=" * 60)
        tex_results = run_experiments(
            model, processor, TEXTURE_DYNAMICS_EXPERIMENTS, "texture", 
            skip_existing=not args.force
        )
        all_results.extend(tex_results)
    
    # 保存结果
    summary = {
        "model": MODEL_NAME,
        "mode": "CPU (CUDA disabled — Blackwell sm_120 incompatibility)",
        "max_new_tokens": MAX_NEW_TOKENS,
        "sample_rate": SAMPLE_RATE,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_experiments": len(all_results),
        "completed": sum(1 for r in all_results if r["status"] == "ok"),
        "skipped": sum(1 for r in all_results if r["status"] == "skipped"),
        "errors": sum(1 for r in all_results if r["status"] == "error"),
        "results": all_results,
    }
    
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {summary['completed']} done, {summary['skipped']} skipped, {summary['errors']} errors")
    print(f"Results saved to: {RESULTS_FILE}")
    print(f"Audio outputs: {OUTPUT_DIR}/")
    print(f"\nNext step: Listen to each .wav file and fill in subjective scores (0-5)")
    print(f"         Then compare groups to quantify the model capability gap.")


if __name__ == "__main__":
    main()
