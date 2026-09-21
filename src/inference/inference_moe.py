#!/usr/bin/env python3
"""inference_moe.py — GPU加速 + 自动CPU fallback"""
import sys, os, time
if sys.stdout.encoding != "utf-8": sys.stdout.reconfigure(encoding="utf-8")
import torch, torch.nn.functional as F
import soundfile as sf, numpy as np
from pathlib import Path

# ── 尝试 GPU ──
if torch.cuda.is_available():
    try:
        _ = torch.randn(1, device="cuda")  # 冒烟测试
        DEVICE = torch.device("cuda")
        USE_AMP = True
        print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB)")
    except Exception as e:
        DEVICE = torch.device("cpu"); USE_AMP = False
        print(f"GPU not available ({e}), using CPU")
else:
    DEVICE = torch.device("cpu"); USE_AMP = False
    print("CPU mode")

sys.path.insert(0, ".")
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from model.train_moe import MoEControlNet, tokenize

# ═══════════════════════════════════════════════════════════
# 1. 加载
# ═══════════════════════════════════════════════════════════
model = MoEControlNet()
ckpt = torch.load("C:/Deepseek/outputs/checkpoints/best_moe_checkpoint.pt", map_location="cpu", weights_only=True)
model.load_state_dict(ckpt, strict=False)
model = model.to(DEVICE).eval()

def build_token(style, bpm=90, key="A_minor"):
    tag = {"string":"erhu","wind":"dizi","brass":"suona"}[style]
    return f"[GLOBAL:STYLE:{tag}] [GLOBAL:EXPERT:{style}] [GLOBAL:KEY:{key}] [GLOBAL:TEMPO:{bpm}] [STEM:{tag}] [DYN:mp|ART:legato|TEX:sparse|TEMPO:{bpm}]"

# ── Router验证 ──
STYLES = [("string","demo_erhu.wav"), ("wind","demo_dizi.wav"), ("brass","demo_suona.wav")]
print("=== Router ===")
for style, _ in STYLES:
    tids = tokenize(build_token(style)).unsqueeze(0).to(DEVICE)
    sid = torch.tensor({"string":0,"wind":1,"brass":2}[style]).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        _, w, _ = model(tids, sid)
    eid = torch.argmax(w).item()
    names = ["erhu","dizi","suona"]
    print(f"  {style:8s} -> Expert {eid} ({names[eid]})  w={w[0,eid]:.3f}")

# ═══════════════════════════════════════════════════════════
# 2. MusicGen 生成
# ═══════════════════════════════════════════════════════════
print("\n=== MusicGen 生成 ===")
from transformers import MusicgenForConditionalGeneration, AutoProcessor

mg = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
mg = mg.to(DEVICE)
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
OUT = Path("C:/Deepseek/outputs/inference")
OUT.mkdir(parents=True, exist_ok=True)

for style, fname in STYLES:
    tag = {"string":"erhu","wind":"dizi","brass":"suona"}[style]
    prompt = f"A beautiful Chinese traditional melody played by {tag}, 90 bpm, minor key"
    print(f"\n  [{style}] {prompt}")

    inputs = processor(text=[prompt], padding=True, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    t0 = time.time()
    with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16, enabled=USE_AMP):
        with torch.no_grad():
            audio = mg.generate(**inputs, max_new_tokens=128)
    elapsed = time.time() - t0

    duration = audio.shape[2] / mg.config.audio_encoder.frame_rate
    sf.write(str(OUT / fname), audio[0, 0].cpu().numpy(), mg.config.audio_encoder.sampling_rate)
    print(f"    {fname} ({duration:.1f}s) in {elapsed:.1f}s")

print(f"\nDone! Files in {OUT}/")
