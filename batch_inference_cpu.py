#!/usr/bin/env python3
"""batch_inference_cpu.py — CPU 极限马拉松 · 10s 音频 · 30 首 · 断点续推"""
import sys, os, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
if sys.stdout.encoding != "utf-8": sys.stdout.reconfigure(encoding="utf-8")

import torch, soundfile as sf, numpy as np
from pathlib import Path
from tqdm import tqdm

from transformers import MusicgenForConditionalGeneration, AutoProcessor

# ═══════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════
OUT_DIR   = Path("C:/Deepseek/outputs/eval_audio")
PROGRESS  = OUT_DIR / "_progress.json"
MAX_TOKENS = 800  # ≈10s at 50Hz EnCodec frame rate
SR         = 32000

OUT_DIR.mkdir(parents=True, exist_ok=True)

# 10 prompts × 3 styles = 30 tracks
PROMPTS = [
    "A peaceful Chinese traditional melody, flowing like water",
    "A bright festive Chinese folk tune with energetic rhythm",
    "A melancholic gufeng piece, slow and introspective",
    "An elegant Chinese court music, dignified and ceremonial",
    "A lively Chinese harvest celebration dance tune",
    "A solemn Chinese temple music, deep and meditative",
    "A romantic Chinese love ballad, tender and sweet",
    "A heroic Chinese battle march, powerful and determined",
    "A dreamy Chinese landscape painting in music, ethereal",
    "A nostalgic Chinese lullaby, gentle and warm",
]
STYLES = [
    ("string", "erhu"),
    ("wind",   "dizi"),
    ("brass",  "suona"),
]

# ═══════════════════════════════════════════════════════════
# 断点续推
# ═══════════════════════════════════════════════════════════
def load_progress():
    if PROGRESS.exists():
        return set(json.loads(PROGRESS.read_text(encoding="utf-8")))
    return set()

def save_progress(done):
    PROGRESS.write_text(json.dumps(list(done)), encoding="utf-8")

# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
print("="*60)
print("CPU 极限马拉松 — 10s 长音频批量推理")
print(f"  Prompts: {len(PROMPTS)}  Styles: {len(STYLES)}  Total: {len(PROMPTS)*len(STYLES)} tracks")
print(f"  Output:   {OUT_DIR}")
print(f"  Duration: ~10s/track  Tokens: {MAX_TOKENS}")
print("="*60)

# 加载模型 (一次)
print("\nLoading musicgen-medium...")
mg = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-medium")
processor = AutoProcessor.from_pretrained("facebook/musicgen-medium")
print("Model loaded.")

done = load_progress()
todo = [(p, s, t) for p in PROMPTS for s, t in STYLES if f"{p}_{t}" not in done]
print(f"  已完成: {len(done)}  待生成: {len(todo)}")

if not todo:
    print("All done!")
    sys.exit(0)

pbar = tqdm(todo, desc="CPU 生成", unit="track")
for prompt, style, tag in pbar:
    track_id = f"{prompt[:30]}_{tag}"
    out_wav = OUT_DIR / f"{track_id}.wav"
    t0 = time.time()

    try:
        styled_prompt = f"{prompt}, played by {tag}, traditional Chinese instrument"
        inputs = processor(text=[styled_prompt], padding=True, return_tensors="pt")
        with torch.no_grad():
            audio = mg.generate(**inputs, max_new_tokens=MAX_TOKENS)
        duration = audio.shape[2] / SR
        sf.write(str(out_wav), audio[0, 0].numpy(), SR)

        elapsed = time.time() - t0
        pbar.set_postfix_str(f"{tag:6s} {duration:.0f}s in {elapsed:.0f}s")
        done.add(track_id)
        save_progress(done)
    except Exception as e:
        pbar.write(f"  ✗ {track_id}: {e}")

pbar.close()
print(f"\nDone! {len(done)} tracks in {OUT_DIR}/")
