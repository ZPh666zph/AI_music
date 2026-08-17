#!/usr/bin/env python3
"""gen_modern_styles.py — 快速生成 50 首 Electronic + 50 首 Pop Piano 训练数据"""
import sys, json, random
sys.path.insert(0, ".")
from pathlib import Path
random.seed(42)

OUT = Path("C:/Deepseek/data/modern_styles_train.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

# ── 风格模板 ──
TEMPLATES = {
    "electronic": [
        "An energetic electronic synth melody, {bpm} bpm, futuristic sound design",
        "A deep electronic ambient piece, {bpm} bpm, layered synthesizers",
        "A driving electronic dance beat, {bpm} bpm, pulsating bass",
    ],
    "piano": [
        "A beautiful pop piano ballad, {bpm} bpm, emotional and melodic",
        "An upbeat pop piano riff, {bpm} bpm, bright and catchy",
        "A gentle pop piano accompaniment, {bpm} bpm, warm and soft",
    ],
}

TOKEN_TEMPLATE = """[GLOBAL:STYLE:{style}]
[GLOBAL:EXPERT:{expert}]
[GLOBAL:TEMPO:{bpm}]
[GLOBAL:KEY:{key}]
[GLOBAL:TIME:4/4]

[SEC:intro|BAR:1-8]
  [STEM:{style}]
    [DYN:mp|ART:legato|TEX:medium|TEMPO:{bpm}]
  [/STEM]

[SEC:verse1|BAR:9-16]
  [STEM:{style}]
    [DYN:mp|ART:legato|TEX:medium|TEMPO:{bpm}]
  [/STEM]

[SEC:chorus1|BAR:17-24]
  [STEM:{style}]
    [DYN:f|ART:marcato|TEX:dense|TEMPO:{bpm_high}]
  [/STEM]

[SEC:outro|BAR:25-32]
  [STEM:{style}]
    [DYN:p|ART:legato|TEX:sparse|TEMPO:{bpm_low}]
  [/STEM]"""

keys = ["C_major","G_major","D_minor","A_minor","E_minor","F_major","A#_major"]

rows = 0
with open(OUT, "w", encoding="utf-8") as f:
    for style, expert_name in [("electronic","electronic"),("piano","piano")]:
        for i in range(50):
            bpm = random.randint(100, 140)
            tmpl = random.choice(TEMPLATES[style])
            text = tmpl.format(bpm=bpm)
            key = random.choice(keys)
            token = TOKEN_TEMPLATE.format(
                style=style, expert=expert_name,
                bpm=bpm, bpm_high=bpm+10, bpm_low=bpm-10, key=key
            )
            # 使用现有二胡 WAV 作为音频参考 (快速测试用)
            row = {
                "track_name": f"{style}_{i:03d}",
                "style_expert": expert_name,
                "text": text,
                "tokens": token,
                "audio_path": "C:/Deepseek/outputs/guofeng_v3/赤伶/erhu/mix.wav",
                "meta": {"bpm": bpm, "key": key, "meter": "4/4", "purity": 0.5},
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows += 1

print(f"Generated {rows} rows → {OUT}")
