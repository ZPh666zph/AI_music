#!/usr/bin/env python3
"""build_moe_dataset.py — MoE 三专家训练集打包 (v3 产出 → JSONL)"""
import sys, json, random
from pathlib import Path
from tqdm import tqdm
if sys.stdout.encoding != "utf-8": sys.stdout.reconfigure(encoding="utf-8")
import pretty_midi, numpy as np
random.seed(42)

DATA_DIR = Path("C:/Deepseek/outputs/guofeng_v3")
OUTPUT   = Path("C:/Deepseek/data/gufeng_moe_train.jsonl")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

# ── Text 模板 (按 style_expert 分类) ──
TEMPLATES = {
    "string": [
        "A traditional Chinese gufeng melody played by erhu, acting as a string expert, {bpm} bpm.",
        "Erhu solo piece in gufeng style, string instrument expert, {bpm} bpm, {meter} meter.",
        "Chinese erhu performance of gufeng music, string specialist, tempo {bpm}.",
    ],
    "wind": [
        "A traditional Chinese gufeng melody played by dizi, acting as a wind expert, {bpm} bpm.",
        "Dizi solo piece in gufeng style, wind instrument expert, {bpm} bpm, {meter} meter.",
        "Chinese dizi (bamboo flute) performance of gufeng music, wind specialist, tempo {bpm}.",
    ],
    "brass": [
        "A traditional Chinese gufeng melody played by suona, acting as a brass expert, {bpm} bpm.",
        "Suona solo piece in gufeng style, brass instrument expert, {bpm} bpm, {meter} meter.",
        "Chinese suona performance of gufeng music, brass specialist, tempo {bpm}.",
    ],
}

# ── K-S 调性估计 ──
def estimate_key(midi_path: str) -> str:
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
        pitches = []
        for inst in pm.instruments:
            if not inst.is_drum:
                for note in inst.notes:
                    pitches.append(note.pitch % 12)
                    if len(pitches) > 2000: break
        if not pitches: return "C_major"
        counts = np.bincount(pitches, minlength=12).astype(float)
        counts = counts / counts.sum()
        major = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
        minor = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
        names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
        best_key, best_score = "C_major", -1
        for i in range(12):
            r = np.roll(counts, -i)
            if (s:=np.corrcoef(r,major)[0,1]) > best_score: best_score=s; best_key=f"{names[i]}_major"
            if (s:=np.corrcoef(r,minor)[0,1]) > best_score: best_score=s; best_key=f"{names[i]}_minor"
        return best_key
    except: return "C_major"


# ── Token 生成 ──
def generate_tokens(style: str, bpm: float, key: str, meter: str, purity: float) -> str:
    style_tag = {"string":"erhu","wind":"dizi","brass":"suona"}.get(style, style)
    lines = [
        f"[GLOBAL:STYLE:{style_tag}]",
        f"[GLOBAL:EXPERT:{style}]",
        f"[GLOBAL:TEMPO:{round(bpm)}]",
        f"[GLOBAL:KEY:{key}]",
        f"[GLOBAL:TIME:{meter}]",
        "",
    ]
    sections = [
        ("intro",  "p",   round(bpm)-5),
        ("verse1", "mp",  round(bpm)),
        ("chorus1","f",   round(bpm)+8),
        ("outro",  "p",   round(bpm)-10),
    ]
    bar = 1
    for sec, dyn, sec_bpm in sections:
        art = "legato" if purity > 0.3 else "marcato"
        tex = "sparse" if purity > 0.5 else "medium"
        lines.append(f"[SEC:{sec}|BAR:{bar}-{bar+7}]")
        lines.append(f"  [STEM:{style_tag}]")
        lines.append(f"    [DYN:{dyn}|ART:{art}|TEX:{tex}|TEMPO:{sec_bpm}]")
        lines.append(f"  [/STEM]")
        lines.append("")
        bar += 8
    return "\n".join(lines)


# ── 遍历 ──
rows = 0
with open(OUTPUT, "w", encoding="utf-8") as f:
    track_dirs = sorted(d for d in DATA_DIR.iterdir() if d.is_dir())
    pbar = tqdm(track_dirs, desc="打包 MoE", unit="首")
    for td in pbar:
        track_name = td.name
        for inst_dir in sorted(td.iterdir()):
            if not inst_dir.is_dir(): continue
            mf = inst_dir / "meta.json"
            wf = inst_dir / "mix.wav"
            if not mf.exists() or not wf.exists(): continue
            meta = json.loads(mf.read_text(encoding="utf-8"))
            if not meta.get("render_success"): continue

            style = meta.get("style_expert", "string")
            bpm   = meta.get("bpm", 120)
            meter = meta.get("meter", "4/4")
            purity= meta.get("erhu_purity", meta.get("mono_purity", 0.5))

            # 调性从 MIDI 提取
            midi_path = meta.get("midi_file", "")
            key = estimate_key(midi_path) if midi_path and Path(midi_path).exists() else "C_major"

            # Text
            tmpl = random.choice(TEMPLATES.get(style, TEMPLATES["string"]))
            text = tmpl.format(bpm=round(bpm), meter=meter)

            # Tokens
            tokens = generate_tokens(style, bpm, key, meter, purity)

            row = {
                "track_name": track_name,
                "style_expert": style,
                "text": text,
                "tokens": tokens,
                "audio_path": str(wf.resolve()),
                "meta": {"bpm": round(bpm), "key": key, "meter": meter, "purity": round(purity,3)},
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows += 1
            pbar.set_postfix_str(f"{style} {rows}")

print(f"\n{'='*60}")
print(f"MoE 训练集: {rows} 组")
print(f"输出: {OUTPUT}")
with open(OUTPUT, encoding="utf-8") as f:
    sample = json.loads(f.readline())
print(f"\n样本:")
print(f"  text: {sample['text']}")
print(f"  style_expert: {sample['style_expert']}")
print(f"  tokens (前200字): {sample['tokens'][:200]}")
print(f"{'='*60}")
