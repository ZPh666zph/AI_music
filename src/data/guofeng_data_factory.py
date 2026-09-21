#!/usr/bin/env python3
"""
guofeng_data_factory_v3.py — MoE 多乐器并行产线 (v3.0)
==========================================================
每首 MIDI → 3 路并行渲染: 二胡(erhu) · 笛子(dizi) · 唢呐(suona)
全部强制 Top-Note 单音化, 自动生成 MoE 训练标签
"""

import sys, os, re, json, time, shutil, subprocess, argparse, random
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import pretty_midi, numpy as np
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════
# 路径（均可用环境变量覆盖，便于在 Linux/服务器上运行）
#   GF_MIDI_DIR / GF_OUT_DIR / GF_TEMP_DIR / GF_FLUIDSYNTH / GF_SOUNDFONT_DIR
# ═══════════════════════════════════════════════════════════
import os as _os
MIDI_DIR    = Path(_os.environ.get("GF_MIDI_DIR", "C:/Deepseek/data/raw_midis"))
OUT_DIR     = Path(_os.environ.get("GF_OUT_DIR", "C:/Deepseek/outputs/guofeng_v3"))
TEMP_DIR    = Path(_os.environ.get("GF_TEMP_DIR", "C:/Deepseek/data/guofeng_temp"))
FLUIDSYNTH  = Path(_os.environ.get("GF_FLUIDSYNTH", "fluidsynth"))  # Linux 下即 PATH 中的 fluidsynth
SF_DIR      = Path(_os.environ.get("GF_SOUNDFONT_DIR", "C:/Deepseek/soundfonts"))
SR          = 16000

# 三路产线配置
PIPELINES = [
    {
        "name": "erhu",
        "style_expert": "string",
        "sf2": SF_DIR / "guzheng.sf2",  # 二胡
        "gm_replacement_program": 40,   # Violin → Erhu
    },
    {
        "name": "dizi",
        "style_expert": "wind",
        "sf2": SF_DIR / "Dizi.sf2",     # 笛子
        "gm_replacement_program": 73,   # Flute → Dizi
    },
    {
        "name": "suona",
        "style_expert": "brass",
        "sf2": SF_DIR / "BrightTrombone.sf2",  # 长号平替唢呐
        "gm_replacement_program": 56,   # Trumpet → Suona
    },
]

OUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)
random.seed(42)


# ═══════════════════════════════════════════════════════════
# Top-Note 单音提取 (二胡/笛子/唢呐 全部强制)
# ═══════════════════════════════════════════════════════════

def make_monophonic(pm: pretty_midi.PrettyMIDI) -> tuple:
    """重叠音符只保留最高音, 返回 (过滤后音符数, 原始音符数)"""
    original = sum(len(i.notes) for i in pm.instruments if not i.is_drum)
    for inst in pm.instruments:
        if inst.is_drum: continue
        notes = sorted(inst.notes, key=lambda n: n.start)
        to_del, active = [], []
        for note in notes:
            active = [n for n in active if n.end > note.start]
            if active:
                all_n = active + [note]
                top = max(all_n, key=lambda n: n.pitch)
                for n in all_n:
                    if n is not top: to_del.append(n)
            active.append(note)
        for n in set(to_del):
            if n in inst.notes: inst.notes.remove(n)
    filtered = sum(len(i.notes) for i in pm.instruments if not i.is_drum)
    return filtered, original


# ═══════════════════════════════════════════════════════════
# 单路渲染
# ═══════════════════════════════════════════════════════════

def extract_meta(midi_path: str) -> dict:
    pm = pretty_midi.PrettyMIDI(midi_path)
    t = pm.get_tempo_changes()
    bpm = float(np.mean(t[1])) if len(t[0]) > 0 else 120
    ts = pm.time_signature_changes
    meter = f"{ts[0].numerator}/{ts[0].denominator}" if ts else "4/4"
    dur = pm.get_end_time()
    return {"bpm": round(bpm, 1), "meter": meter, "duration": round(dur, 1)}


def render_stem(midi_path: str, output_wav: str, sf2_path: str) -> bool:
    cmd = [str(FLUIDSYNTH), "-ni", "-r", str(SR), "-g", "2.0",
           "-F", output_wav, sf2_path, midi_path]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="ignore", timeout=300)
        return r.returncode == 0 and os.path.exists(output_wav)
    except:
        return False


def render_one_pipeline(midi_path: Path, out_parent: Path, pipe: dict) -> dict:
    """对单首 MIDI 执行一路产线渲染"""
    track_name = midi_path.stem
    inst_name = pipe["name"]
    out_dir = out_parent / track_name / inst_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 消音版 MIDI: Top-Note + 转英文名
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    filtered, original = make_monophonic(pm)
    temp_midi = TEMP_DIR / f"temp_{inst_name}.mid"
    pm.write(str(temp_midi))

    # 2. 渲染
    temp_wav = TEMP_DIR / f"temp_{inst_name}.wav"
    sf2 = str(pipe["sf2"])
    success = render_stem(str(temp_midi), str(temp_wav), sf2)

    mix_wav = out_dir / "mix.wav"
    if success and temp_wav.exists():
        shutil.copy2(str(temp_wav), str(mix_wav))

    temp_midi.unlink(missing_ok=True)
    temp_wav.unlink(missing_ok=True)

    # 3. 元数据
    midi_meta = extract_meta(str(midi_path))
    purity = filtered / max(original, 1)

    meta = {
        "track_name": track_name,
        "midi_file": str(midi_path.resolve()),
        "instrument": inst_name,
        "style_expert": pipe["style_expert"],       # ← MoE Ground Truth
        "sf2_used": str(pipe["sf2"].name),
        "bpm": midi_meta["bpm"],
        "meter": midi_meta["meter"],
        "duration_sec": midi_meta["duration"],
        "erhu_purity" if inst_name == "erhu" else "mono_purity": round(purity, 3),
        "original_notes": original,
        "filtered_notes": filtered,
        "render_success": success,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta


# ═══════════════════════════════════════════════════════════
# 主循环: 每首 MIDI → 3 路并行
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="guofeng_data_factory v3.0")
    parser.add_argument("--demo", action="store_true", help="只处理第1首")
    parser.add_argument("--pipes", type=str, default="erhu,dizi,suona",
                        help="逗号分隔的产线名 (如 erhu,dizi)")
    args = parser.parse_args()

    active_pipes = [p for p in PIPELINES if p["name"] in args.pipes.split(",")]

    print("=" * 60)
    print("guofeng_data_factory v3.0 — MoE 多乐器并行")
    print("=" * 60)
    print(f"\n  MIDI 输入:  {MIDI_DIR}")
    print(f"  输出:       {OUT_DIR}")
    print(f"  激活产线:   {', '.join(p['name'] for p in active_pipes)} ({len(active_pipes)} 路)")
    for p in active_pipes:
        sf2_ok = p["sf2"].exists()
        print(f"    {p['name']:8s} → {p['sf2'].name:25s}  {'存在' if sf2_ok else '缺失!'}")
    print(f"\n  预期产出:   {len(list(MIDI_DIR.glob('*.mid'))) * len(active_pipes)} 个 WAV\n")

    midi_files = sorted(MIDI_DIR.glob("*.mid")) + sorted(MIDI_DIR.glob("*.midi"))
    if args.demo:
        midi_files = midi_files[:1]

    results = {"ok": 0, "fail": 0, "total_wavs": len(midi_files) * len(active_pipes)}
    pbar = tqdm(midi_files, desc=f"x{len(active_pipes)}路渲染", unit="首")
    for f in pbar:
        for pipe in active_pipes:
            pbar.set_postfix_str(f"{f.stem[:15]} → {pipe['name']}")
            try:
                meta = render_one_pipeline(f, OUT_DIR, pipe)
                if meta["render_success"]: results["ok"] += 1
                else: results["fail"] += 1
            except Exception as e:
                results["fail"] += 1
                tqdm.write(f"  ✗ {f.stem}·{pipe['name']}: {e}")

    print(f"\n{'='*60}")
    print(f"完成: OK {results['ok']} | FAIL {results['fail']} | 总WAV {results['total_wavs']}")
    print(f"输出: {OUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
