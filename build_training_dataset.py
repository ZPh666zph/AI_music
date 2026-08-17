#!/usr/bin/env python3
"""
build_training_dataset.py — 最终打包: meta.json → Token → Text → JSONL
======================================================================
扫描 223 个二胡渲染目录, 生成六维 Token + 英文 Text Prompt, 输出训练文件
"""

import sys, os, json, re, random
from pathlib import Path
from collections import Counter

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

try:
    import pretty_midi
    import numpy as np
    from tqdm import tqdm
except ImportError:
    print("pip install pretty_midi numpy tqdm")
    sys.exit(1)

# ── 路径 ──
DATA_DIR  = Path("C:/Deepseek/outputs/guofeng_data")
MIDI_DIR  = Path("C:/Deepseek/data/raw_midis")
OUTPUT    = Path("C:/Deepseek/data/gufeng_train_dataset.jsonl")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

random.seed(42)


# ═══════════════════════════════════════════════════════════
# 1. MIDI 元数据快速提取 (调性 + 拍号)
# ═══════════════════════════════════════════════════════════

def estimate_key_from_midi(midi_path: str) -> str:
    """简化的 K-S 调性估计 (超大文件采样)"""
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
        pitches = []
        for inst in pm.instruments:
            if not inst.is_drum:
                for note in inst.notes:
                    pitches.append(note.pitch % 12)
                    if len(pitches) > 2000:  # 超过 2000 个音符就采样
                        break
        if not pitches:
            return "C_major"
        # 最多取 2000 个音符做 KS

        counts = np.bincount(pitches, minlength=12).astype(float)
        counts = counts / counts.sum()
        major = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
        minor = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
        names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
        best_key, best_score = "C_major", -1
        for i in range(12):
            r = np.roll(counts, -i)
            ms = np.corrcoef(r, major)[0,1]
            mn = np.corrcoef(r, minor)[0,1]
            if ms > best_score:
                best_score = ms
                best_key = f"{names[i]}_major"
            if mn > best_score:
                best_score = mn
                best_key = f"{names[i]}_minor"
        return best_key
    except:
        return "C_major"


def get_midi_meta(midi_path: str) -> dict:
    """从 MIDI 文件提取 tempo + key + meter + duration"""
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
        t = pm.get_tempo_changes()
        bpm = float(np.mean(t[1])) if len(t[0]) > 0 else 120
        ts = pm.time_signature_changes
        meter = f"{ts[0].numerator}/{ts[0].denominator}" if len(ts) > 0 else "4/4"
        dur = pm.get_end_time()
        key = estimate_key_from_midi(midi_path)
        return {"bpm": round(bpm), "meter": meter, "duration": round(dur, 1), "key": key}
    except:
        return {"bpm": 120, "meter": "4/4", "duration": 0, "key": "C_major"}


# ═══════════════════════════════════════════════════════════
# 2. Token 生成器
# ═══════════════════════════════════════════════════════════

def purity_to_dynamics(purity: float) -> str:
    """基于二胡纯度推断动态等级"""
    if purity > 0.7:  return "mp"      # 高纯度 → 细腻
    if purity > 0.4:  return "mf"      # 中纯度 → 表现力
    return "f"                          # 低纯度 → 丰富


def purity_to_texture(purity: float, total_notes: int) -> str:
    """基于纯度和音符总数推断织体"""
    if total_notes < 100:
        return "sparse"
    if total_notes < 500:
        return "medium" if purity > 0.3 else "dense"
    return "dense"


def purity_to_articulation(purity: float) -> str:
    """基于纯度推断奏法"""
    if purity > 0.6:  return "legato"
    if purity > 0.3:  return "legato"
    return "marcato"


def key_to_color(key: str) -> str:
    """调性 → 音色倾向"""
    if "minor" in key:  return "dark"
    return "warm"


def generate_tokens(meta: dict, midi_meta: dict) -> str:
    """组装六维 Token 序列"""
    lines = []

    bpm = midi_meta["bpm"]
    key = midi_meta["key"]
    meter = midi_meta["meter"]
    purity = meta.get("erhu_purity", 0.5)
    total = meta.get("total_notes", 200)

    dyn = purity_to_dynamics(purity)
    tex = purity_to_texture(purity, total)
    art = purity_to_articulation(purity)
    color = key_to_color(key)

    # 全局
    lines.append(f"[GLOBAL:STYLE:erhu_solo]")
    lines.append(f"[GLOBAL:TEMPO:{bpm}]")
    lines.append(f"[GLOBAL:KEY:{key}]")
    lines.append(f"[GLOBAL:TIME:{meter}]")
    lines.append(f"[GLOBAL:MOOD:gufeng|instrumental]")
    lines.append("")

    # 结构 (简化: 默认 verse + chorus)
    sections = [
        ("intro",   "p",   "sparse",    "legato",  color, bpm-5),
        ("verse1",  "mp",  tex,         art,       color, bpm),
        ("chorus1", "f",   "dense",     "marcato", "bright", bpm+8),
        ("verse2",  "mp",  tex,         art,       color, bpm),
        ("outro",   "p",   "sparse",    "legato",  color, bpm-10),
    ]

    bar = 1
    for sec, s_dyn, s_tex, s_art, s_color, s_tempo in sections:
        lines.append(f"[SEC:{sec}|BAR:{bar}-{bar+7}]")
        lines.append(f"  [STEM:erhu]")
        lines.append(f"    [PATTERN:auto|DYN:{s_dyn}|ART:{s_art}|TEX:{s_tex}|COLOR:{s_color}|TEMPO:{s_tempo}]")
        lines.append(f"  [/STEM]")
        lines.append("")
        bar += 8

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# 3. Text Prompt 模板 (4 种随机切换)
# ═══════════════════════════════════════════════════════════

TEMPLATES = [
    # 模板 1: classic
    lambda n, m: (
        f"A traditional Chinese gufeng melody played by erhu, "
        f"{m['bpm']} bpm, {m['key'].replace('_',' ')} key, {m['meter']} meter."
    ),
    # 模板 2: mood-based
    lambda n, m: (
        f"Erhu solo piece \"{n}\" in {m['key'].replace('_',' ')} at {m['bpm']} bpm, "
        f"gufeng instrumental style with expressive phrasing."
    ),
    # 模板 3: quality
    lambda n, m: (
        f"A {'pure' if m.get('purity',0) > 0.5 else 'rich'} erhu performance of "
        f"traditional Chinese music, {m['bpm']} bpm, {m['key'].replace('_',' ')}."
    ),
    # 模板 4: short
    lambda n, m: (
        f"Gufeng erhu music: {m['key'].replace('_',' ')}, "
        f"{m['bpm']} bpm, Chinese traditional instrumental."
    ),
]


def generate_text(meta: dict, midi_meta: dict) -> str:
    """随机选择一个模板生成英文 Text Prompt"""
    midi_meta["purity"] = meta.get("erhu_purity", 0.5)
    template = random.choice(TEMPLATES)
    return template(meta.get("track_name", "untitled"), midi_meta)


# ═══════════════════════════════════════════════════════════
# 4. 主流程
# ═══════════════════════════════════════════════════════════

def main():
    # 收集所有成功目录
    track_dirs = sorted(
        d for d in DATA_DIR.iterdir()
        if d.is_dir() and (d / "meta.json").exists() and (d / "mix.wav").exists()
    )
    print(f"找到 {len(track_dirs)} 个成功渲染的 Track")

    if not track_dirs:
        print("没有找到数据！")
        return

    # 统计
    stats = Counter()
    rows = 0

    with open(OUTPUT, "w", encoding="utf-8") as f_out:
        pbar = tqdm(track_dirs, desc="打包", unit="首")
        for d in pbar:
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            name = meta.get("track_name", d.name)
            pbar.set_postfix_str(name[:20])

            # 跳过渲染失败的
            if not meta.get("render_success"):
                continue

            # 从 MIDI 提取调性/拍号
            midi_path = meta.get("midi_file", "")
            if midi_path and os.path.exists(midi_path):
                midi_meta = get_midi_meta(midi_path)
            else:
                midi_meta = {"bpm": meta.get("tempo", 120), "meter": "4/4",
                             "duration": 0, "key": "C_major"}

            # 生成 Token
            tokens = generate_tokens(meta, midi_meta)

            # 生成 Text
            text = generate_text(meta, midi_meta)

            # 打包一行
            row = {
                "track_name": name,
                "text": text,
                "tokens": tokens,
                "audio_path": str((d / "mix.wav").resolve()),
                "meta": {
                    "bpm": midi_meta["bpm"],
                    "key": midi_meta["key"],
                    "meter": midi_meta["meter"],
                    "erhu_purity": meta.get("erhu_purity", 0),
                    "duration_sec": midi_meta["duration"],
                },
            }
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows += 1

            # 统计
            stats["key_" + midi_meta["key"]] += 1
            stats[f"bpm_{midi_meta['bpm']//10*10}"] += 1

    # 汇总
    print(f"\n{'=' * 60}")
    print(f"训练数据集已生成!")
    print(f"  输出:     {OUTPUT}")
    print(f"  三元组:   {rows} 组")
    print(f"  每行:    text + tokens + audio_path + meta")
    print(f"\n调性分布 (Top 5):")
    for k, v in stats.most_common(5):
        if k.startswith("key_"):
            print(f"    {k[4:]:20s} {v} 首")

    # 打印一个样本
    print(f"\n样本展示 (第一行):")
    with open(OUTPUT, encoding="utf-8") as f:
        sample = json.loads(f.readline())
    print(f"  text: {sample['text']}")
    print(f"  tokens (前 300 字符):")
    for line in sample["tokens"].split("\n")[:12]:
        print(f"    {line}")
    print(f"  audio: .../{Path(sample['audio_path']).parent.name}/mix.wav")

    # 文件大小
    total_mb = os.path.getsize(OUTPUT) / 1024 / 1024
    print(f"\n文件大小: {total_mb:.1f} MB")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
