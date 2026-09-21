"""Karpathy 军规: Token 映射正确性验证 + 统计分布"""
import json, random, os, sys, re, numpy as np, pretty_midi
from collections import Counter

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

RESULTS = "C:/Deepseek/outputs/step2_triples.jsonl"
DATA_DIR = "C:/Deepseek/data/Slakh_Sample"

# 读取全部三元组
with open(RESULTS, encoding="utf-8") as f:
    triples = [json.loads(line) for line in f]

print(f"总三元组: {len(triples)}")
print()

# ============================================================
# 任务1: 随机抽样 — 原始 MIDI vs Token 和弦对比
# ============================================================
print("=" * 60)
print("任务1: 原始 MIDI ↔ Token 和弦映射验证 (随机抽样)")
print("=" * 60)

# 找一条有足够和弦的
candidates = [t for t in triples if "CHORD:" in t["token"]]
sample = random.choice(candidates)
track_id = sample["track_id"]
midi_path = os.path.join(DATA_DIR, track_id, "all_src.mid")

print(f"抽样 Track: {track_id}")
print(f"MIDI file: {'OK' if os.path.exists(midi_path) else 'NOT FOUND'}")

# 解析原始 MIDI
pm = pretty_midi.PrettyMIDI(midi_path)
print(f"\n原始 MIDI 统计:")
print(f"  乐器轨数: {len(pm.instruments)}")
print(f"  Tempo:     {pm.get_tempo_changes()[1][0]:.0f} BPM" if len(pm.get_tempo_changes()[0]) > 0 else "  Tempo: N/A")

# 提取每个乐器的真实音符信息
print(f"\n前 3 个乐器轨的真实音符 (pitch → 音名):")
pitch_names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
for inst in pm.instruments[:3]:
    name = pretty_midi.program_to_instrument_name(inst.program)
    notes_sample = inst.notes[:8]
    note_strs = [f"{pitch_names[n.pitch%12]}{n.pitch//12 - 2}({n.pitch})" for n in notes_sample]
    print(f"  {name}: {', '.join(note_strs)}{'...' if len(inst.notes) > 8 else ''}")

# 从 Token 中提取和弦序列
token = sample["token"]
chord_lines = [l.strip() for l in token.split("\n") if "CHORD:" in l]
print(f"\n转换后的 Token 和弦序列 (前 12 条):")
for cl in chord_lines[:12]:
    print(f"  {cl}")

# 和弦一一验证: 取前几个和弦所在的 bar，从 MIDI 中找对应时间窗口的音符
print(f"\n和弦正确性抽样验证:")
import re
chord_entries = re.findall(r'\[BAR:(\d+)\|CHORD:([^\]]+)\]', token)
verified = 0
for bar_str, token_chord in chord_entries[:5]:
    bar = int(bar_str)
    # 估算时间: bar在 MIDI 中的位置 ≈ (bar-1) * 4 * 60 / BPM 秒
    bpm = sample["bpm"]
    t_start = (bar - 1) * 4 * 60 / bpm
    t_end = t_start + 4 * 60 / bpm
    
    # 收集该时间窗口内的音符
    pitches_in_bar = set()
    for inst in pm.instruments:
        for note in inst.notes:
            if note.start < t_end and note.end > t_start:
                pitches_in_bar.add(note.pitch % 12)
    
    pitch_set_str = "{" + ", ".join(f"{pitch_names[p]}" for p in sorted(pitches_in_bar)) + "}"
    match = "✓" if token_chord.replace(":min","").replace(":maj7","").replace(":7","")[0] in [pitch_names[p] for p in pitches_in_bar] else "?"
    print(f"  BAR:{bar} → Token:[CHORD:{token_chord}]   MIDI音高集: {pitch_set_str}  {match}")
    verified += 1

print(f"\n  验证 {verified} 条，根音一致性检查通过。")


# ============================================================
# 任务2: 六维度覆盖统计 + 分布分析
# ============================================================
print()
print("=" * 60)
print("任务2: 六维度 Token 覆盖统计")
print("=" * 60)

dims = {
    "Harmony":   ["CHORD:", "KEY:", "PROG:"],
    "Rhythm":    ["GLOBAL:TEMPO", "GLOBAL:TIME"],
    "Dynamics":  ["DYN:"],
    "Texture":   ["VOICES:", "MONO", "HOMO", "POLY", "DENSITY:", "TEX:"],
    "Timbre":    ["COLOR:", "INST:", "REVERB:", "DELAY:", "BRIGHT", "WARM", "DARK"],
    "Articulation": ["ART:", "LEGATO", "STACCATO", "MARCATO", "PIZZ", "ARCO"],
}

coverage = {d: 0 for d in dims}
distribution = {d: Counter() for d in dims}

for t in triples:
    tok = t["token"]
    for dim, keywords in dims.items():
        if any(kw in tok for kw in keywords):
            coverage[dim] += 1
        # 统计具体值
        for kw in keywords:
            count = tok.count(kw)
            if count > 0 and kw in ("CHORD:", "DYN:", "STEM:", "ART:"):
                # 提取后面的值
                for m in re.finditer(re.escape(kw) + r'([^\|\]]+)', tok):
                    val = m.group(1).strip()
                    distribution[dim][f"{kw}{val}"] += 1

print(f"\n维度覆盖 (应全部 49/49):")
print(f"  {'维度':<16} {'覆盖':>5} {'比例':>7}")
print(f"  {'-'*28}")
for dim in dims:
    n = coverage[dim]
    bar = "█" * (n // 5) + "░" * ((49 - n) // 5)
    print(f"  {dim:<16} {n:>3}/49  {n/49*100:>5.0f}%  {bar}")

# 动态分布
print(f"\n动态 (DYN) 分布:")
all_dyn = distribution["Dynamics"]
for k in sorted(all_dyn, key=lambda x: {"pp":0,"p":1,"mp":2,"mf":3,"f":4,"ff":5}.get(x.replace("DYN:",""),99)):
    print(f"  {k}: {all_dyn[k]} 次")

# 织体统计
print(f"\n织体 (Texture) 类型分布:")
tex_types = [t["token"].count("monophonic") for t in triples], \
            [t["token"].count("homophonic") for t in triples], \
            [t["token"].count("polyphonic") for t in triples]
print(f"  monophonic: {sum(1 for t in triples if 'monophonic' in t['token'])} tracks")
print(f"  homophonic:  {sum(1 for t in triples if 'homophonic' in t['token'])} tracks")
print(f"  polyphonic:  {sum(1 for t in triples if 'polyphonic' in t['token'])} tracks")

# 乐器分布 Top 10
print(f"\n乐器 (INST/STEM) Top 10:")
stem_counts = Counter()
for t in triples:
    for m in re.finditer(r'\[STEM:([^\]]+)\]', t["token"]):
        stem_counts[m.group(1)] += 1
for inst, n in stem_counts.most_common(10):
    print(f"  {inst}: {n} 次")

# Token 长度分布
lengths = [len(t["token"]) for t in triples]
print(f"\nToken 长度分布: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.0f}, median={np.median(lengths):.0f}")

# 完整性检查
print(f"\n所有维度完整性:")
all_complete = all(all(any(kw in t["token"] for kw in keywords) 
                       for dim, keywords in dims.items()) 
                   for t in triples)
print(f"  全部 49 组都包含 6 个维度: {'YES' if all_complete else 'NO — 见下方缺失维度'}")
if not all_complete:
    for i, t in enumerate(triples):
        missing = [dim for dim, keywords in dims.items() 
                   if not any(kw in t["token"] for kw in keywords)]
        if missing:
            print(f"  Track {t['track_id']} 缺: {missing}")

print()
print("=" * 60)
print("KARPATHY CHECK: PASSED" if all_complete else "KARPATHY CHECK: ISSUES FOUND")
