#!/usr/bin/env python3
"""
slakh_to_tokens.py — Slakh2100 → PDF「镜头语言」Token 转换器
==============================================================
输入: Slakh2100 的一个 Track 目录（含 metadata.yaml + MIDI 文件 + mix.flac）
输出: 六维度结构化 Token 序列（对应 PDF 报告第 4-5 章定义的词汇表和语法）

Token 维度:
  Harmony  [KEY][CHORD][PROG][CONS/DISS]
  Rhythm   [BPM][TIME][SWING/STRAIGHT][SYNC][DENS]
  Texture  [MONO/HOMO/POLY][VOICES][DENSITY]
  Dynamics [DYN][CRESC/DIM][RANGE]
  Timbre   [INST][REVERB][DELAY][BRIGHT/WARM/DARK]
  Articulation [LEGATO/STACCATO/MARCATO][PIZZ/ARCO][ACCENT]

输出格式（策略 B：结构化标签）:
  [TEMPO:120] [KEY:C] [4/4]
  [0:00-0:10|SEC:intro|INST:piano|ART:legato|DYN:mp|TEX:sparse|VOICES:2]
  [BAR:1-4|CHORD:C] [BAR:5-8|CHORD:G/B] ...

依赖: pip install pretty_midi pyyaml numpy
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import yaml
import numpy as np
import pretty_midi


# ============================================================
# 第1层: MIDI 基础解析
# ============================================================

def load_slakh_track(track_dir: str) -> dict:
    """加载 Slakh2100 单个 Track 的全部数据"""
    track_path = Path(track_dir)
    
    # 1. metadata.yaml → 乐器信息
    # 1. metadata.yaml（可选，HF 版无此文件）
    metadata = {}
    meta_file = track_path / "metadata.yaml"
    if meta_file.exists():
        with open(meta_file, encoding="utf-8") as f:
            metadata = yaml.safe_load(f)
    
    # 2. 收集所有 MIDI 文件
    midi_files = sorted(track_path.glob("**/*.mid")) + sorted(track_path.glob("**/*.midi"))
    
    # 3. mix 音频（.flac 或 .wav）
    mix_audio = track_path / "mix.flac"
    if not mix_audio.exists():
        mix_audio = track_path / "mix.wav"
    
    return {
        "track_id": track_path.name,
        "metadata": metadata,
        "midi_files": midi_files,
        "mix_audio": str(mix_audio) if mix_audio.exists() else None,
    }


def midi_to_notes(midi_path: str) -> List[dict]:
    """将单个 MIDI 文件转为音符列表"""
    pm = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    for instrument in pm.instruments:
        inst_name = pretty_midi.program_to_instrument_name(instrument.program)
        for note in instrument.notes:
            notes.append({
                "pitch": note.pitch,
                "velocity": note.velocity,
                "start": note.start,
                "end": note.end,
                "duration": note.end - note.start,
                "instrument": inst_name,
                "program": instrument.program,
            })
    return sorted(notes, key=lambda n: n["start"])


# ============================================================
# 第2层: 六维度 Token 提取
# ============================================================

def extract_tempo_and_meter(midi_path: str) -> Dict:
    """提取节奏维度: BPM + Time Signature"""
    pm = pretty_midi.PrettyMIDI(midi_path)
    tempo_changes = pm.get_tempo_changes()
    
    if len(tempo_changes[0]) > 0:
        # 使用第一个 tempo 或平均 tempo
        bpm = float(np.mean(tempo_changes[1])) if len(tempo_changes[1]) > 0 else 120.0
    else:
        bpm = 120.0
    
    time_sigs = pm.time_signature_changes
    if len(time_sigs) > 0:
        ts = time_sigs[0]
        meter = f"{ts.numerator}/{ts.denominator}"
    else:
        meter = "4/4"
    
    return {"bpm": round(bpm), "meter": meter}


def estimate_key(notes: List[dict]) -> str:
    """简化的调性估计（基于音高分布，Krumhansl-Schmuckler 类似方法）"""
    # 12 个半音的分布
    pitch_counts = np.zeros(12)
    for n in notes:
        pitch_counts[n["pitch"] % 12] += n["duration"]
    
    # 大调和小调模板（简化版 K-S 权重）
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                               2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                               2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    
    total = pitch_counts.sum()
    if total == 0:
        return "C_major"
    
    norm = pitch_counts / total
    
    # 旋转匹配
    best_key, best_score = "C_major", -1
    pitch_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    
    for i in range(12):
        rotated = np.roll(norm, -i)
        major_score = np.corrcoef(rotated, major_profile)[0, 1]
        minor_score = np.corrcoef(rotated, minor_profile)[0, 1]
        if major_score > best_score:
            best_score = major_score
            best_key = f"{pitch_names[i]}_major"
        if minor_score > best_score:
            best_score = minor_score
            best_key = f"{pitch_names[i]}_minor"
    
    return best_key


def detect_chords(notes: List[dict], time_resolution: float = 0.5) -> List[dict]:
    """从音符序列检测和弦（时间窗口 + 音符集合 → 和弦简写）"""
    if not notes:
        return []
    
    max_time = max(n["end"] for n in notes)
    num_windows = int(np.ceil(max_time / time_resolution))
    chords = []
    
    for i in range(num_windows):
        t_start = i * time_resolution
        t_end = (i + 1) * time_resolution
        
        # 收集该窗口内活跃的音符
        active_pitches = set()
        for n in notes:
            if n["start"] < t_end and n["end"] > t_start:
                active_pitches.add(n["pitch"] % 12)
        
        if len(active_pitches) >= 2:
            chord_name = pitches_to_chord(active_pitches)
            if chord_name:
                chords.append({
                    "time": round(t_start, 1),
                    "end_time": round(t_end, 1),
                    "chord": chord_name,
                })
    
    return chords


def pitches_to_chord(pitches: set) -> Optional[str]:
    """音符集合 → 和弦简写 (M3+m3=Major, m3+M3=minor, M3+M3=aug, m3+m3=dim)"""
    if len(pitches) < 2:
        return None
    
    pitch_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    sorted_p = sorted(pitches)
    
    # 尝试以每个音为根音构建和弦
    for root in sorted_p:
        third = (root + 4) % 12  # 大三度
        min_third = (root + 3) % 12  # 小三度
        fifth = (root + 7) % 12  # 纯五度
        seventh = (root + 10) % 12  # 小七度
        maj_seventh = (root + 11) % 12  # 大七度
        
        remaining = set(sorted_p) - {root}
        
        if third in sorted_p and fifth in sorted_p:
            chord = pitch_names[root]
            if maj_seventh in sorted_p:
                return f"{chord}:maj7"
            if seventh in sorted_p:
                return f"{chord}:7"
            return chord  # Major
        
        if min_third in sorted_p and fifth in sorted_p:
            chord = pitch_names[root]
            if seventh in sorted_p:
                return f"{chord}:min7"
            return f"{chord}:min"
    
    return None


def map_dynamics(velocity: int) -> str:
    """MIDI velocity → 动态等级 (pp→ff)"""
    if velocity <= 20:   return "pp"
    if velocity <= 40:   return "p"
    if velocity <= 60:   return "mp"
    if velocity <= 80:   return "mf"
    if velocity <= 100:  return "f"
    return "ff"


def map_articulation(duration_ms: float) -> str:
    """根据音符时长（ms）推测奏法"""
    if duration_ms < 100:
        return "staccato"
    if duration_ms < 250:
        return "marcato"
    return "legato"


def map_texture_info(notes: List[dict]) -> Dict:
    """从音符统计推导织体信息"""
    if not notes:
        return {"type": "monophonic", "voices": 0, "density": "sparse"}
    
    # 同时发声的最大音符数 ≈ 声部数
    events = []
    for n in notes:
        events.append((n["start"], 1))
        events.append((n["end"], -1))
    events.sort()
    
    current = 0
    max_voices = 0
    for _, delta in events:
        current += delta
        max_voices = max(max_voices, current)
    
    if max_voices <= 2:
        tex_type = "monophonic" if max_voices == 1 else "homophonic"
    else:
        tex_type = "polyphonic"
    
    if max_voices <= 2:
        density = "sparse"
    elif max_voices <= 6:
        density = "medium"
    else:
        density = "dense"
    
    return {"type": tex_type, "voices": max_voices, "density": density}


def map_color(instruments: List[str]) -> str:
    """从乐器列表推断整体音色倾向"""
    warm_insts = {"Acoustic Grand Piano", "String Ensemble", "Cello", "Viola", 
                  "Fretless Bass", "Acoustic Guitar (nylon)", "Pad", "Choir"}
    bright_insts = {"Bright Acoustic Piano", "Electric Piano", "Electric Guitar",
                    "Overdriven Guitar", "Brass", "Trumpet", "Flute", "Piccolo"}
    dark_insts = {"Electric Bass", "Synth Bass", "Contrabass"}
    warm = sum(1 for i in instruments if i in warm_insts)
    bright = sum(1 for i in instruments if i in bright_insts)
    dark = sum(1 for i in instruments if i in dark_insts)
    if bright > warm and bright > dark: return "bright"
    if dark > warm and dark > bright:   return "dark"
    if warm > 0:                        return "warm"
    return "neutral"


# ============================================================
# 第3层: 分段 (Section Segmentation)
# ============================================================

def segment_by_texture_change(chords: List[dict], texture_history: List[dict]) -> List[dict]:
    """根据和弦变化点分段（简化版：每 8 小节或每和声变化分段）"""
    if not chords:
        return [{"bar_start": 1, "bar_end": 8, "label": "section"}]
    
    segments = []
    # 简化：按 4 个和弦为一组（约 4 小节一段）
    for i in range(0, len(chords), 4):
        chunk = chords[i:i+4]
        if chunk:
            segments.append({
                "bar_start": i + 1,
                "bar_end": min(i + 4, len(chords)),
                "label": "verse" if i < len(chords)//2 else "chorus",
                "chords": [c["chord"] for c in chunk],
            })
    
    return segments


# ============================================================
# 第4层: Token 序列组装
# ============================================================

class TokenSequence:
    """组装 PDF 定义的六维度 Token"""
    
    def __init__(self):
        self.tokens = []
        self.global_header = {}
    
    def set_global(self, key: str, value):
        """设置全局 Token"""
        self.global_header[key] = value
    
    def add_section(self, name: str, bar_start: int, bar_end: int):
        """开始新段落"""
        self.tokens.append(f"\n[SEC:{name}|BAR:{bar_start}-{bar_end}]")
    
    def add_stem(self, instrument: str):
        """开始新声部"""
        self.tokens.append(f"  [STEM:{instrument}]")
    
    def close_stem(self):
        self.tokens.append("  [/STEM]")
    
    def add_bar_chord(self, bar: int, chord: str, duration_bars: int = 1):
        """添加逐小节和弦"""
        if duration_bars == 1:
            self.tokens.append(f"    [BAR:{bar}|CHORD:{chord}]")
        else:
            self.tokens.append(f"    [BAR:{bar}-{bar+duration_bars-1}|CHORD:{chord}]")
    
    def add_pattern(self, pattern: str, dyn: str, reg: str, art: str, tex_type: str = "", voices: int = 0):
        """添加演奏模式 + 动态 + 音区 + 奏法 + 织体"""
        extra = ""
        if tex_type:
            extra += f"|TEX:{tex_type}"
        if voices:
            extra += f"|VOICES:{voices}"
        self.tokens.append(
            f"    [PATTERN:{pattern}|DYN:{dyn}|REG:{reg}|ART:{art}{extra}]"
        )
    
    def render(self) -> str:
        """输出完整 Token 序列"""
        lines = []
        # 全局头
        for k, v in sorted(self.global_header.items()):
            lines.append(f"[GLOBAL:{k}:{v}]")
        lines.append("")
        
        # Token 序列
        for token in self.tokens:
            lines.append(token)
        
        return "\n".join(lines)


def generate_text_prompt(instruments: List[str], genre_hint: str = "", bpm: int = 120) -> str:
    """生成配套的自然语言 Text Prompt"""
    inst_str = ", ".join(instruments[:4])
    prompt = f"{genre_hint} piece with {inst_str}, {bpm} bpm"
    return prompt


def process_one_track(track_dir: str, output_dir: str = None) -> dict:
    """
    主流程: 一个 Slakh Track → (Text + Token + Audio 路径) 三元组
    
    返回:
      {"text": "...", "token": "[GLOBAL:...]", "audio": "path/to/mix.flac"}
    """
    data = load_slakh_track(track_dir)
    
    # 合并所有 MIDI 文件的音符
    all_notes = []
    for mf in data["midi_files"]:
        all_notes.extend(midi_to_notes(str(mf)))
    
    if not all_notes:
        print(f"WARN: No notes in {track_dir}")
        return None
    
    # 提取全局参数（从第一个 MIDI）
    rhythm = extract_tempo_and_meter(str(data["midi_files"][0]))
    key_est = estimate_key(all_notes)
    
    # 乐器列表
    instruments = sorted(set(n["instrument"] for n in all_notes))
    
    # 生成 Text Prompt
    text = generate_text_prompt(instruments, bpm=rhythm["bpm"])
    
    # 构建 Token 序列
    ts = TokenSequence()
    ts.set_global("TEMPO", rhythm["bpm"])
    ts.set_global("KEY", key_est)
    ts.set_global("TIME", rhythm["meter"])
    ts.set_global("MOOD", "instrumental")
    
    # 按乐器分声部
    from itertools import groupby
    notes_by_inst = defaultdict(list)
    for n in all_notes:
        notes_by_inst[n["instrument"]].append(n)
    
    # 和弦检测（基于所有音符）
    chords = detect_chords(all_notes)
    
    # 分段
    segments = segment_by_texture_change(chords, [])
    
    bar_idx = 1
    color = map_color(instruments)
    ts.set_global("COLOR", color)
    for seg in segments:
        ts.add_section(seg["label"], seg["bar_start"], seg["bar_end"])
        
        for inst_name, inst_notes in sorted(notes_by_inst.items()):
            ts.add_stem(inst_name)
            
            # 和弦
            for chord_info in chords:
                bar_num = int(chord_info["time"] * rhythm["bpm"] / 60 / 4) + 1
                if seg["bar_start"] <= bar_num <= seg["bar_end"]:
                    ts.add_bar_chord(bar_num, chord_info["chord"])
            
            # 乐器的平均动态和奏法
            velocities = [n["velocity"] for n in inst_notes]
            durations = [n["duration"] * 1000 for n in inst_notes]
            
            avg_vel = int(np.mean(velocities)) if velocities else 64
            avg_dur = float(np.mean(durations)) if durations else 300
            
            dyn = map_dynamics(avg_vel)
            art = map_articulation(avg_dur)
            
            # 音区判断
            pitches = [n["pitch"] for n in inst_notes]
            avg_pitch = int(np.mean(pitches)) if pitches else 60
            reg = "low" if avg_pitch < 48 else ("high" if avg_pitch > 72 else "mid")
            
            # 织体信息
            tex_info = map_texture_info(inst_notes)
            
            ts.add_pattern("auto", dyn, reg, art, 
                          tex_type=tex_info["type"], 
                          voices=tex_info["voices"])
            ts.close_stem()
    
    token_str = ts.render()
    
    return {
        "track_id": data["track_id"],
        "text": text,
        "token": token_str,
        "audio": data["mix_audio"],
        "instruments": instruments,
        "key": key_est,
        "bpm": rhythm["bpm"],
        "meter": rhythm["meter"],
    }


# ============================================================
# 第5层: 批量处理 + 质量过滤
# ============================================================

def quality_filter(result: dict) -> bool:
    """质量过滤: 淘汰不合格的三元组"""
    if result is None:
        return False
    if result["audio"] is None:
        return False
    if len(result["instruments"]) < 3:
        return False  # 太简单，跳过
    if len(result["token"]) < 100:
        return False  # Token 太短
    return True


def batch_process(slakh_root: str, output_jsonl: str, max_tracks: int = 500):
    """批量处理 Slakh2100 全部 Track，输出 JSONL"""
    import random
    
    all_track_dirs = sorted(Path(slakh_root).glob("Track*"))
    # 排除 omitted 目录（重复 MIDI）
    all_track_dirs = [d for d in all_track_dirs 
                      if "omitted" not in str(d).lower() 
                      and d.is_dir()]
    
    random.shuffle(all_track_dirs)
    
    processed = 0
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for track_dir in all_track_dirs:
            result = process_one_track(str(track_dir))
            if quality_filter(result):
                # 精简输出（去掉内部列表）
                out = {k: v for k, v in result.items() 
                       if k in ("track_id", "text", "token", "audio", "key", "bpm", "meter")}
                out["instruments_count"] = len(result["instruments"])
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                processed += 1
                if processed % 50 == 0:
                    print(f"  Progress: {processed}/{max_tracks}")
                if processed >= max_tracks:
                    break
    
    print(f"\nDone: {processed} valid triples saved to {output_jsonl}")
    return processed


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Slakh2100 → PDF「镜头语言」Token 转换器"
    )
    parser.add_argument("slakh_root", help="Slakh2100 根目录 (含 TrackXXXXX 子目录)")
    parser.add_argument("--output", default="C:/Deepseek/outputs/step2_triples.jsonl",
                        help="输出 JSONL 路径")
    parser.add_argument("--max", type=int, default=500,
                        help="最大输出三元组数量 (默认: 500)")
    parser.add_argument("--demo", metavar="TRACK_DIR",
                        help="Demo 模式: 仅处理单个 Track 并打印 Token")
    
    args = parser.parse_args()
    
    if args.demo:
        result = process_one_track(args.demo)
        if result:
            print("=" * 60)
            print("TEXT PROMPT:")
            print(result["text"])
            print("\n" + "=" * 60)
            print("TOKEN SEQUENCE:")
            print(result["token"])
            print("\n" + "=" * 60)
            print(f"Audio: {result['audio']}")
            print(f"Key: {result['key']} | BPM: {result['bpm']} | Meter: {result['meter']}")
            print(f"Instruments: {result['instruments']}")
        else:
            print("ERROR: Failed to process track")
    else:
        batch_process(args.slakh_root, args.output, args.max)
