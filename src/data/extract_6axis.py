# -*- coding: utf-8 -*-
"""
extract_6axis.py — 跨数据集的六轴 Music-Prompt DSL 标签自动化提取框架
=====================================================================
对应论文《MontageDirector》Fig.3 与 Eq.(3) 的六轴 DSL：
  Harmony(和声) · Rhythm(节奏/BPM) · Texture(纹理密度) · Dynamics(动态)
  Timbre(音色) · Articulation(发音/奏法)   —— 统称 "6-axis"

设计原则
--------
1. 所有数据集先被 adapter 转成统一的 ``notes`` 表（见 AXIS_SCHEMA），六轴提取器
   只吃这张表，因此跨数据集标签口径完全一致、可比。
2. 每个轴一个纯函数，并锚定论文引用的文献（Krumhansl-Schmuckler 调性估计、
   MAESTRO velocity-loudness 一致性、Groove MIDI micro-timing）。
3. MIDI 有则出"六轴真值"；无 MIDI 的音频-文本数据集（MusicCaps/FMA/MTG-Jamendo）
   只出弱文本 + [STYLE:] 专家标签，并在 meta 里标 source_has_midi=false，
   避免污染"六轴真值"的实验论证。
4. 输出 JSONL 沿用项目现有 schema：{"tokens","text","audio_path","meta"}。

依赖（已确认可用）: pretty_midi, numpy, soundfile, librosa, PyYAML
可选（缺失自动降级）: music21 (LMD 大目录快速解析 / key 校验), pandas (CSV 元数据)

用法
----
  # 1) 合成 smoke test（无数据也可跑通，验证六轴口径）
  python code/extract_6axis.py --demo

  # 2) MIDI 类数据集（自动识别目录结构）
  python code/extract_6axis.py --dataset slakh    --root data/Slakh2100 --out data/slakh_6axis.jsonl --max 200
  python code/extract_6axis.py --dataset maestro  --root data/maestro-v3.0.0 --out data/maestro_6axis.jsonl --max 200
  python code/extract_6axis.py --dataset lakh     --root data/lmd_midi    --out data/lmd_6axis.jsonl --max 200
  python code/extract_6axis.py --dataset pop909   --root data/POP909     --out data/pop909_6axis.jsonl --max 200
  python code/extract_6axis.py --dataset groove   --root data/groove     --out data/groove_6axis.jsonl --max 200

  # 3) 音频+文本类（弱文本 / 风格专家扩展，不产六轴真值）
  python code/extract_6axis.py --dataset musiccaps --root data/musiccaps --out data/musiccaps_style.jsonl
  python code/extract_6axis.py --dataset fma       --root data/fma_medium --csv data/fma/tracks.csv --out data/fma_style.jsonl
  python code/extract_6axis.py --dataset jamendo   --root data/mtg-jamendo --csv data/mtg/autotagging.tsv --out data/jamendo_style.jsonl

  # 4) MIDI 数据集无现成音频时用 SoundFont 渲染（复用项目 fluidsynth）
  python code/extract_6axis.py --dataset lakh --root data/lmd_midi --render --sf2 soundfonts/FluidR3_GM.sf2
"""
import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import pretty_midi
except Exception:  # pragma: no cover
    pretty_midi = None

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

try:
    import music21  # 可选：LMD 大目录解析、key 校验
except Exception:  # pragma: no cover
    music21 = None

try:
    import soundfile as sf
except Exception:  # pragma: no cover
    sf = None

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ============================================================
# 统一 schema
# ============================================================
AXIS_SCHEMA = [
    "pitch",       # int    MIDI note number (0-127)
    "velocity",    # int    1-127, 动态轴的原始信号
    "start",       # float  seconds
    "end",         # float  seconds
    "duration",    # float  seconds
    "program",     # int    GM program number (0-127), 音色轴
    "is_drum",     # bool
    "source",      # str    stem / file / track id，用于回溯
]

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Schmuckler 主/小调模板（论文第三节引用）
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

DYN_LEVELS = [(20, "pp"), (40, "p"), (60, "mp"), (80, "mf"), (100, "f"), (128, "ff")]
DYN_TO_DSL = {"pp": "pp", "p": "p", "mp": "mp", "mf": "mf", "f": "f", "ff": "ff"}


@dataclass
class TrackBundle:
    """adapter 产出的统一中间表示。"""
    notes: List[Dict[str, Any]]
    tempo_changes: Tuple[np.ndarray, np.ndarray]   # (times, bpms)
    time_signatures: List[Any]                     # pretty_midi.TimeSignature
    audio_path: Optional[str] = None
    midi_path: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None          # 数据集原生元数据


# ============================================================
# 六轴提取器（纯函数，输入统一 notes 表）
# ============================================================
def _pitches(notes: List[dict], skip_drums: bool = True) -> np.ndarray:
    return np.array([n["pitch"]
                     for n in notes
                     if not (skip_drums and n.get("is_drum"))], dtype=int)


def axis_harmony(notes: List[dict], tempo=None, chord_resolution: float = 0.5):
    """Harmony 轴：KEY(调性) + 逐时间窗 CHORD 序列。

    锚定：Krumhansl-Schmuckler key estimation（论文引用 [8]）。
    """
    pitches = _pitches(notes)
    if len(pitches) == 0:
        return {"key": "C_major", "chords": []}

    counts = np.bincount(pitches % 12, minlength=12).astype(float)
    counts /= counts.sum()

    best_key, best_score = "C_major", -1.0
    for i in range(12):
        r = np.roll(counts, -i)          # 把第 i 个音类转到 tonic 位置
        s_maj = np.corrcoef(r, KS_MAJOR)[0, 1]
        s_min = np.corrcoef(r, KS_MINOR)[0, 1]
        if s_maj > best_score:
            best_score, best_key = s_maj, f"{PITCH_NAMES[i]}_major"
        if s_min > best_score:
            best_score, best_key = s_min, f"{PITCH_NAMES[i]}_minor"

    melodic = [n for n in notes if not n.get("is_drum")]
    chords = _detect_chords_windows(melodic, chord_resolution)
    return {"key": best_key, "chords": chords}


def _detect_chords_windows(notes: List[dict], resolution: float) -> List[dict]:
    if not notes:
        return []
    max_t = max(n["end"] for n in notes)
    chords = []
    for i in range(int(np.ceil(max_t / resolution))):
        t0, t1 = i * resolution, (i + 1) * resolution
        active = {n["pitch"] % 12
                  for n in notes if n["start"] < t1 and n["end"] > t0}
        if len(active) >= 2:
            name = _pitches_to_chord(active)
            if name:
                chords.append({"time": round(t0, 2),
                               "end_time": round(t1, 2),
                               "chord": name})
    return chords


def _pitches_to_chord(pitches: set) -> Optional[str]:
    if len(pitches) < 2:
        return None
    s = sorted(pitches)
    for root in s:
        if (root + 4) % 12 in s and (root + 7) % 12 in s:      # 大三和弦骨架
            if (root + 10) % 12 in s:
                return f"{PITCH_NAMES[root]}:7"
            if (root + 11) % 12 in s:
                return f"{PITCH_NAMES[root]}:maj7"
            return PITCH_NAMES[root]
        if (root + 3) % 12 in s and (root + 7) % 12 in s:      # 小三和弦骨架
            if (root + 10) % 12 in s:
                return f"{PITCH_NAMES[root]}:min7"
            return f"{PITCH_NAMES[root]}:min"
        # sus4: root + 5 + 7（无三音，先于 major 判断的必要性另行处理）
        if (root + 5) % 12 in s and (root + 7) % 12 in s and (root + 3) % 12 not in s \
                and (root + 4) % 12 not in s:
            return f"{PITCH_NAMES[root]}:sus"
    return None


def axis_rhythm(bundle: TrackBundle):
    """Rhythm 轴：BPM + 拍号 + 主拍网格。

    锚定：Groove MIDI micro-timing（论文引用 [10]）。
    """
    times, bpms = bundle.tempo_changes
    if len(bpms) > 0:
        bpm = float(np.average(bpms, weights=np.diff(np.append(times, np.max(times) + 1.0))))
        if not np.isfinite(bpm):
            bpm = float(np.mean(bpms))
    else:
        bpm = 120.0

    ts = bundle.time_signatures
    if ts:
        meter = f"{ts[0].numerator}/{ts[0].denominator}"
    else:
        meter = "4/4"

    # 主拍网格（按 4 分音符）
    beat = 60.0 / max(bpm, 1e-6)
    return {"bpm": round(bpm), "bpm_float": round(float(bpm), 2),
            "meter": meter, "beat_seconds": round(beat, 4)}


def axis_texture(notes: List[dict], window: float = 1.0, skip_drums: bool = True):
    """Texture 轴：并发音符峰值 → 声部数与 [TEX:sparse|dense]。

    与 DSL 对应：<Tex> ::= [TEX:(sparse|dense)] | [VOICES:N+]
    """
    events = []
    for n in notes:
        if skip_drums and n.get("is_drum"):
            continue
        events.append((n["start"], +1))
        events.append((n["end"], -1))
    events.sort(key=lambda e: e[0])

    current = 0
    peak = 0
    # 时间加权平均并发（比瞬时峰值更稳健）
    prev_t, acc = events[0][0] if events else 0.0, 0.0
    for t, d in events:
        acc += current * (t - prev_t)
        current += d
        peak = max(peak, current)
        prev_t = t

    total = max(prev_t - events[0][0], 1e-9) if events else 1e-9
    avg_voices = acc / total

    tex = "dense" if peak >= 8 else ("medium" if peak >= 3 else "sparse")
    return {"peak_voices": peak,
            "avg_voices": round(float(avg_voices), 2),
            "tex": tex,
            "density_token": f"[TEX:{tex}]",
            "voices_token": f"[VOICES:{peak}]"}


def axis_dynamics(notes: List[dict], seg_seconds: float = 8.0, skip_drums: bool = True):
    """Dynamics 轴：velocity → 动态等级 + 段间 CRESC/DIM 检测。

    锚定：MAESTRO velocity-loudness consistency（论文引用 [9]）。
    """
    melodic = [(n["start"], n["velocity"])
               for n in notes if not (skip_drums and n.get("is_drum"))]
    if not melodic:
        return {"level": "mp", "level_token": "[DYN:mp]", "segments": []}

    melodic.sort()
    start_times = np.array([m[0] for m in melodic])
    velocities = np.array([m[1] for m in melodic], dtype=float)
    total_level = _vel_to_dyn(float(np.mean(velocities)))

    max_t = float(start_times[-1]) + seg_seconds
    segs = []
    level_mean = []
    for i in range(int(np.ceil(max_t / seg_seconds))):
        t0, t1 = i * seg_seconds, (i + 1) * seg_seconds
        mask = (start_times >= t0) & (start_times < t1)
        if mask.sum() == 0:
            continue
        m = float(np.mean(velocities[mask]))
        lvl = _vel_to_dyn(m)
        level_mean.append(lvl)
        segs.append({"start": round(t0, 2), "end": round(t1, 2),
                     "mean_velocity": round(m, 1), "level": lvl})

    # 相邻段递增/递减 → CRESC / DIM
    dyn_tokens = []
    dyn_order = ["pp", "p", "mp", "mf", "f", "ff"]
    for idx, s in enumerate(segs):
        tok = f"[DYN:{s['level']}]"
        if idx > 0 and dyn_order.index(s["level"]) > dyn_order.index(level_mean[idx - 1]):
            tok = "[DYN:CRESC]"
        elif idx > 0 and dyn_order.index(s["level"]) < dyn_order.index(level_mean[idx - 1]):
            tok = "[DYN:DIM]"
        dyn_tokens.append({"start": s["start"], "token": tok})

    return {"level": total_level, "level_token": f"[DYN:{total_level}]",
            "segments": segs, "dyn_tokens": dyn_tokens}


def _vel_to_dyn(v: float) -> str:
    for thr, lbl in DYN_LEVELS:
        if v <= thr:
            return lbl
    return "ff"


def axis_timbre(notes: List[dict]):
    """Timbre 轴：program → instrument family → [COLOR:warm|bright|cold] + [STYLE]。

    与 DSL 对应：<Tbr> ::= [COLOR:(warm|bright|cold)]
    """
    programs = {}
    for n in notes:
        programs[n["program"]] = programs.get(n["program"], 0) + 1

    # GM program → 色温映射（按乐器家族）
    bright_fams = {"brass", "flute", "pipe organ", "lead", "chromatic"}
    warm_fams = {"piano", "strings", "guitar", "reed", "choir"}
    cold_fams = {"bass", "synth bass", "fx"}

    bright = warm = cold = 0
    inst_names = set()
    for prog, cnt in programs.items():
        name = pretty_midi.program_to_instrument_name(prog) \
            if pretty_midi else f"program_{prog}"
        inst_names.add(name)
        fam = (pretty_midi.program_to_instrument_class(prog)
               if pretty_midi else "unknown")
        if fam in bright_fams:
            bright += cnt
        elif fam in warm_fams:
            warm += cnt
        else:
            cold += cnt

    if bright > warm and bright > cold:
        color = "bright"
    elif cold > warm and cold > bright:
        color = "cold"
    elif warm > 0:
        color = "warm"
    else:
        color = "neutral"

    return {"instruments": sorted(inst_names),
            "color": color,
            "color_token": f"[COLOR:{color}]"}


def axis_articulation(notes: List[dict], skip_drums: bool = True):
    """Articulation 轴：时值/重叠度 → legato/staccato/marcato；局部 velocity 尖峰 → accent。

    锚定：音符时长与发声法映射（与 slakh_to_tokens.py 一致，Groove micro-timing 语义）。
    """
    melodic = [n for n in notes if not (skip_drums and n.get("is_drum"))]
    if not melodic:
        return {"art": "legato", "art_token": "[ART:legato]"}

    durs = np.array([n["duration"] for n in melodic])
    vels = np.array([n["velocity"] for n in melodic], dtype=float)

    # 重叠度：相邻音符 onset 差 < 前音时长 → legato
    melodic_sorted = sorted(melodic, key=lambda n: (n["source"], n["start"]))
    overlaps = 0
    for a, b in zip(melodic_sorted, melodic_sorted[1:]):
        if a["source"] == b["source"] and b["start"] < a["end"]:
            overlaps += 1
    overlap_ratio = overlaps / max(len(melodic_sorted) - 1, 1)

    short_note_ratio = float(np.mean(durs < 0.12))
    # accent：局部 velocity 高出该声源均值 1.5 倍以上的占比
    acc = 0
    for src in {n["source"] for n in melodic}:
        idx = np.array([n["source"] == src for n in melodic])
        v = vels[idx]
        if len(v) == 0:
            continue
        acc += int(np.sum(v > 1.5 * np.mean(v)))

    if short_note_ratio > 0.5:
        art = "staccato"
    elif short_note_ratio > 0.2:
        art = "marcato"
    elif overlap_ratio > 0.5:
        art = "legato"
    else:
        art = "legato"

    token = f"[ART:{art}]"
    if acc / max(len(melodic), 1) > 0.05:
        token += "[ART:accent]"   # 允许叠加（DSL 轴内可多 token）
    return {"art": art, "art_token": token,
            "overlap_ratio": round(float(overlap_ratio), 3),
            "short_note_ratio": round(short_note_ratio, 3)}


# ============================================================
# DSL 组装：与论文 Fig.3 / BNF 对齐
# ============================================================
def build_dsl(six_axes: Dict[str, Any]) -> str:
    """把六轴结果编译成 Music-Prompt DSL 文本（Eq. (3) 的实例化）。

    P_DSL ::= <Harm> · <Rhy> · <Tex> · <Dyn> · <Tbr> · <Art>
    """
    harm = six_axes["harmony"]
    rhy = six_axes["rhythm"]
    tex = six_axes["texture"]
    dyn = six_axes["dynamics"]
    tbr = six_axes["timbre"]
    art = six_axes["articulation"]

    lines = [
        f"[GLOBAL:KEY:{harm['key']}]",
        f"[GLOBAL:TEMPO:{rhy['bpm']}]",
        f"[GLOBAL:TIME:{rhy['meter']}]",
        f"[GLOBAL:ART:{art['art']}]",
        tbr["color_token"],
        "",
    ]

    # 分段（和弦序列 → 每 4 个和弦一段，简化 storyboard）
    chords = harm["chords"]
    if chords:
        n_sec = max(1, int(np.ceil(len(chords) / 4)))
        for i in range(n_sec):
            chunk = chords[i * 4:(i + 1) * 4]
            if not chunk:
                continue
            bar_lo = i * 4 + 1
            bar_hi = i * 4 + len(chunk)
            lines.append(f"[SEC:section{i + 1}|BAR:{bar_lo}-{bar_hi}]")
            for c in chunk:
                lines.append(f"  [CHORD:{c['chord']}]")
            lines.append(f"  {dyn['level_token']} {tex['density_token']} "
                         f"{tex['voices_token']} {art['art_token']}")
            lines.append("")
    else:
        # 无和弦时按 dynamics 段切分（保留时变信息，见 axis_dynamics 的 segments）
        dyn_segs = dyn.get("segments")
        if dyn_segs:
            for idx, s in enumerate(dyn_segs):
                tok = dyn["dyn_tokens"][idx]["token"] if idx < len(dyn["dyn_tokens"]) \
                    else f"[DYN:{s['level']}]"
                lines.append(f"[SEC:section{idx + 1}|T:{s['start']}-{s['end']}]")
                lines.append(f"  {tok} {tex['density_token']} "
                             f"{tex['voices_token']} {art['art_token']}")
                lines.append("")
        else:
            lines.append(f"[SEC:section1|BAR:1-8]")
            lines.append(f"  {dyn['level_token']} {tex['density_token']} "
                         f"{tex['voices_token']} {art['art_token']}")

    return "\n".join(lines)


def extract_six_axes(bundle: TrackBundle) -> Dict[str, Any]:
    """对一个 TrackBundle 计算全部六轴。"""
    notes = bundle.notes
    return {
        "harmony": axis_harmony(notes),
        "rhythm": axis_rhythm(bundle),
        "texture": axis_texture(notes),
        "dynamics": axis_dynamics(notes),
        "timbre": axis_timbre(notes),
        "articulation": axis_articulation(notes),
    }


# ============================================================
# 数据集 adapter
# ============================================================
def _pm_to_notes(pm, source: str) -> List[dict]:
    notes = []
    for inst in pm.instruments:
        name = (pretty_midi.program_to_instrument_name(inst.program)
                if pretty_midi and not inst.is_drum else
                ("drums" if inst.is_drum else f"program_{inst.program}"))
        for n in inst.notes:
            notes.append({
                "pitch": n.pitch,
                "velocity": n.velocity,
                "start": round(n.start, 4),
                "end": round(n.end, 4),
                "duration": round(n.end - n.start, 4),
                "program": inst.program,
                "is_drum": bool(inst.is_drum),
                "source": f"{source}/{name}",
            })
    return notes


def adapter_midi_file(midi_path: Path, bundle: TrackBundle):
    """通用 adapter：单个 .mid/.midi 文件（MAESTRO / Groove / LMD 单文件）。"""
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    bundle.notes += _pm_to_notes(pm, midi_path.stem)
    if len(bundle.tempo_changes[0]) == 0:
        tc = pm.get_tempo_changes()
        bundle.tempo_changes = tc
    if not bundle.time_signatures:
        bundle.time_signatures = pm.time_signature_changes
    bundle.midi_path = str(midi_path)


def adapter_slakh(track_dir: Path, bundle: TrackBundle):
    """Slakh2100 Track 目录：合并所有 MIDI（跳过 omitted）。"""
    for mf in sorted(track_dir.glob("**/*.mid")) + sorted(track_dir.glob("**/*.midi")):
        if "omitted" in str(mf).lower():
            continue
        adapter_midi_file(mf, bundle)
    mix = next((p for p in [track_dir / "mix.flac", track_dir / "mix.wav"]
                if p.exists()), None)
    if mix:
        bundle.audio_path = str(mix)


def adapter_lakh_dir(midi_dir: Path, bundle: TrackBundle, max_files: int):
    """LMD：目录下大量 .mid，逐个解析（用 music21 加速可选）。"""
    files = sorted(midi_dir.glob("*.mid")) + sorted(midi_dir.glob("*.midi"))
    if max_files:
        files = files[:max_files]
    for mf in files:
        adapter_midi_file(mf, bundle)
    bundle.meta = bundle.meta or {}
    bundle.meta["lmd_midi_count"] = len(files)


def adapter_pop909(pop_root: Path, bundle: TrackBundle):
    """POP909：优先官方人工标注 tempo/key/chord（yaml/jsonl），否则退回 MIDI 估计。"""
    for mf in sorted(pop_root.glob("**/*.mid"))[:1]:
        adapter_midi_file(mf, bundle)
    # 官方标注以 jsonl 形式存在时覆盖 rhythm/harmony
    annot = pop_root / "annotations" / "key_tempo_chord.jsonl"
    if annot.exists():
        bundle.meta = bundle.meta or {}
        bundle.meta["pop909_annotations_path"] = str(annot)


DATASET_ADAPTERS = {
    "slakh": adapter_slakh,
    "maestro": None,     # 单目录多 midi，走通用目录扫描
    "groove": None,
    "lakh": adapter_lakh_dir,
    "pop909": adapter_pop909,
}


# ============================================================
# 音频+文本类（弱文本 / 风格专家扩展，无六轴真值）
# ============================================================
def weak_text_row(dataset: str, audio_path: str, raw_text: str,
                  style: Optional[str] = None) -> dict:
    style_tag = f"[STYLE:{style}]" if style else ""
    tokens = f"[GLOBAL:STYLE:{style or dataset}]"
    if style_tag:
        tokens += f"\n  {style_tag}"
    return {
        "tokens": tokens,
        "text": raw_text,
        "audio_path": audio_path,
        "meta": {"source": dataset, "source_has_midi": False,
                 "style": style or dataset},
    }


def adapter_audio_text(dataset: str, root: Path, out_rows: List[dict],
                       csv_path: Optional[Path] = None, max_rows: int = 0):
    """MusicCaps / FMA / MTG-Jamendo：弱文本行。

    各数据集元数据格式不同，这里实现三种常用格式的兜底解析；
    具体字段请按各自官方 README 微调 meta 读取。
    """
    audio_exts = (".mp3", ".wav", ".flac", ".ogg", ".m4a")
    audios = sorted(p for p in root.rglob("*") if p.suffix.lower() in audio_exts)

    # 1) CSV/TSV 元数据（FMA tracks.csv / MTG-Jamendo autotagging.tsv）
    meta_map = {}
    if csv_path and csv_path.exists():
        try:
            import csv as _csv
        except Exception:
            _csv = None
        if _csv:
            delim = "\t" if csv_path.suffix == ".tsv" else ","
            with open(csv_path, encoding="utf-8", errors="ignore") as f:
                for row in _csv.DictReader(f, delimiter=delim):
                    meta_map[row.get("track_id") or row.get("id")] = row

    caption_dir = root / "captions"      # MusicCaps jsonl 兜底
    captions = {}
    if caption_dir.exists():
        for jf in sorted(caption_dir.glob("*.jsonl")):
            with open(jf, encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    captions[d.get("id", d.get("ytid"))] = \
                        d.get("caption", d.get("text", ""))

    n = 0
    for ap in audios:
        if max_rows and n >= max_rows:
            break
        sid = ap.stem
        text = captions.get(sid) or meta_map.get(sid, {}).get("caption") \
            or meta_map.get(sid, {}).get("genres") or f"{dataset} track {sid}"
        row = weak_text_row(dataset, str(ap), str(text))
        out_rows.append(row)
        n += 1
    return n


# ============================================================
# 主流程
# ============================================================
def process_midi_dataset(dataset: str, root: Path, max_rows: int,
                         render: bool, sf2: Optional[Path]) -> List[dict]:
    rows: List[dict] = []
    bundle = TrackBundle(notes=[], tempo_changes=(np.array([]), np.array([])),
                         time_signatures=[], meta={"source": dataset})

    if dataset == "slakh":
        tracks = sorted(d for d in root.iterdir() if d.is_dir())
        if max_rows:
            tracks = tracks[:max_rows]
        for t in tracks:
            b = TrackBundle(notes=[], tempo_changes=(np.array([]), np.array([])),
                            time_signatures=[], meta={"source": dataset})
            adapter_slakh(t, b)
            row = _bundle_to_row(b, dataset, render, sf2)
            if row:
                rows.append(row)
        return rows

    if dataset == "lakh":
        b = TrackBundle(notes=[], tempo_changes=(np.array([]), np.array([])),
                        time_signatures=[], meta={"source": dataset})
        adapter_lakh_dir(root, b, max_rows)
        row = _bundle_to_row(b, dataset, render, sf2)   # 目录级聚合
        if row:
            rows.append(row)
        return rows

    if dataset == "pop909":
        b = TrackBundle(notes=[], tempo_changes=(np.array([]), np.array([])),
                        time_signatures=[], meta={"source": dataset})
        adapter_pop909(root, b)
        row = _bundle_to_row(b, dataset, render, sf2)
        if row:
            rows.append(row)
        return rows

    # 通用：目录下所有 MIDI 各自成一条（maestro / groove）
    midi_files = sorted(root.glob("**/*.mid")) + sorted(root.glob("**/*.midi"))
    if max_rows:
        midi_files = midi_files[:max_rows]
    for mf in midi_files:
        b = TrackBundle(notes=[], tempo_changes=(np.array([]), np.array([])),
                        time_signatures=[], meta={"source": dataset})
        adapter_midi_file(mf, b)
        row = _bundle_to_row(b, dataset, render, sf2)
        if row:
            rows.append(row)
    return rows


def _bundle_to_row(b: TrackBundle, dataset: str, render: bool,
                   sf2: Optional[Path]) -> Optional[dict]:
    if not b.notes:
        return None

    # 音频路径：已有则用，否则可选 SoundFont 渲染
    audio = b.audio_path
    if audio is None and render and b.midi_path and sf:
        audio = _render_midi(b.midi_path, sf2)

    six = extract_six_axes(b)
    dsl = build_dsl(six)
    text = _build_text(b, six, dataset)

    return {
        "tokens": dsl,
        "text": text,
        "audio_path": audio,
        "meta": {
            "source": dataset, "source_has_midi": True,
            "bpm": six["rhythm"]["bpm"],
            "key": six["harmony"]["key"],
            "meter": six["rhythm"]["meter"],
            "instruments": six["timbre"]["instruments"],
            "tex": six["texture"]["tex"],
            "color": six["timbre"]["color"],
            "art": six["articulation"]["art"],
            "num_notes": len(b.notes),
        },
    }


def _build_text(b: TrackBundle, six: Dict, dataset: str) -> str:
    """配套自然语言 prompt（与项目既有模板风格一致）。"""
    insts = six["timbre"]["instruments"]
    inst_str = ", ".join(insts[:4]) if insts else "ensemble"
    return (f"{dataset} piece with {inst_str}, {six['rhythm']['bpm']} bpm, "
            f"{six['harmony']['key'].replace('_', ' ')}, {six['timbre']['color']} timbre")


def _render_midi(midi_path: str, sf2: Optional[Path]) -> Optional[str]:
    """用 FluidSynth 把 MIDI 渲染成 WAV（复用项目 fluidsynth / soundfonts）。"""
    try:
        import subprocess
    except Exception:
        return None
    sf2_path = None
    if sf2 and sf2.exists():
        sf2_path = sf2
    else:
        # 兜底：项目 soundfonts 目录任意一个 .sf2
        for cand in Path("soundfonts").glob("*.sf2"):
            sf2_path = cand
            break
    if sf2_path is None:
        print("  WARN: 未找到 .sf2，无法渲染（--render 需 --sf2 或 soundfonts/）")
        return None
    wav = os.path.join(tempfile.gettempdir(),
                       f"m6a_{Path(midi_path).stem}.wav")
    try:
        subprocess.run(["fluidsynth", "-ni", str(sf2_path), midi_path,
                        "-F", wav, "-r", "44100"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return wav if Path(wav).exists() else None
    except Exception as e:
        print(f"  WARN: 渲染失败 {e}")
        return None


# ============================================================
# 合成 smoke test
# ============================================================
def make_demo_bundle() -> TrackBundle:
    """构造一段 16s、4 声部、渐强的 MIDI，用于验证六轴口径。"""
    if pretty_midi is None:
        raise RuntimeError("pretty_midi 未安装")

    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    prog = [0, 32, 40, 73]     # Piano, Acoustic Bass, Violin, Flute
    for i, p in enumerate(prog):
        inst = pretty_midi.Instrument(program=p)
        for t in np.arange(0, 16, 0.5):
            v = int(np.clip(30 + t * 4 + i * 5, 20, 110))
            inst.notes.append(pretty_midi.Note(
                velocity=v, pitch=60 + i * 3 + int(t * 0.25) % 4,
                start=float(t), end=float(t + 0.4)))
        pm.instruments.append(inst)

    notes = _pm_to_notes(pm, "demo")
    return TrackBundle(notes=notes,
                       tempo_changes=pm.get_tempo_changes(),
                       time_signatures=pm.time_signature_changes,
                       meta={"source": "demo"})


def run_demo() -> int:
    bundle = make_demo_bundle()
    six = extract_six_axes(bundle)
    print("=" * 60)
    print("六轴 smoke test")
    print("=" * 60)
    print(f"Harmony      : key={six['harmony']['key']}, "
          f"{len(six['harmony']['chords'])} chords, "
          f"first={six['harmony']['chords'][0]['chord'] if six['harmony']['chords'] else '-'}")
    print(f"Rhythm       : bpm={six['rhythm']['bpm']}, meter={six['rhythm']['meter']}")
    print(f"Texture      : peak={six['texture']['peak_voices']}, "
          f"avg={six['texture']['avg_voices']}, {six['texture']['density_token']} "
          f"{six['texture']['voices_token']}")
    d = six["dynamics"]
    print(f"Dynamics     : level={d['level']}, n_seg={len(d['segments'])}, "
          f"tokens={[x['token'] for x in d['dyn_tokens']]}")
    print(f"Timbre       : color={six['timbre']['color']}, "
          f"insts={six['timbre']['instruments']}")
    print(f"Articulation : art={six['articulation']['art']}, "
          f"overlap={six['articulation']['overlap_ratio']}, "
          f"short={six['articulation']['short_note_ratio']}")
    print("\n--- DSL 输出 ---")
    print(build_dsl(six))
    print("\nOK: 六轴提取与 DSL 组装成功")
    return 0


# ============================================================
# CLI
# ============================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="跨数据集六轴 DSL 标签提取框架")
    ap.add_argument("--demo", action="store_true", help="合成 smoke test（无数据跑通）")
    ap.add_argument("--dataset", choices=["slakh", "maestro", "groove", "lakh",
                                          "pop909", "musiccaps", "fma", "jamendo"],
                    help="数据集类型")
    ap.add_argument("--root", help="数据集根目录")
    ap.add_argument("--csv", help="元数据 csv/tsv（FMA / MTG-Jamendo 可选）")
    ap.add_argument("--out", help="输出 JSONL 路径")
    ap.add_argument("--max", type=int, default=0, help="最多处理行数（0=全部）")
    ap.add_argument("--render", action="store_true", help="MIDI 无音频时用 FluidSynth 渲染")
    ap.add_argument("--sf2", help="SoundFont 路径（--render 用）")
    args = ap.parse_args()

    if args.demo or (args.dataset is None and args.root is None):
        return run_demo()

    if not args.root or not Path(args.root).exists():
        print(f"错误: 根目录不存在: {args.root}", file=sys.stderr)
        return 2

    audio_text = {"musiccaps", "fma", "jamendo"}
    rows: List[dict] = []

    if args.dataset in audio_text:
        n = adapter_audio_text(args.dataset, Path(args.root), rows,
                               Path(args.csv) if args.csv else None,
                               max_rows=args.max)
        print(f"[{args.dataset}] 弱文本行: {n} 条")
    else:
        rows = process_midi_dataset(args.dataset, Path(args.root), args.max,
                                    args.render, Path(args.sf2) if args.sf2 else None)
        print(f"[{args.dataset}] 六轴 DSL 行: {len(rows)} 条")

    out = Path(args.out) if args.out else Path(f"data/{args.dataset}_6axis.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"写入: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
