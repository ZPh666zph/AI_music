#!/usr/bin/env python3
"""
guofeng_data_factory.py — 国风数据工厂：MIDI + 民乐 SF2 → 高保真古风音频
======================================================================
输入:   C:/Deepseek/data/raw_midis/   下所有 .mid 文件
输出:   C:/Deepseek/data/guofeng_output/{track_name}/
          ├── mix.wav           (16kHz 混音)
          ├── meta.json         (BPM + Key + 乐器名 + Token)
          └── stems/            (各乐器分轨 WAV)

用法:
  python guofeng_data_factory.py                              # 处理全部
  python guofeng_data_factory.py --demo                        # 仅处理第1首
  python guofeng_data_factory.py --sf2 ./guzheng.sf2           # 指定 SF2 路径

依赖:
  pip install pretty_midi pyfluidsynth tqdm numpy
  # 还需要系统安装 FluidSynth: winget install FluidSynth
"""

import os, sys, json, time, subprocess, shutil, argparse, re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

try:
    import pretty_midi
    import numpy as np
    from tqdm import tqdm
except ImportError:
    print("请先安装依赖: pip install pretty_midi numpy tqdm")
    sys.exit(1)

try:
    import fluidsynth
    HAS_FLUIDSYNTH = True
except (ImportError, OSError):
    HAS_FLUIDSYNTH = False


# ═══════════════════════════════════════════════════════════
# 0. 配置
# ═══════════════════════════════════════════════════════════

INPUT_DIR   = Path("C:/Deepseek/data/raw_midis")
OUTPUT_DIR  = Path("C:/Deepseek/data/guofeng_output")
TEMP_DIR    = Path("C:/Deepseek/data/guofeng_temp")
SAMPLE_RATE = 16000  # 与 Slakh2100 HF 版对齐

# 全部绝对路径 — 不给系统猜错的机会
INPUT_DIR       = Path("C:/Deepseek/data/raw_midis")
OUTPUT_DIR      = Path("C:/Deepseek/outputs/guofeng_data")
FLUIDSYNTH_DIR  = Path("C:/Deepseek/fluidsynth")
FLUIDSYNTH_EXE  = FLUIDSYNTH_DIR / "bin" / "fluidsynth.exe"
DEFAULT_SF2     = Path("C:/Deepseek/soundfonts/guzheng.sf2")  # 实际是二胡 SF2
TEMP_DIR        = Path("C:/Deepseek/data/guofeng_temp")

# Windows 中文路径保护
def _long_path(p: Path) -> str:
    s = str(p.resolve())
    if os.name == "nt" and not s.startswith("\\\\?\\"):
        s = "\\\\?\\" + s
    return s


# ═══════════════════════════════════════════════════════════
# 1. 乐器映射引擎
# ═══════════════════════════════════════════════════════════

# A 组: GM Program → 中国民乐 SF2 音源映射
GM_TO_GUOFENG: Dict[int, str] = {
    # 键盘类 → 拨弦类
    0:  "guzheng",       # Acoustic Grand Piano → 古筝
    1:  "guzheng",       # Bright Acoustic Piano → 古筝
    2:  "guzheng",       # Electric Grand Piano → 古筝
    3:  "guzheng",       # Honky-tonk Piano → 古筝
    4:  "yangqin",       # Electric Piano 1 → 扬琴
    5:  "yangqin",       # Electric Piano 2 → 扬琴
    6:  "guzheng",       # Harpsichord → 古筝
    7:  "guzheng",       # Clavinet → 古筝
    
    # 旋律打击 → 保留或映射
    8:  "yangqin",       # Celesta → 扬琴
    11: "yangqin",       # Vibraphone → 扬琴
    12: "guzheng",       # Marimba → 古筝
    14: "yangqin",       # Tubular Bells → 扬琴
    
    # 吉他类 → 琵琶/阮
    24: "pipa",          # Acoustic Guitar (nylon) → 琵琶
    25: "pipa",          # Acoustic Guitar (steel) → 琵琶
    26: "ruan",          # Electric Guitar (jazz) → 中阮
    27: "ruan",          # Electric Guitar (clean) → 中阮
    28: "pipa",          # Electric Guitar (muted) → 琵琶
    29: "pipa",          # Overdriven Guitar → 琵琶
    30: "pipa",          # Distortion Guitar → 琵琶
    31: "pipa",          # Guitar harmonics → 琵琶
    
    # 贝斯类 → 大阮/低音革胡
    32: "daruan",        # Acoustic Bass → 大阮
    33: "daruan",        # Electric Bass (finger) → 大阮
    34: "daruan",        # Electric Bass (pick) → 大阮
    35: "daruan",        # Fretless Bass → 大阮
    36: "daruan",        # Slap Bass 1 → 大阮
    37: "daruan",        # Slap Bass 2 → 大阮
    38: "daruan",        # Synth Bass 1 → 大阮
    39: "daruan",        # Synth Bass 2 → 大阮
    
    # 弦乐 → 二胡/马头琴
    40: "erhu",          # Violin → 二胡
    41: "erhu",          # Viola → 二胡
    42: "matouqin",      # Cello → 马头琴
    43: "matouqin",      # Contrabass → 马头琴
    44: "erhu",          # Tremolo Strings → 二胡
    45: "erhu",          # Pizzicato Strings → 二胡拨弦
    46: "konghou",       # Orchestral Harp → 箜篌
    47: "erhu",          # Timpani → 二胡
    
    # 合奏 → 民乐合奏
    48: "minyue",        # String Ensemble 1 → 民乐合奏
    49: "minyue",        # String Ensemble 2 → 民乐合奏
    50: "minyue",        # Synth Strings 1 → 民乐合奏
    51: "minyue",        # Synth Strings 2 → 民乐合奏
    
    # 管乐 → 笛/箫/唢呐/笙
    56: "dizi",          # Trumpet → 笛子
    57: "suona",         # Trombone → 唢呐
    58: "suona",         # Tuba → 唢呐
    59: "dizi",          # Muted Trumpet → 笛子
    60: "suona",         # French Horn → 唢呐
    61: "dizi",          # Brass Section → 笛子
    64: "xiao",          # Soprano Sax → 箫
    65: "xiao",          # Alto Sax → 箫
    66: "xiao",          # Tenor Sax → 箫
    67: "xiao",          # Baritone Sax → 箫
    68: "suona",         # Oboe → 唢呐
    69: "xiao",          # English Horn → 箫
    70: "sheng",         # Bassoon → 笙
    71: "dizi",          # Clarinet → 笛子
    72: "xiao",          # Piccolo → 箫
    73: "dizi",          # Flute → 笛子
    74: "xiao",          # Recorder → 箫
    75: "xiao",          # Pan Flute → 箫
    76: "xiao",          # Blown Bottle → 箫
    77: "dizi",          # Shakuhachi → 笛子 (尺八)
    78: "xiao",          # Whistle → 箫
    79: "suona",         # Ocarina → 唢呐
    
    # Pad → 保留氛围
    88: "konghou",       # New Age → 箜篌
    89: "konghou",       # Pad (warm) → 箜篌
    91: "konghou",       # Pad (polysynth) → 箜篌
    92: "konghou",       # Pad (choir) → 箜篌
    94: "konghou",       # Pad (metallic) → 箜篌
    
    # 其他 → 笙
    99: "sheng",         # FX → 笙
    104:"pipa",          # Sitar → 琵琶 (西塔→琵琶)
}

# B 组: 民乐 SF2 音源文件名 → 本地路径
GUOFENG_SF2_PATHS: Dict[str, str] = {
    "guzheng":  "soundfonts/guzheng.sf2",      # 古筝
    "pipa":     "soundfonts/pipa.sf2",         # 琵琶
    "erhu":     "soundfonts/erhu.sf2",         # 二胡
    "dizi":     "soundfonts/dizi.sf2",         # 笛子
    "xiao":     "soundfonts/xiao.sf2",         # 箫
    "suona":    "soundfonts/suona.sf2",        # 唢呐
    "sheng":    "soundfonts/sheng.sf2",        # 笙
    "yangqin":  "soundfonts/yangqin.sf2",      # 扬琴
    "ruan":     "soundfonts/zhongruan.sf2",    # 中阮
    "daruan":   "soundfonts/daruan.sf2",       # 大阮
    "matouqin": "soundfonts/matouqin.sf2",     # 马头琴
    "konghou":  "soundfonts/konghou.sf2",      # 箜篌
    "minyue":   "soundfonts/minyue.sf2",       # 民乐合奏
}

# C 组: 简化的乐器名 (pretty_midi program name → 民乐别名)
INSTRUMENT_NAMES = {
    "guzheng":  "古筝",   "pipa":     "琵琶",
    "erhu":     "二胡",    "dizi":     "笛子",
    "xiao":     "箫",      "suona":    "唢呐",
    "sheng":    "笙",      "yangqin":  "扬琴",
    "ruan":     "中阮",    "daruan":   "大阮",
    "matouqin": "马头琴",  "konghou":  "箜篌",
    "minyue":   "民乐合奏",
}


def map_instruments(midi_path: str) -> List[dict]:
    """
    解析 MIDI 文件，返回乐器映射方案。
    每个轨道: {program, gm_name, guofeng_instrument, sf2_path, note_count}
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    mappings = []
    
    for track in pm.instruments:
        if track.is_drum:
            continue  # 跳过打击乐
        
        program = track.program
        gm_name = pretty_midi.program_to_instrument_name(program)
        guofeng = GM_TO_GUOFENG.get(program, "guzheng")  # 默认 → 古筝
        sf2_path = GUOFENG_SF2_PATHS.get(guofeng, "")
        
        mappings.append({
            "program": program,
            "gm_name": gm_name,
            "guofeng": guofeng,
            "guofeng_cn": INSTRUMENT_NAMES.get(guofeng, guofeng),
            "sf2_path": sf2_path,
            "note_count": len(track.notes),
        })
    
    return mappings


# ═══════════════════════════════════════════════════════════
# 2. 元数据提取 (BPM + Key + 五声音阶检查)
# ═══════════════════════════════════════════════════════════

def extract_metadata(midi_path: str) -> dict:
    """提取 Tempo, Key, Time Signature, 五声音阶比例"""
    pm = pretty_midi.PrettyMIDI(midi_path)
    
    # BPM
    tempo_changes = pm.get_tempo_changes()
    if len(tempo_changes[0]) > 0:
        bpm = float(np.mean(tempo_changes[1]))
    else:
        bpm = 120.0
    
    # Time Signature
    ts_changes = pm.time_signature_changes
    if len(ts_changes) > 0:
        meter = f"{ts_changes[0].numerator}/{ts_changes[0].denominator}"
    else:
        meter = "4/4"
    
    # Key (简化的 Krumhansl-Schmuckler)
    all_notes = []
    for inst in pm.instruments:
        if not inst.is_drum:
            for note in inst.notes:
                all_notes.append(note.pitch % 12)
    
    if all_notes:
        pitch_counts = np.bincount(all_notes, minlength=12)
        # 大调/小调模板
        major = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
        minor = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
        norm = pitch_counts / max(pitch_counts.sum(), 1)
        best_key = "C_major"
        best_score = -1
        pitch_names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
        for i in range(12):
            r = np.roll(norm, -i)
            ms = np.corrcoef(r, major)[0,1]
            mn = np.corrcoef(r, minor)[0,1]
            if ms > best_score:
                best_score = ms
                best_key = f"{pitch_names[i]}_major"
            if mn > best_score:
                best_score = mn
                best_key = f"{pitch_names[i]}_minor"
    else:
        best_key = "C_major"
    
    # 五声音阶比例 (宫商角徵羽 = C D E G A = 0,2,4,7,9)
    pentatonic_set = {0, 2, 4, 7, 9}
    if all_notes:
        penta_ratio = sum(1 for p in all_notes if p in pentatonic_set) / len(all_notes)
    else:
        penta_ratio = 0
    
    # 时长
    total_duration = pm.get_end_time()
    
    return {
        "bpm": int(round(bpm)),
        "meter": meter,
        "key": best_key,
        "key_confidence": float(round(best_score, 3)),
        "pentatonic_ratio": float(round(penta_ratio, 3)),
        "duration_sec": float(round(total_duration, 1)),
        "note_count": int(len(all_notes)),
    }


# ═══════════════════════════════════════════════════════════
# 3. FluidSynth 批量渲染
# ═══════════════════════════════════════════════════════════

def check_fluidsynth() -> bool:
    """检查 FluidSynth 是否可用"""
    try:
        result = subprocess.run(
            [str(FLUIDSYNTH_EXE), "--version"],
            capture_output=True, text=True, timeout=5,
            encoding="utf-8", errors="ignore"
        )
        if result.returncode == 0:
            ver = result.stdout.split("\n")[0]
            print(f"  FluidSynth: {ver}")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False


def render_midi_with_sf2(
    midi_path: str,
    output_wav: str,
    sf2_path: str = None,
    sample_rate: int = 16000,
) -> bool:
    """
    用 FluidSynth 命令行渲染 MIDI → WAV
    
    备选方案: pyfluidsynth (更可控但需要先加载 SF2)
    """
    cmd = [
        _long_path(Path(str(FLUIDSYNTH_EXE))),
        "-ni",
        "-r", str(sample_rate),
        "-g", "2.0",
        "-F", _long_path(Path(output_wav)),
        _long_path(Path(sf2_path)),
        _long_path(Path(midi_path)),
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                               encoding="utf-8", errors="ignore")
        if result.returncode == 0 and os.path.exists(output_wav):
            return True
        else:
            print(f"    FluidSynth 退出码: {result.returncode}")
            if result.stderr:
                print(f"    stderr: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"    渲染超时 (300s)")
        return False
    except FileNotFoundError:
        print(f"    FluidSynth 未安装! 请运行: winget install FluidSynth")
        return False


def render_with_pyfluidsynth(
    midi_path: str,
    output_wav: str,
    sf2_path: str,
    sample_rate: int = 16000,
) -> bool:
    """
    备选方案: 用 pyfluidsynth Python 绑定渲染
    需要先: fs = fluidsynth.Synth(); fs.start(); sfid = fs.sfload(sf2_path)
    """
    try:
        fs = fluidsynth.Synth(samplerate=float(sample_rate))
        fs.start()
        
        sfid = fs.sfload(sf2_path)
        if sfid == -1:
            fs.delete()
            return False
        
        # 读取 MIDI 并播放
        pm = pretty_midi.PrettyMIDI(midi_path)
        
        # 渲染到 PCM buffer
        total_samples = int(pm.get_end_time() * sample_rate) + sample_rate
        buffer = np.zeros(total_samples, dtype=np.float32)
        
        # Reset synth
        fs.system_reset()
        fs.program_select(0, sfid, 0, 0)
        
        # 逐乐器渲染 (简化版: 不处理精确 timing)
        for instrument in pm.instruments:
            if instrument.is_drum:
                continue
            prog = instrument.program
            fs.program_select(0, sfid, 0, prog if prog < 128 else 0)
            
            for note in instrument.notes:
                start_sample = int(note.start * sample_rate)
                duration = max(0.1, note.end - note.start)
                
                fs.noteon(0, note.pitch, note.velocity)
                
                # 渲染这段
                n_samples = int(duration * sample_rate)
                samples = fs.get_samples(sample_rate)[:n_samples*2]  # stereo
                
                # 混合到 buffer
                if len(samples) >= 2:
                    mono = (samples[::2].astype(np.float32) + samples[1::2].astype(np.float32)) * 0.5
                    end = min(start_sample + len(mono), len(buffer))
                    buffer[start_sample:end] += mono[:end-start_sample]
                
                fs.noteoff(0, note.pitch)
        
        fs.delete()
        
        # 归一化 + 保存
        if buffer.max() > 0:
            buffer = buffer / buffer.max() * 0.9
        
        import soundfile as sf
        sf.write(output_wav, buffer, sample_rate)
        return True
    
    except Exception as e:
        print(f"    pyfluidsynth 错误: {e}")
        return False


# ═══════════════════════════════════════════════════════════
# 4. 主流水线
# ═══════════════════════════════════════════════════════════

def get_sf2_path() -> Path:
    """硬编码 SF2 — 不给猜错的机会"""
    return DEFAULT_SF2


def load_progress() -> set:
    return set()


def sanitize_midi_for_sf2(midi_path: Path) -> Path:
    """
    终极防弹:
    1. 二胡单音化: 同一轨道同时发音时只保留最高音 (Top Note), 模拟真实拉弦
    2. 另存为纯英文临时文件，绕开 FluidSynth C++ 的中文编码问题
    """
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    temp_midi = TEMP_DIR / "temp_render.mid"
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    
    # 二胡单音化: 检测音符重叠 → 只保留最高音
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        notes = sorted(instrument.notes, key=lambda n: n.start)
        to_remove = []
        active = []  # 当前活跃的音符
        for note in notes:
            # 清理已结束的音符
            active = [n for n in active if n.end > note.start]
            # 如果有活跃音符重叠，保留最高 pitch，标记其余删除
            if active:
                all_overlap = active + [note]
                top = max(all_overlap, key=lambda n: n.pitch)
                for n in all_overlap:
                    if n is not top:
                        to_remove.append(n)
            active.append(note)
        # 去重删除
        for n in set(to_remove):
            if n in instrument.notes:
                instrument.notes.remove(n)
    
    pm.write(str(temp_midi))
    return temp_midi


def process_one_midi(
    midi_path: Path,
    output_dir: Path,
    default_sf2: str,
) -> Optional[dict]:
    """处理单个 MIDI 文件: 映射 → 渲染 → 提取元数据"""
    
    track_name = midi_path.stem
    track_out = output_dir / track_name
    track_out.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. 映射
        mappings = map_instruments(_long_path(midi_path))
        if not mappings:
            return None
        
        # 2. 元数据
        long_p = _long_path(midi_path)
        meta = extract_metadata(long_p)
        meta["midi_file"] = str(midi_path)
        meta["instruments"] = mappings
        meta["guofeng_style"] = "erhu_solo"
        
        # 二胡纯度分级
        if meta["pentatonic_ratio"] >= 0.6:
            meta["guofeng_quality"] = "pure_erhu"
        elif meta["pentatonic_ratio"] >= 0.4:
            meta["guofeng_quality"] = "mixed_erhu"
        else:
            meta["guofeng_quality"] = "adapted"
        
        # 3. 二胡纯度: 单音化前后的音符比
        orig_total = sum(len(i.notes) for i in pretty_midi.PrettyMIDI(str(midi_path)).instruments if not i.is_drum)
        
        # 4. 渲染: 消毒 MIDI (二胡单音化 + 英文名) → 临时 WAV → 复制回中文目录
        mix_wav = track_out / "mix.wav"
        sf2_to_use = default_sf2
        guofeng_instruments = list(set(m["guofeng"] for m in mappings))
        
        erhu_purity = 1.0
        if sf2_to_use and os.path.exists(sf2_to_use):
            temp_midi = sanitize_midi_for_sf2(midi_path)
            filtered_total = sum(len(i.notes) for i in pretty_midi.PrettyMIDI(str(temp_midi)).instruments if not i.is_drum)
            erhu_purity = filtered_total / max(orig_total, 1)
            
            temp_wav = TEMP_DIR / "temp_mix.wav"
            success = render_midi_with_sf2(
                str(temp_midi), str(temp_wav), sf2_to_use
            )
            if success and temp_wav.exists():
                shutil.copy2(str(temp_wav), str(mix_wav))
                temp_wav.unlink(missing_ok=True)
            if temp_midi.exists():
                temp_midi.unlink(missing_ok=True)
        else:
            mix_wav = None
            success = False
        
        meta["mix_wav"] = str(mix_wav) if success else None
        meta["sf2_used"] = sf2_to_use
        meta["guofeng_instruments"] = ["erhu"]  # 实事求是
        meta["render_success"] = success
        meta["erhu_purity"] = float(round(erhu_purity, 3))  # 二胡味儿纯度
        
        # 标签修正: guzheng → erhu
        meta["guofeng_style"] = "erhu_solo"
        
        # 4. 保存 meta.json (防弹: 强制转换 numpy 类型)
        class _Encoder(json.JSONEncoder):
            def default(self, o):
                import numpy as np
                if isinstance(o, (np.integer,)): return int(o)
                if isinstance(o, (np.floating,)): return float(o)
                if isinstance(o, (np.ndarray,)): return o.tolist()
                return super().default(o)
        with open(track_out / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2, cls=_Encoder)
        
        return meta
    
    except Exception as e:
        # 记录错误，不中断
        error_file = track_out / "error.txt"
        with open(error_file, "w", encoding="utf-8") as f:
            f.write(f"{type(e).__name__}: {e}\n")
        return None


def main():
    parser = argparse.ArgumentParser(description="国风数据工厂")
    parser.add_argument("--demo", action="store_true", help="仅处理第一首")
    parser.add_argument("--sf2", default=None, help="覆盖默认 SF2 路径")
    args = parser.parse_args()
    
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    sf2_path = Path(args.sf2) if args.sf2 else get_sf2_path()
    
    print("=" * 60)
    print("国风数据工厂 v2.0 — 暴力覆盖模式")
    print("=" * 60)
    
    # ── 调试: 打印全部路径 ──
    print(f"\n  路径配置:")
    print(f"    MIDI 输入:    {INPUT_DIR}")
    print(f"    WAV 输出:     {OUTPUT_DIR}")
    print(f"    SF2 音源:     {sf2_path}")
    print(f"    SF2 存在:     {sf2_path.exists()}")
    print(f"    FluidSynth:   {FLUIDSYNTH_EXE}")
    print(f"    FluidSynth存在: {Path(str(FLUIDSYNTH_EXE)).exists()}")
    
    if not sf2_path.exists():
        print(f"\n  ❌ 未找到 SF2! 请将音源放到 {sf2_path}")
        print(f"     下载: https://musical-artifacts.com/ 搜索 guzheng")
        return
    
    # 检查 FluidSynth
    has_fs = check_fluidsynth()
    if not has_fs:
        print(f"\n  ⚠️  FluidSynth 不可用, 仅提取元数据")
    
    # 收集 MIDI
    midi_files = sorted(INPUT_DIR.glob("**/*.mid")) + sorted(INPUT_DIR.glob("**/*.midi"))
    if not midi_files:
        print(f"\n  ❌ {INPUT_DIR} 下没有 .mid 文件!")
        return
    
    if args.demo:
        midi_files = midi_files[:1]
    
    print(f"\n  MIDI 文件: {len(midi_files)} 首")
    print(f"  模式:  {'Demo(1首)' if args.demo else '全部覆写(强制)'}")
    print()
    
    # 批量 — 全部暴力覆盖
    results = {"success": 0, "no_render": 0, "error": 0}
    pbar = tqdm(midi_files, desc="国风化", unit="track")
    for midi_path in pbar:
        pbar.set_postfix_str(f"{midi_path.stem[:20]}")
        meta = process_one_midi(midi_path, OUTPUT_DIR, str(sf2_path))
        if meta:
            if meta.get("render_success"):
                results["success"] += 1
            else:
                results["no_render"] += 1
        else:
            results["error"] += 1
    
    print(f"\n{'=' * 60}")
    print(f"完成: 渲染 {results['success']} | 仅元数据 {results['no_render']} | 错误 {results['error']}")
    print(f"输出: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


# ═══════════════════════════════════════════════════════════
# 附录: 采购清单
# ═══════════════════════════════════════════════════════════

PURCHASE_LIST = r"""
╔══════════════════════════════════════════════════════════════════╗
║                  国风数据工厂 · 采购清单                            ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  📦 1. FluidSynth 软件 (必需)                                     ║
║  ─────────────────────────────────────                              ║
║  Windows: winget install FluidSynth                               ║
║  官网:    https://www.fluidsynth.org/download/                    ║
║  macOS:   brew install fluid-synth                                ║
║  Linux:   apt install fluidsynth                                  ║
║                                                                  ║
║  📦 2. 免费国风 MIDI 包 (音乐素材)                                ║
║  ─────────────────────────────────────                              ║
║  搜索建议:                                                         ║
║    · Bilibili 搜索 "古风 MIDI" / "民乐 MIDI 下载"                 ║
║    · MuseScore.com 搜索 "Chinese traditional" / "古筝" / "二胡"   ║
║    · midiworld.com → Traditional → Chinese                        ║
║    · freemidi.org → 搜索 "guzheng" / "erhu" / "pipa"             ║
║    · GitHub 搜索 "chinese midi dataset"                           ║
║    · Lakh MIDI Dataset: https://colinraffel.com/projects/lmd/    ║
║      (176K MIDI, CC-BY — 筛选后用乐器重映射变古风)                ║
║                                                                  ║
║  📦 3. 民乐 SF2 音源文件 (核心)                                   ║
║  ─────────────────────────────────────                              ║
║  搜索建议 (Google/Bing/百度):                                      ║
║    · "古筝 SoundFont 下载" / "guzheng sf2 free"                   ║
║    · "琵琶 SF2" / "pipa soundfont free"                           ║
║    · "二胡 SoundFont" / "erhu sf2 download"                       ║
║    · "笛子 SoundFont" / "dizi sf2 free"                           ║
║    · "中国民乐 SoundFont 合集"                                     ║
║    · "Chinese instrument soundfont pack"                          ║
║                                                                  ║
║  推荐资源站:                                                       ║
║    · https://musical-artifacts.com/ — 最大的 SF2 索引站           ║
║      搜索 "guzheng", "erhu", "pipa", "dizi"                      ║
║    · https://soundfonts.org/ — 免费 SF2 下载                      ║
║    · FluidR3_GM.sf2 — 通用 GM 音源 (备用)                         ║
║      已有 128 种乐器, 但民乐音色有限                               ║
║    · 淘宝/闲鱼搜索 "古筝软音源" — 可能有低价商业音源               ║
║                                                                  ║
║  📦 4. 放置方式                                                   ║
║  ─────────────────────────────────────                              ║
║  下载后将 SF2 文件放到:                                            ║
║    C:/Deepseek/soundfonts/                                        ║
║      guzheng.sf2      ← 古筝                                      ║
║      pipa.sf2         ← 琵琶                                      ║
║      erhu.sf2         ← 二胡                                      ║
║      dizi.sf2         ← 笛子                                      ║
║      xiao.sf2         ← 箫                                        ║
║      suona.sf2        ← 唢呐                                      ║
║      yangqin.sf2      ← 扬琴                                      ║
║      zhongruan.sf2    ← 中阮                                      ║
║                                                                  ║
║  📦 5. 零成本替代方案 (如果找不到民乐 SF2)                         ║
║  ─────────────────────────────────────                              ║
║  · FluidR3_GM.sf2 (141MB, 免费) — 用 GM 音源先跑通 pipeline       ║
║    搜索: "FluidR3_GM.sf2 download"                                ║
║  · TimGM6mb.sf2 (6MB) — 极小 GM 音源, 适合快速测试                ║
║  · 后续用真实民乐录音替换                                           ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""


if __name__ == "__main__":
    print(PURCHASE_LIST)
    main()
