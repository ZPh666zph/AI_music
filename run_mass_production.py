#!/usr/bin/env python3
"""
run_mass_production.py — 226 首国风 MIDI 大规模量产
====================================================
步骤: 清洗文件名 → 批量渲染 → 容错记录 → tqdm 进度条
输出: C:/Deepseek/outputs/guofeng_data/ 下每首一个目录
"""

import sys, os, re, json, time, shutil, subprocess
from pathlib import Path
from collections import defaultdict

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

try:
    import pretty_midi
    import numpy as np
    from tqdm import tqdm
except ImportError:
    print("请安装: pip install pretty_midi numpy tqdm")
    sys.exit(1)

# ── 硬路径 ──
MIDI_DIR   = Path("C:/Deepseek/data/raw_midis")
OUT_DIR    = Path("C:/Deepseek/outputs/guofeng_data")
TEMP_DIR   = Path("C:/Deepseek/data/guofeng_temp")
SF2_PATH   = Path("C:/Deepseek/soundfonts/guzheng.sf2")
FLUIDSYNTH = Path("C:/Deepseek/fluidsynth/bin/fluidsynth.exe")
BAD_LOG    = OUT_DIR / "bad_files.log"

OUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════
# 第 1 步: 文件名清洗
# ═══════════════════════════════════════════════════════════

def clean_filename(name: str) -> str:
    """去掉杂质，只保留干净的歌曲原名"""
    original = name

    # 1. 去掉方括号前缀 [xxx]、【xxx】
    name = re.sub(r'^[\[【].+?[\]】]\s*', '', name)

    # 2. 去掉 " - 歌手名" 后缀 (含括号内别名)
    #    匹配 " - 闻人听書_"  " - 邓寓君(等什么君)" 等
    name = re.sub(r'\s*[-–—]\s*.+$', '', name)

    # 3. 去掉 _(Vocals) 后缀
    name = re.sub(r'_?\(Vocals\)', '', name)

    # 4. 去掉前缀标记: 中国风_、古风_
    name = re.sub(r'^(中国风|古风|国风)_', '', name)

    # 5. 去掉 _自截曲、自截曲_
    name = re.sub(r'_?自截曲_?', '', name)

    # 6. 去掉 _少司命、_闻人听書_ 等残留
    name = re.sub(r'_[^\s.]{1,6}$', '', name)

    # 7. 去掉连续下划线
    name = re.sub(r'_{2,}', '_', name)

    # 8. 去掉首尾下划线/空格
    name = name.strip('_ ')

    if name != original:
        print(f"  {original[:50]:<50} → {name}.mid")
    return name


# ═══════════════════════════════════════════════════════════
# 第 2 步: 核心渲染 (从 guofeng_data_factory 精简)
# ═══════════════════════════════════════════════════════════

def make_monophonic(pm: pretty_midi.PrettyMIDI) -> int:
    """二胡单音化: 重叠音符仅保留最高音，返回移除的音符数"""
    removed = 0
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        notes = sorted(inst.notes, key=lambda n: n.start)
        to_delete = []
        active = []
        for note in notes:
            active = [n for n in active if n.end > note.start]
            if active:
                all_notes = active + [note]
                top = max(all_notes, key=lambda n: n.pitch)
                for n in all_notes:
                    if n is not top:
                        to_delete.append(n)
            active.append(note)
        for n in set(to_delete):
            if n in inst.notes:
                inst.notes.remove(n)
                removed += 1
    return removed


def render_one(midi_path: Path, track_name: str) -> dict:
    """渲染单首 MIDI → WAV，返回 meta"""
    track_out = OUT_DIR / track_name
    track_out.mkdir(parents=True, exist_ok=True)

    # 1. 加载 MIDI
    pm = pretty_midi.PrettyMIDI(str(midi_path))

    # 2. 元数据
    tempo = pm.get_tempo_changes()[1][0] if len(pm.get_tempo_changes()[0]) > 0 else 120
    total_notes = sum(len(i.notes) for i in pm.instruments if not i.is_drum)

    # 3. 二胡单音化
    removed = make_monophonic(pm)
    filtered_notes = total_notes - removed

    # 4. 保存临时英文 MIDI
    temp_midi = TEMP_DIR / f"temp_{track_name[:20]}.mid"
    pm.write(str(temp_midi))

    # 5. FluidSynth 渲染
    temp_wav = TEMP_DIR / f"temp_{track_name[:20]}.wav"
    cmd = [
        str(FLUIDSYNTH), "-ni", "-r", "16000", "-g", "2.0",
        "-F", str(temp_wav), str(SF2_PATH), str(temp_midi),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            encoding="utf-8", errors="ignore", timeout=300)
    success = result.returncode == 0 and temp_wav.exists()

    # 6. 复制结果
    mix_wav = track_out / "mix.wav"
    if success:
        shutil.copy2(str(temp_wav), str(mix_wav))

    # 7. 清理临时文件
    temp_midi.unlink(missing_ok=True)
    temp_wav.unlink(missing_ok=True)

    # 8. 写入 meta.json
    purity = filtered_notes / max(total_notes, 1)
    meta = {
        "track_name": track_name,
        "midi_file": str(midi_path),
        "tempo": float(round(tempo, 1)),
        "total_notes": total_notes,
        "filtered_notes": filtered_notes,
        "erhu_purity": float(round(purity, 3)),
        "render_success": success,
        "sf2_used": str(SF2_PATH),
        "guofeng_style": "erhu_solo",
    }
    with open(track_out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta


# ═══════════════════════════════════════════════════════════
# 第 3 步: 主流程 — 高容错批量
# ═══════════════════════════════════════════════════════════

def main():
    import subprocess

    print("=" * 60)
    print("国风 MIDI 大规模量产")
    print(f"MIDI 目录: {MIDI_DIR}")
    print(f"输出目录: {OUT_DIR}")
    print(f"SF2 音源: {SF2_PATH}  {'存在' if SF2_PATH.exists() else '不存在!'}")
    print("=" * 60)

    # ── 前置清洗: 重命名 ──
    print("\n[1/3] 文件名清洗...")
    midi_files = sorted(MIDI_DIR.glob("*.mid")) + sorted(MIDI_DIR.glob("*.midi"))
    renamed = 0
    for f in midi_files:
        clean = clean_filename(f.stem)
        if clean != f.stem:
            new_path = f.with_stem(clean)
            if not new_path.exists():
                f.rename(new_path)
                renamed += 1
    print(f"  重命名: {renamed} 首")
    print(f"  总计:   {len(midi_files)} 首")

    # ── 重新扫描 ──
    midi_files = sorted(MIDI_DIR.glob("*.mid")) + sorted(MIDI_DIR.glob("*.midi"))
    total = len(midi_files)

    # ── 批量渲染 ──
    print(f"\n[2/3] 批量渲染 ({total} 首)...")
    results = {"ok": 0, "bad_midi": 0, "silent": 0}
    bad_list = []

    pbar = tqdm(midi_files, desc="二胡渲染", unit="首")
    for f in pbar:
        track_name = f.stem
        pbar.set_postfix_str(track_name[:18])

        try:
            meta = render_one(f, track_name)

            if meta["render_success"]:
                # 检查是否静音
                import soundfile as sf
                audio, sr = sf.read(str(OUT_DIR / track_name / "mix.wav"))
                rms = float(np.sqrt(np.mean(audio**2)))
                if rms < 0.001:
                    results["silent"] += 1
                    meta["warning"] = "silent (RMS < 0.001)"
                    with open(OUT_DIR / track_name / "meta.json", "w", encoding="utf-8") as jf:
                        json.dump(meta, jf, ensure_ascii=False, indent=2)
                else:
                    results["ok"] += 1
            else:
                results["bad_midi"] += 1
                bad_list.append(f"{track_name}: FluidSynth 渲染失败")

        except Exception as e:
            results["bad_midi"] += 1
            bad_list.append(f"{track_name}: {type(e).__name__}: {e}")
            tqdm.write(f"  ✗ {track_name}: {type(e).__name__}")

    # ── 写入坏文件日志 ──
    print(f"\n[3/3] 写入日志...")
    with open(BAD_LOG, "w", encoding="utf-8") as f:
        f.write(f"# 坏文件日志 — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# 总数: {total}, 成功: {results['ok']}, 失败: {results['bad_midi']}, 静音: {results['silent']}\n\n")
        for line in bad_list:
            f.write(line + "\n")

    # ── 汇总 ──
    print(f"\n{'=' * 60}")
    print(f"量产完成!")
    print(f"  成功:     {results['ok']}")
    print(f"  静音:     {results['silent']} (已标记)")
    print(f"  损坏:     {results['bad_midi']}")
    print(f"  坏文件日志: {BAD_LOG}")
    print(f"  输出目录:   {OUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
