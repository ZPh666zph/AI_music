# -*- coding: utf-8 -*-
"""
midi_to_musicongen_conditions.py — 从国风 MIDI 提取 MusiConGen 的 (chords, bpm, meter)。

依赖（musicongen 环境）：
    pip install mido music21

输入：
    MIDI 文件目录：默认 C:\\Deepseek\\data\\raw_midis
    或一个 JSONL：每行 {"id": 序号, "midi_path": "..."}（与 baseline_prompts 的 id 对齐）

输出：
    musicongen_conditions.jsonl —— 每行 {"id": n, "text_chords": "C G A:min F", "bpm": 120, "meter": 4}

MusiConGen 和弦格式：
    空格分隔 = 每小节一个和弦；C G A:min F
    同一小节多和弦用逗号：C G,G:7 E:min
    性质用 :maj/:min/:dim/:7/:maj7 等，纯大写字母默认大调。
"""
import argparse
import json
import sys
from pathlib import Path


def parse_midi(midi_path: str):
    """返回 (bpm, chord_list, meter_numer)。chord_list 是每小节的 MusiConGen 和弦字符串。"""
    import mido
    from music21 import converter

    score = converter.parse(midi_path)

    # 1) tempo：取第一个 tempo 事件
    bpm = 120.0
    try:
        mm = score.metronomeMarkBoundaries()
        if mm:
            bpm = float(mm[0][1].number)
    except Exception:
        pass

    # 2) meter：取第一个拍号
    meter = 4
    try:
        ts = score.parts[0].timeSignature
        meter = int(ts.numerator) if ts else 4
    except Exception:
        pass

    # 3) 每小节和弦：chordify -> 根音 + 性质
    chords = []
    for part in score.parts:
        try:
            for m in part.getElementsByClass("Measure"):
                ch = m.chordify()
                if ch is None or not ch.notes:
                    continue
                root = ch.root()
                if root is None:
                    continue
                root_name = root.name.replace("-", "b")
                quality = ch.commonName.lower()
                q = ""
                if "minor" in quality:
                    q = "min"
                elif "diminished" in quality:
                    q = "dim"
                elif "augmented" in quality:
                    q = "aug"
                elif "dominant" in quality:
                    q = "7"
                chords.append(f"{root_name}:{q}" if q else root_name)
        except Exception:
            continue
        if chords:
            break

    if not chords:
        chords = ["C"]
    return bpm, chords, meter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi_dir", default=None, help="MIDI 目录（按文件名排序与 id 对齐）")
    ap.add_argument("--manifest", default=None, help='JSONL: {"id": n, "midi_path": "..."}')
    ap.add_argument("--out", default="musicongen_conditions.jsonl")
    args = ap.parse_args()

    tasks = []
    if args.manifest:
        for line in Path(args.manifest).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                d = json.loads(line)
                tasks.append((int(d.get("id", 0)), d["midi_path"]))
    elif args.midi_dir:
        midis = sorted(Path(args.midi_dir).glob("*.mid")) + \
                sorted(Path(args.midi_dir).glob("*.midi"))
        tasks = [(i, str(p)) for i, p in enumerate(midis)]
    else:
        raise SystemExit("需要 --midi_dir 或 --manifest")

    out = []
    for idx, path in tasks:
        try:
            bpm, chords, meter = parse_midi(path)
            out.append({"id": idx, "text_chords": " ".join(chords),
                        "bpm": round(bpm, 2), "meter": meter})
        except Exception as e:
            print(f"[skip] id={idx} {path}: {e}")
            out.append({"id": idx, "text_chords": "C", "bpm": 120, "meter": 4})

    with open(args.out, "w", encoding="utf-8") as f:
        for item in out:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"已写出 {len(out)} 条到 {args.out}")


if __name__ == "__main__":
    main()
