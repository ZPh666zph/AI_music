#!/usr/bin/env python3
"""batch_rename_midi.py — 清洗 MIDI 文件名中的杂质"""
import sys, os, re
from pathlib import Path
if sys.stdout.encoding != "utf-8": sys.stdout.reconfigure(encoding="utf-8")

DIR = Path("C:/Deepseek/data/raw_midis")
PATTERNS = [
    r"_爱给网_aigei_com",
    r"_爱给网",
    r"_aigei_com",
    r"\（Cover[^）]*）",
    r"\(Cover[^)]*\)",
    r"_人人钢琴网抄谱",
    r"\(完整版\)",
    r"-s670弹奏_修改",
    r"\(钢琴版\)",
    r"_修改",
    r"\(2\)",
    r"\(Vocals\)_",
    r"（非自制）",
    r"_自截曲",
    r"自截曲_",
    r"_完整版",
]

renamed = 0
for f in sorted(DIR.glob("*.mid")):
    name = f.stem
    original = name
    for p in PATTERNS:
        name = re.sub(p, "", name)
    # 清理多余的下划线/空格/括号
    name = re.sub(r"[_]{2,}", "_", name)
    name = re.sub(r"^[_\s]+|[_\s]+$", "", name)
    name = re.sub(r"\s+", "_", name)
    if name != original:
        new = DIR / f"{name}.mid"
        if new.exists():
            print(f"  SKIP (exists): {name}.mid")
        else:
            f.rename(new)
            print(f"  {original}  →  {name}.mid")
            renamed += 1
    else:
        print(f"  KEEP: {name}.mid")

print(f"\nRenamed: {renamed} files")
