#!/usr/bin/env python3
"""inspect_mco.py — 扫描 SF2 音源的所有 Preset (Bank/Program/Name)"""
import sys, struct
if sys.stdout.encoding != "utf-8": sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

SF2 = Path("C:/Deepseek/soundfonts/MCO.sf2")
if not SF2.exists():
    print(f"NOT FOUND: {SF2}")
    sys.exit(1)

# 方法1: 尝试 fluidsynth CLI
print("=" * 60)
print("方法1: FluidSynth CLI")
print("=" * 60)
import subprocess
fs = Path("C:/Deepseek/fluidsynth/bin/fluidsynth.exe")
if fs.exists():
    # FluidSynth 没有直接的"列出preset"命令, 用 script 方式
    # 加载SF2后遍历bank/prog
    tmp_script = SF2.parent / "_list_presets.txt"
    with open(tmp_script,"w") as f:
        for bank in range(3):
            for prog in range(128):
                f.write(f"inst {bank*256+prog}\n")
    result = subprocess.run(
        [str(fs),"-ni",str(SF2),"-f",str(tmp_script),"-o","shell.port=1"],
        capture_output=True,text=True,encoding="utf-8",errors="ignore",timeout=30
    )
    tmp_script.unlink(missing_ok=True)
    print(result.stdout[:500] if result.stdout else "(no output)")
else:
    print("FluidSynth CLI not found - using binary parser")

# 方法2: 二进制解析 (可靠)
print()
print("=" * 60)
print("方法2: 二进制 RIFF/SF2 解析")
print("=" * 60)

with open(SF2,"rb") as f:
    data = f.read()

def walk_riff(data, depth=0):
    pos = 0
    while pos < len(data) - 8:
        tag = data[pos:pos+4]
        size = struct.unpack_from("<I", data, pos+4)[0]
        indent = "  " * depth
        if tag in (b"RIFF", b"LIST"):
            form = data[pos+8:pos+12]
            name = form.decode("latin-1","replace")
            if name == "pdta":
                inner = data[pos+12:pos+8+size]
                # PHDR
                phdr_off = inner.find(b"phdr")
                if phdr_off >= 0:
                    phdr_size = struct.unpack_from("<I", inner, phdr_off+4)[0]
                    n = phdr_size // 38
                    print(f"\n  PHDR — {n} presets:")
                    print(f"  {'Name':<25} {'Preset':>6} {'Bank':>5}")
                    print(f"  {'-'*36}")
                    found = 0
                    for i in range(n):
                        off = phdr_off + 8 + i*38
                        name = inner[off:off+20].rstrip(b"\x00").decode("latin-1","replace")
                        preset = struct.unpack_from("<H", inner, off+20)[0]
                        bank = struct.unpack_from("<H", inner, off+22)[0]
                        if name.strip() and name != "EOI":
                            print(f"  {name:<25} {preset:>6} {bank:>5}")
                            found += 1
                    print(f"\n  总计: {found} 个有效 preset")
                break
            elif name in ("sfbk","sdta"):
                walk_riff(data[pos+12:pos+8+size], depth+1)
        pos += 8 + size

walk_riff(data)

print()
print("=" * 60)
print(f"文件: {SF2} · {len(data)/1024:.0f} KB")
print("使用: instrument.program = <Preset值>")
print("=" * 60)
