#!/usr/bin/env python3
"""check_sf2.py — 读取 SF2 音源，列出所有 Bank/Program/乐器名"""
import sys
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
SF2_PATH = Path("C:/Deepseek/soundfonts/guzheng.sf2")

if not SF2_PATH.exists():
    print(f"NOT FOUND: {SF2_PATH}")
    print("请将古筝 SF2 放到 C:/Deepseek/soundfonts/guzheng.sf2")
    sys.exit(1)

# 方法1: pyfluidsynth
print("=" * 60)
print("方法1: pyfluidsynth 读取预设列表")
print("=" * 60)
try:
    import fluidsynth
    fs = fluidsynth.Synth()
    fs.start()
    sfid = fs.sfload(str(SF2_PATH))
    if sfid == -1:
        print("  ❌ SF2 加载失败")
    else:
        # 枚举所有预设
        count = 0
        for bank in range(0, 3):  # Bank 0, 1, 2
            for prog in range(128):
                name = fs.sfpreset_name(sfid, bank, prog)
                if name and name.strip() and "Piano" not in name:
                    # 过滤 GM 默认名
                    pass
                if name:
                    print(f"  Bank {bank:3d}  Prog {prog:3d}: {name}")
                    count += 1
                    if count > 10:
                        break
            if count > 10:
                break
        
        # 只打印有名字的预设
        print(f"\n  所有预设 (Bank 0):")
        for prog in range(128):
            name = fs.sfpreset_name(sfid, 0, prog)
            if name and name.strip():
                print(f"    Program {prog:3d}: {name}")
        
        fs.delete()
except Exception as e:
    print(f"  pyfluidsynth 错误: {e}")
    print("  尝试方法2...")

# 方法2: 直接二进制解析 SF2 头部
print()
print("=" * 60)
print("方法2: 二进制解析 SF2 文件头")
print("=" * 60)

try:
    import struct
    
    with open(SF2_PATH, "rb") as f:
        data = f.read()
    
    # SF2 文件头: RIFF + sfbk + INFO + sdta + pdta
    riff = data[:4]
    if riff != b"RIFF":
        print("  不是有效的 RIFF/SF2 文件!")
        sys.exit(1)
    
    # 查找 pdta 块 (Preset/Instrument data)
    pdta_offset = data.find(b"pdta")
    if pdta_offset == -1:
        print("  未找到 pdta 块!")
        sys.exit(1)
    
    pdta_size = struct.unpack_from("<I", data, pdta_offset + 4)[0]
    print(f"  pdta 块大小: {pdta_size} 字节")
    
    # 查找 pdta 内的子块
    pdta_data = data[pdta_offset + 8 : pdta_offset + 8 + pdta_size]
    
    sub_chunks = {}
    pos = 0
    while pos < len(pdta_data) - 8:
        tag = pdta_data[pos:pos+4]
        size = struct.unpack_from("<I", pdta_data, pos + 4)[0]
        sub_chunks[tag] = (pos + 8, size)
        pos += 8 + size
    
    print(f"  子块: {list(sub_chunks.keys())}")
    
    # PHDR: Preset Header — 包含 Bank/Program 映射
    if b"phdr" in sub_chunks:
        offset, size = sub_chunks[b"phdr"]
        phdr_data = pdta_data[offset : offset + size]
        record_size = 38  # 每条 phdr 记录 38 字节
        num_presets = size // record_size
        
        print(f"\n  Presets (共 {num_presets} 条):")
        print(f"  {'Name':<22} {'Preset':>6} {'Bank':>5} {'BagIdx':>8}")
        print(f"  {'-'*41}")
        
        for i in range(num_presets):
            rec = phdr_data[i * record_size : (i + 1) * record_size]
            name = rec[:20].rstrip(b"\x00").decode("latin-1", errors="replace")
            preset = struct.unpack_from("<H", rec, 20)[0]
            bank = struct.unpack_from("<H", rec, 22)[0]
            bag_idx = struct.unpack_from("<H", rec, 24)[0]
            
            if name.strip() and name != "EOI":
                print(f"  {name:<22} {preset:>6} {bank:>5} {bag_idx:>8}")
                # 标注：乐器名 + 正确的 Bank/Preset ID
    
    # INST: Instrument Header
    if b"inst" in sub_chunks:
        offset, size = sub_chunks[b"inst"]
        inst_data = pdta_data[offset : offset + size]
        rec_size = 22
        num_insts = size // rec_size
        print(f"\n  Instruments (共 {num_insts} 条):")
        for i in range(min(num_insts, 20)):
            rec = inst_data[i * rec_size : (i + 1) * rec_size]
            name = rec[:20].rstrip(b"\x00").decode("latin-1", errors="replace")
            bag_idx = struct.unpack_from("<H", rec, 20)[0]
            if name.strip() and name != "EOI":
                print(f"    {name:<25} BagIdx: {bag_idx}")

except Exception as e:
    print(f"  解析错误: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("修改指南:")
print("  sf.Program X → instrument.program = X")
print("  把上面的 Preset 值填入 guofeng_data_factory.py 的 sanitize_midi_for_sf2()")
print("=" * 60)
