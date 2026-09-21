#!/usr/bin/env python3
"""
hf_download_slakh.py — 从 HuggingFace 国内镜像下载 Slakh2100 前 50 条
=============================================================
数据集: projectlosangeles/Slakh2100 (16kHz 重采样镜像)
输出:   C:\Deepseek\data\Slakh_Sample\ 下每 Track 一个目录
       每个目录含 all_src.mid (MIDI) + mix.wav (音频)

用法:   python hf_download_slakh.py
        python hf_download_slakh.py --num 100  # 下载 100 条
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import io
import sys
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

import argparse
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import numpy as np

try:
    from datasets import load_dataset, Features, Value
except ImportError:
    print("请先安装: pip install datasets soundfile tqdm")
    sys.exit(1)


OUTPUT_ROOT = Path("C:/Deepseek/data/Slakh_Sample")


def save_midi_bytes(midi_bytes: bytes, track_name: str) -> Path:
    """保存原始 MIDI 字节到 .mid 文件"""
    track_dir = OUTPUT_ROOT / track_name
    track_dir.mkdir(parents=True, exist_ok=True)
    midi_path = track_dir / "all_src.mid"
    midi_path.write_bytes(midi_bytes)
    return midi_path


def save_audio_bytes(audio_bytes: bytes, track_name: str) -> Path:
    """将 WAV 字节写入文件（不解码，直写）"""
    track_dir = OUTPUT_ROOT / track_name
    track_dir.mkdir(parents=True, exist_ok=True)
    wav_path = track_dir / "mix.wav"
    wav_path.write_bytes(audio_bytes)
    return wav_path


def extract_track_id(sample: dict) -> str:
    """从 __key__ 提取 Track ID"""
    key = sample.get("__key__", "")
    for part in key.split("/"):
        if part.startswith("Track"):
            return part
    return f"Track_{hash(key) & 0xFFFF:05d}"


def main():
    parser = argparse.ArgumentParser(description="从 HuggingFace 下载 Slakh2100 前 N 条")
    parser.add_argument("--num", type=int, default=50, help="下载条数 (默认: 50)")
    parser.add_argument("--offset", type=int, default=0, help="跳过前 N 条 (默认: 0)")
    args = parser.parse_args()

    print(f"加载数据集: projectlosangeles/Slakh2100 (streaming 模式)")
    print(f"镜像站:     {os.environ['HF_ENDPOINT']}")
    print(f"目标:       {args.num} 条, 跳过前 {args.offset} 条")
    print(f"输出:       {OUTPUT_ROOT}")
    print()

    # 关键: wav 列用 Value("binary") 而非 Audio(), 避开 torchcodec
    features = Features({
        "mid": Value("binary"),
        "wav": Value("binary"),
        "__key__": Value("string"),
        "__url__": Value("string"),
    })

    ds = load_dataset(
        "projectlosangeles/Slakh2100",
        split="train",
        streaming=True,
        features=features,
        trust_remote_code=False,
    )

    ds_slice = ds.skip(args.offset).take(args.num)

    success = 0
    errors = 0

    with tqdm(total=args.num, desc="下载进度", unit="track") as pbar:
        for i, sample in enumerate(ds_slice):
            track_id = extract_track_id(sample)
            pbar.set_postfix_str(f"当前: {track_id}")

            try:
                # MIDI: 原始字节
                midi_bytes = sample["mid"]
                if isinstance(midi_bytes, str):
                    import base64
                    midi_bytes = base64.b64decode(midi_bytes)
                save_midi_bytes(midi_bytes, track_id)

                # 音频: 原始字节，直写不解码
                wav_bytes = sample["wav"]
                save_audio_bytes(wav_bytes, track_id)

                success += 1
            except Exception as e:
                errors += 1
                tqdm.write(f"  ⚠️ {track_id} 失败: {e}")

            pbar.update(1)

    # 汇总
    total_mb = sum(
        f.stat().st_size
        for d in OUTPUT_ROOT.rglob("*")
        for f in [d]
        if f.is_file()
    ) / (1024 * 1024)

    print(f"\n{'='*50}")
    print(f"下载完成: {success} 成功, {errors} 失败")
    print(f"总大小:   {total_mb:.1f} MB")
    print(f"输出目录: {OUTPUT_ROOT}")

    if success > 0:
        print(f"\n目录结构示例:")
        for d in sorted(OUTPUT_ROOT.iterdir())[:3]:
            files = "  ".join(f.name for f in d.iterdir())
            print(f"  {d.name}/  →  {files}")

        print(f"\n{'='*50}")
        print("✅ 下一步: 运行 Token 转换")
        print(f"   python slakh_to_tokens.py {OUTPUT_ROOT} --max {success}")


if __name__ == "__main__":
    main()
