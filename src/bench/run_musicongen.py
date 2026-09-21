# -*- coding: utf-8 -*-
"""
run_musicongen.py — MusiConGen 批量推理（chord + rhythm 控制）

前置条件（由 run_setup 完成）：
    1. clone 了 sakemin/cog-MusiConGen 到 musicongen_repo/
    2. 下载权重 Cyan0731/MusiConGen 的 state_dict.bin / compression_state_dict.bin
       到 musicongen_repo/audiocraft/ckpt/musicongen/
    3. 激活 musicongen 环境（Python 3.9 + torch 2.0.0 + audiocraft 依赖）

输入：
    baseline_prompts.json —— 每条的 baseline_prompt（自然语言）
    MIDI 对照（可选）—— musicongen_conditions.jsonl，每行含 text_chords / bpm / meter，
          由 midi_to_musicongen_conditions.py 生成；缺省时用全曲默认 C 大调 120 BPM 4/4。

输出：
    ./results/musicongen/NNN.wav —— 10 秒（duration=10），文件名与测试集 ID 对齐
    ./results/musicongen/failed.jsonl —— 失败清单

用法（在 audiocraft 目录外，用 sys.path 挂载仓库）：
    python run_musicongen.py --repo musicongen_repo --prompts baseline_prompts.json \
                             --conds musicongen_conditions.jsonl --out_root ./results
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).parent / "musicongen_repo"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "audiocraft") not in sys.path:
    sys.path.insert(0, str(REPO / "audiocraft"))


# ─────────────────────────────────────────────────────────────
# 输入读取
# ─────────────────────────────────────────────────────────────
def load_prompts(path: str):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        records = json.loads(text)
    else:
        records = [json.loads(l) for l in text.splitlines() if l.strip()]
    out = []
    for i, rec in enumerate(records):
        idx = rec.get("id", i)
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            idx = i
        prompt = rec.get("baseline_prompt") or rec.get("text") or rec.get("prompt") or ""
        out.append((idx, prompt.strip()))
    return [x for x in out if x[1]]


def load_conds(path: str):
    """musicongen_conditions.jsonl -> {id: {text_chords, bpm, meter}}"""
    p = Path(path)
    if not p.exists():
        return {}
    conds = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        idx = int(d.get("id", d.get("idx", 0)))
        conds[idx] = {
            "chords": d.get("text_chords", d.get("chords", "C")),
            "bpm": float(d.get("bpm", 120)),
            "meter": int(d.get("meter", d.get("time_sig_numer", 4))),
        }
    return conds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="musicongen_repo")
    ap.add_argument("--ckpt", default="audiocraft/ckpt/musicongen",
                    help="权重目录（相对 repo 或绝对路径）")
    ap.add_argument("--prompts", default="baseline_prompts.json")
    ap.add_argument("--conds", default="musicongen_conditions.jsonl")
    ap.add_argument("--out_root", default="./results")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    repo = Path(args.repo)
    ckpt = Path(args.ckpt)
    if not ckpt.is_absolute():
        ckpt = repo / ckpt
    if not (ckpt / "state_dict.bin").exists():
        raise SystemExit(f"未找到权重: {ckpt}/state_dict.bin —— 请先下载 Cyan0731/MusiConGen 权重")

    import torch
    import audiocraft.models

    pairs = load_prompts(args.prompts)
    conds = load_conds(args.conds)
    print(f"读取 prompts: {len(pairs)} 条，conds: {len(conds)} 条")

    musicgen = audiocraft.models.MusicGen.get_pretrained(str(ckpt))
    musicgen.set_generation_params(
        duration=args.seconds, extend_stride=args.seconds // 2, top_k=250)

    out_dir = Path(args.out_root) / "musicongen"
    out_dir.mkdir(parents=True, exist_ok=True)
    failed = []

    for n, (idx, prompt) in enumerate(pairs):
        fname = f"{max(int(idx), 0):03d}.wav"
        dest = out_dir / fname
        if dest.exists():
            continue
        c = conds.get(int(idx), {"chords": "C", "bpm": 120, "meter": 4})
        try:
            torch.manual_seed(args.seed + int(idx))
            wav = musicgen.generate_with_chords_and_beats(
                [prompt], [c["chords"]], [c["bpm"]], [c["meter"]])
            a = wav[0].cpu().numpy()
            if a.ndim > 1:
                a = a[0]
            sf.write(str(dest), a, int(musicgen.sample_rate))
            if n % 10 == 0:
                print(f"[{n+1}/{len(pairs)}] {fname} ok", flush=True)
        except Exception as e:
            failed.append({"id": idx, "error": str(e)})
            print(f"[error] id={idx} -> {fname}: {e}", flush=True)

    if failed:
        with open(out_dir / "failed.jsonl", "w", encoding="utf-8") as f:
            for item in failed:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"完成: 成功 {len(pairs) - len(failed)}，失败 {len(failed)}，目录 {out_dir}")


if __name__ == "__main__":
    main()
