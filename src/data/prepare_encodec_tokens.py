# -*- coding: utf-8 -*-
"""
prepare_encodec_tokens.py — 用 MusicGen 的 frozen EnCodec 把真值音频编码成 CE 训练标签。

用途：真实 EnCodec 交叉熵训练（论文级 Ours）前，先把训练音频离线编码一次，
     把 [text, DSL tokens, audio_path] JSONL 转成「直接可训练」的 npz token 缓存，
     避免训练时反复 encode。

输入：JSONL（每行 {text, tokens/token, audio_path}）
输出：<out_dir>/<idx>.npz  （每文件含 audio_codes [4, T] int）
      <out_dir>/manifest.jsonl（idx -> 行内 text/tokens/style）

编码参数：
  - guofeng_v3 渲染音频是 16 kHz mono（工厂 SR）；MusicGen-medium 的 EnCodec 是 32 kHz，
    这里用 torchaudio 重采样到 32 kHz（Kaiser 窗）。
  - encode 前截断到 max_audio_sec（默认 10s = 320k @32k），防止超长音频炸显存。

用法（服务器，transformers 4.35 env）：
  python -m data.prepare_encodec_tokens \
      --jsonl /root/autodl-tmp/data/gufeng_train.jsonl \
      --out /root/autodl-tmp/data/encodec_cache \
      --max_audio_sec 10
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

SR_ENC = 32000  # EnCodec 32k


def load_audio_encoder(device, model_name="facebook/musicgen-medium"):
    from transformers import MusicgenForConditionalGeneration
    m = MusicgenForConditionalGeneration.from_pretrained(model_name)
    m.audio_encoder.eval().to(device)
    return m


def encode_one(path, model, device, max_samples):
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR_ENC:
        y = AF.resample(torch.from_numpy(y), sr, SR_ENC).numpy()
    y = y[:max_samples]
    if y.size == 0:
        return None
    wav = torch.from_numpy(y).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,T]
    with torch.no_grad():
        out = model.audio_encoder.encode(wav)
        audio_codes = out.audio_codes  # [B, nb_q, T']
    return audio_codes[0].detach().cpu().numpy()  # [4, T']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_audio_sec", type=float, default=10.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model = load_audio_encoder(args.device)
    max_samples = int(SR_ENC * args.max_audio_sec)

    rows = [json.loads(l) for l in open(args.jsonl, encoding="utf-8") if l.strip()]
    print(f"records: {len(rows)}", flush=True)
    manifest = []
    ok = fail = 0
    for i, row in enumerate(rows):
        ap_path = row.get("audio_path") or row.get("audio") or ""
        if not ap_path or not os.path.exists(ap_path):
            fail += 1
            print(f"[skip no-audio] {i} {ap_path}", flush=True)
            continue
        codes = encode_one(ap_path, model, args.device, max_samples)
        if codes is None:
            fail += 1
            continue
        np.savez(os.path.join(args.out, f"{i:04d}.npz"), codes=codes)
        manifest.append({"id": i,
                         "text": row.get("text", ""),
                         "tokens": row.get("tokens") or row.get("token", ""),
                         "audio_path": ap_path})
        ok += 1
        if (i + 1) % 100 == 0:
            print(f"{i+1}/{len(rows)} ok={ok} fail={fail}", flush=True)

    with open(os.path.join(args.out, "manifest.jsonl"), "w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"DONE ok={ok} fail={fail} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
