# -*- coding: utf-8 -*-
"""
run_ours.py — 用 checkpoint_epoch5.pt 的 token_encoder 生成 668 条 MontageDirector 测试音频。

流程：
  1. 等待/确认 outputs/checkpoints/checkpoint_epoch5.pt 存在
  2. 加载 ControlledMusicGen（冻结 backbone）+ checkpoint 的 token_encoder 权重 + vocab
  3. 读取 data/gufeng_moe_train.jsonl（原始 DSL token），按 id 0..667 对齐
  4. 每条：token_seqs=[DSL原始token], text=[自然语言text]，generate 10 秒（max_new_tokens=500 @50Hz）
  5. 输出 results/ours/NNN.wav（强制 10.0s, mono, 32kHz）

用法（musicongen3 环境，训练完成后）：
    python run_ours.py --ckpt outputs/checkpoints/checkpoint_epoch5.pt \
        --data data/gufeng_moe_train.jsonl --out_dir results/ours
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from model.train_controlnet import ControlledMusicGen, MusicTokenVocabulary


def load_records(path):
    """读 data/gufeng_moe_train.jsonl，返回按顺序的 (text, token) list。"""
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    out = []
    for rec in records:
        text = rec.get("text", "")
        token = rec.get("tokens", rec.get("token", ""))
        out.append((text.strip(), token))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/checkpoints/checkpoint_epoch5.pt")
    ap.add_argument("--data", default="data/gufeng_moe_train.jsonl")
    ap.add_argument("--out_dir", default="results/ours")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--guidance", type=float, default=None,
                    help="classifier-free guidance；None=关闭（KV-concat 注入下 CFG 会复制 encoder_outputs 导致形状错误）")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    # 连招：等待 checkpoint_epoch5.pt 出现（最多 90 分钟）
    dead_from = time.time()
    while not ckpt_path.exists():
        if time.time() - dead_from > 5400:
            raise SystemExit(f"等待超时：{ckpt_path} 未出现")
        if (time.time() - dead_from) > 30:
            print(f"等待 {ckpt_path} ...", flush=True)
            dead_from = time.time()
        time.sleep(10)
    print(f"发现 checkpoint: {ckpt_path}", flush=True)

    # 1. 载入架构 + 冻结 backbone
    use_gpu = args.device.startswith("cuda")
    dtype = torch.bfloat16 if use_gpu else torch.float32
    model = ControlledMusicGen("facebook/musicgen-medium", dtype=dtype)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # 2. 恢复 vocab（checkpoint 里有 token2id）
    if "vocab" in ckpt and isinstance(ckpt["vocab"], dict):
        model.token_vocab.token2id = ckpt["vocab"]
        model.token_vocab.id2token = {v: k for k, v in ckpt["vocab"].items()}
    # 3. 载入 token_encoder 权重（strict=False 兼容 headroom / 缺少 optimizer 字段）
    model.token_encoder.load_state_dict(ckpt["token_encoder"], strict=False)
    model.ce_trained = bool(ckpt.get("loss_type") == "ce")

    # 4. 推理：整个模型上 GPU（单条生成，8GB 足够）
    if use_gpu:
        # 分块上卡：先 decoder（最大），再 T5，再 audio_encoder，最后 adapter
        for name, mod in [("decoder", model.musicgen.decoder),
                          ("text_encoder", model.musicgen.text_encoder),
                          ("audio_encoder", model.musicgen.audio_encoder),
                          ("enc_to_dec_proj", model.musicgen.enc_to_dec_proj),
                          ("token_encoder", model.token_encoder)]:
            print(f"  transferring {name} to cuda...", flush=True)
            mod.to("cuda")
        print("  all modules on cuda", flush=True)
    else:
        model.to("cpu")
    model.eval()
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained("facebook/musicgen-medium")

    records = load_records(args.data)
    print(f"测试集 {len(records)} 条，开始生成 10s 音频到 {args.out_dir}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    max_tokens = int(round(args.seconds * 50))  # 50Hz

    for i, (text, token) in enumerate(records):
        fname = f"{i:03d}.wav"
        dest = os.path.join(args.out_dir, fname)
        if os.path.exists(dest):
            continue
        try:
            torch.manual_seed(args.seed + i)
            audio = model.generate(
                text=[text],
                token_seqs=[token],
                processor=processor,
                max_new_tokens=max_tokens,
                do_sample=True,
                guidance_scale=args.guidance,
            )
            # audio: [B, C, T] 或 [B, T]
            a = audio[0].detach().cpu().float().numpy()
            if a.ndim > 1:
                a = a[0]  # 取第一个通道的单声道
            sr = int(model.musicgen.config.audio_encoder.sampling_rate) or 32000
            target = int(round(sr * args.seconds))
            if len(a) > target:
                a = a[:target]
            elif len(a) < target:
                a = np.pad(a, (0, target - len(a)))
            sf.write(dest, a, sr)
            if (i + 1) % 50 == 0:
                print(f"[{i+1}/{len(records)}] {fname} ok", flush=True)
        except Exception as e:
            print(f"[error] id={i} -> {fname}: {e}", flush=True)

    n = len(list(Path(args.out_dir).glob("*.wav")))
    print(f"完成：{n}/{len(records)} 个 wav 已写入 {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
