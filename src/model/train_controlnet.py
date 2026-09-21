#!/usr/bin/env python3
"""
train_controlnet.py — Music ControlNet: Token-Conditioned MusicGen Fine-tuning
（极限显存优化版，适配 RTX 5060 8GB）

架构: Frozen MusicGen + Trainable Token Encoder (Concatenation 注入)
产出: checkpoint_epoch_N.pt = { token_encoder, vocab, epoch, loss }

显存优化（本轮强制落实）:
  1. bf16/fp16 全链路半精度（RTX 5060 优先 bf16）
  2. 严格冻结：除 token_encoder 外所有参数 requires_grad=False（含 T5/decoder/EnCodec）
  3. 可选梯度检查点（--grad_checkpoint，默认开）
  4. batch_size=1 + gradient_accumulation 8/16
  5. AdamW8bit（bitsandbytes 可用时自动启用，否则回退 AdamW）
  6. 训练音频截断（--max_audio_sec，默认 5 秒对应 250 EnCodec tokens）

用法:
  python train_controlnet.py --data outputs/step2_triples.jsonl --epochs 5 --save_every 5 \
      --lr 1e-4 --grad_accum 16 --max_audio_sec 5 --amp bf16
"""
import os, sys, json, argparse, math, time, re
from dataclasses import dataclass
from typing import Optional, Dict, List
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import soundfile as sf

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ─────────────────────────────────────────────────────────────
# 0. 显存工具
# ─────────────────────────────────────────────────────────────
def cuda_mem():
    if torch.cuda.is_available():
        return f"{torch.cuda.memory_allocated()/1024**3:.2f}GB used / {torch.cuda.memory_reserved()/1024**3:.2f}GB reserved"
    return "CPU"


# ═══════════════════════════════════════════════════════════
# 1. Token Vocabulary & Tokenizer（与原版一致）
# ═══════════════════════════════════════════════════════════
class MusicTokenVocabulary:
    SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[MASK]", "[SEP]"]
    DIMENSION_KEYWORDS = {
        "harmony": ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
                    "maj", "min", "dim", "aug", "7", "maj7", "min7", "dim7", "sus2", "sus4",
                    "major", "minor"],
        "rhythm": ["4/4", "3/4", "6/8", "2/4", "5/4", "7/8"],
        "dynamics": ["pp", "p", "mp", "mf", "f", "ff"],
        "texture": ["monophonic", "homophonic", "polyphonic", "sparse", "medium", "dense"],
        "timbre": ["warm", "bright", "dark", "neutral"],
        "articulation": ["legato", "staccato", "marcato"],
        "structure": ["intro", "verse", "chorus", "bridge", "outro", "section"],
        "stem": [],
    }

    def __init__(self):
        self.token2id = {}
        self.id2token = {}
        for tok in self.SPECIAL_TOKENS:
            self._add(tok)
        for group in self.DIMENSION_KEYWORDS.values():
            for tok in group:
                self._add(tok)

    def _add(self, token):
        if token not in self.token2id:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token

    def encode_token_line(self, token_line):
        ids = [self.token2id["[BOS]"]]
        for part in re.findall(r'\[([^\]]+)\]', token_line):
            for atom in part.split("|"):
                atom = atom.strip()
                if ":" in atom:
                    _, val = atom.split(":", 1)
                    for sub in val.strip().split():
                        if sub not in self.token2id:
                            self._add(sub)
                        ids.append(self.token2id[sub])
                elif atom in self.token2id:
                    ids.append(self.token2id[atom])
        ids.append(self.token2id["[EOS]"])
        return ids

    def __len__(self):
        return len(self.token2id)


# ═══════════════════════════════════════════════════════════
# 2. Token Encoder（可训练）
# ═══════════════════════════════════════════════════════════
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class MusicTokenEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=4,
                 dim_feedforward=2048, max_seq_len=1024, text_dim=1536, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(d_model, text_dim)
        self.layer_norm = nn.LayerNorm(text_dim)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, token_ids, attention_mask=None):
        x = self.token_embedding(token_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        if attention_mask is not None:
            x = self.transformer(x, src_key_padding_mask=~attention_mask)
        else:
            x = self.transformer(x)
        return self.layer_norm(self.proj(x))


# ═══════════════════════════════════════════════════════════
# 3. Controlled MusicGen（严格冻结）
# ═══════════════════════════════════════════════════════════
class ControlledMusicGen(nn.Module):
    def __init__(self, model_name="facebook/musicgen-medium",
                 dtype=torch.float32, enable_grad_checkpoint=False,
                 low_cpu_mem_usage=True):
        super().__init__()
        from transformers import MusicgenForConditionalGeneration
        # 优先使用本地预转换好的 bf16 checkpoint（避免每次 2B 参数的 CPU 转换）
        local_bf16 = Path(__file__).parent.parent / "models" / "musicgen-medium-bf16"
        load_id = str(local_bf16) if (local_bf16 / "model.safetensors").exists() else model_name
        # 本地 bf16 safetensors 已是目标 dtype，无需 low_cpu_mem_usage 流式转换
        self.musicgen = MusicgenForConditionalGeneration.from_pretrained(
            load_id, torch_dtype=dtype, low_cpu_mem_usage=False)
        print("", flush=True)

        # ── 严格冻结：整个 musicgen（含 T5 encoder / decoder / EnCodec）──
        for name, p in self.musicgen.named_parameters():
            p.requires_grad_(False)
        print("", flush=True)
        # 兜底：也冻结 buffers 之外的任何可训练子模块
        for p in self.musicgen.parameters():
            p.requires_grad_(False)

        # 可训练的 Token Encoder
        # 词汇表在运行时通过 encode_token_line 动态扩展（乐器名等），
        # 因此 Embedding 表需预留余量，且 dataset 侧对超长 id 做 clip。
        self.token_vocab = MusicTokenVocabulary()
        self._vocab_headroom = 8192
        self.token_vocab.headroom = self._vocab_headroom
        self.token_encoder = MusicTokenEncoder(
            vocab_size=len(self.token_vocab) + self._vocab_headroom,
            d_model=512, nhead=8, num_layers=4,
            dim_feedforward=2048,
            text_dim=self.musicgen.config.decoder.hidden_size,
        ).to(dtype=dtype)

        self.text_dim = self.musicgen.config.decoder.hidden_size
        self.ce_trained = False   # True = 真实 EnCodec CE 训练（推理用满强度注入）

        if enable_grad_checkpoint:
            try:
                self.musicgen.gradient_checkpointing_enable()
                print("", flush=True)
            except Exception as e:
                print(f"  [warn] gradient checkpointing enable failed: {e}")

    def freeze_check(self):
        """返回 (可训练数, 应冻结但误开数, 总参数量)——正确性校验"""
        trainable = []
        leak = []
        for name, p in self.named_parameters():
            if p.requires_grad:
                if name.startswith("token_encoder"):
                    trainable.append(name)
                else:
                    leak.append(name)
        total = sum(p.numel() for p in self.musicgen.parameters())
        adapter = sum(p.numel() for p in self.token_encoder.parameters())
        return trainable, leak, total, adapter

    def train(self, mode=True):
        super().train(mode)
        self.musicgen.eval()          # MusicGen 永远 eval（即使 model.train()）
        self.token_encoder.train(mode)
        return self

    def forward(self, text_input_ids, text_attention_mask, token_ids,
                token_attention_mask, decoder_input_ids,
                decoder_attention_mask=None, labels=None, ce_loss=False):
        # 冻结 T5 text encoder
        with torch.no_grad():
            te_out = self.musicgen.text_encoder(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask).last_hidden_state
            text_emb = self.musicgen.enc_to_dec_proj(te_out)

        # token_encoder 是可训练支路
        token_emb = self.token_encoder(token_ids, token_attention_mask)

        if ce_loss:
            # ── 真实 EnCodec 交叉熵训练（论文级）──
            # text/token 拼接注入 decoder 交叉注意力；decoder 用教师强制 CE 预测下一帧音频码。
            # decoder_input_ids: [B, cb, T-1]（codes[:, :-1]）；labels: [B, T-1, cb]（codes[:, 1:]）。
            # decoder/EnCodec 全冻结，梯度只到 token_encoder。
            if text_emb.shape[-1] != self.musicgen.config.decoder.hidden_size:
                # 兜底：enc_to_dec_proj 缺失时对齐
                text_emb = F.linear(text_emb, torch.eye(text_emb.shape[-1], self.musicgen.config.decoder.hidden_size).to(text_emb.device))
            combined_emb = torch.cat([text_emb, token_emb], dim=1)
            combined_mask = torch.cat([text_attention_mask, token_attention_mask], dim=1)
            # HF 的 MusicgenForCausalLM 明确禁止 labels 路径（NotImplementedError），
            # 因此手工取 logits 自算 shift CE：logits [B*cb, T-1, V] 逐 codebook 对齐标签。
            dec = self.musicgen.decoder(
                input_ids=decoder_input_ids,
                attention_mask=None,          # CE 模式 batch=1、长度固定，无需 mask
                encoder_hidden_states=combined_emb,
                encoder_attention_mask=combined_mask,
                return_dict=True,
            )
            logits = dec.logits                              # [B*cb, T-1, V]
            V = logits.shape[-1]
            target = labels.reshape(-1)                      # 顺序 (b, c, t) 与 logits 一致
            if target.numel() != logits.shape[0] * logits.shape[1]:
                raise ValueError(f"CE 标签/ logits 不匹配: {target.numel()} vs "
                                 f"{logits.shape[0] * logits.shape[1]}")
            loss = F.cross_entropy(logits.reshape(-1, V), target)

            class _OutCE:
                pass
            out = _OutCE()
            out.loss = loss
            return out

        # ── 训练闭环（不依赖真实音频 token）──
        # 本 checkpoint 只保存 token_encoder，decoder/EnCodec 全冻结不更新。
        # 用一个「把 token_emb 对齐到冻结 text_emb 分布」的辅助 loss，
        # 既给 token_encoder 提供稳定梯度，又避开了 MusicGen decoder 的
        # [B,4,T] CE 形状约束（占位音频无法满足那套 CE）。
        target = text_emb.detach()
        # 对齐到相同长度：截断较长者
        L = min(token_emb.shape[1], target.shape[1])
        align_loss = F.mse_loss(token_emb[:, :L], target[:, :L])
        # 轻微正则，防止嵌入数值发散
        reg_loss = token_emb.pow(2).mean() * 1e-4
        loss = align_loss + reg_loss

        class _Out:
            pass
        out = _Out()
        out.loss = loss
        return out

    @torch.no_grad()
    def generate(self, text, token_seqs, processor, max_new_tokens=256, **kw):
        self.eval()
        text_inputs = processor(text=text, padding=True, return_tensors="pt")
        te_out = self.musicgen.text_encoder(
            input_ids=text_inputs["input_ids"].to(self.musicgen.device),
            attention_mask=text_inputs["attention_mask"].to(self.musicgen.device)).last_hidden_state
        text_emb = self.musicgen.enc_to_dec_proj(te_out)
        token_ids_list = [self.token_vocab.encode_token_line(t) for t in token_seqs]
        max_tok = max(len(i) for i in token_ids_list)
        token_ids = torch.zeros(len(token_seqs), max_tok, dtype=torch.long)
        token_mask = torch.zeros(len(token_seqs), max_tok, dtype=torch.bool)
        for i, ids in enumerate(token_ids_list):
            token_ids[i, :len(ids)] = torch.tensor(ids)
            token_mask[i, :len(ids)] = True
        token_emb = self.token_encoder(token_ids.to(self.musicgen.device),
                                       token_mask.to(self.musicgen.device))
        # 尺度对齐：token_encoder 输出 std≈1.0，而 T5 投影后的 text_emb std≈0.44；
        # 不缩放会把 decoder 交叉注意力打发散导致噪音。按样本动态对齐 std 并加衰减。
        with torch.no_grad():
            te_std = text_emb.std(dim=-1, keepdim=True).mean() + 1e-6
            to_std = token_emb.std(dim=-1, keepdim=True).mean() + 1e-6
            # 替身(align)训练：衰减到 text 尺度 0.5 避免注入过强；
            # 真实 CE 训练(ce_trained)：满强度注入。
            factor = 1.0 if self.ce_trained else 0.5
            scale = (te_std / to_std).item() * factor
        token_emb = token_emb * scale
        combined_emb = torch.cat([text_emb, token_emb], dim=1)
        combined_mask = torch.cat([
            text_inputs["attention_mask"].to(self.musicgen.device), token_mask.to(self.musicgen.device)], dim=1)

        # ── KV-concat 注入（transformers 4.35 正确通道）──
        # 4.35 的 _prepare_encoder_decoder_kwargs_for_generation 会在 model_kwargs
        # 已包含 "encoder_outputs" 时【跳过】重新编码，因此只需传 BaseModelOutput
        # + encoder_attention_mask；旧的 _prepare_text_encoder_kwargs_for_generation
        # 名字在 4.35 已不存在（patch 会静默失效，导致 mask 长度错配）。
        from transformers.modeling_outputs import BaseModelOutput
        import torch.nn as _nn
        # combined_emb 已是 decoder 维度（1536），临时把 enc_to_dec_proj 设恒等，
        # 避免 decoder 内部二次投影（generate 返回后恢复）。
        old_proj = self.musicgen.enc_to_dec_proj
        self.musicgen.enc_to_dec_proj = _nn.Identity()
        try:
            out = self.musicgen.generate(
                input_ids=text_inputs["input_ids"].to(self.musicgen.device),
                attention_mask=combined_mask.long(),
                encoder_outputs=BaseModelOutput(last_hidden_state=combined_emb),
                max_new_tokens=max_new_tokens,
                **kw)
        finally:
            self.musicgen.enc_to_dec_proj = old_proj
        return out


# ═══════════════════════════════════════════════════════════
# 4. Dataset（音频截断，decoder_input 占位 —— 训练只针对 token_encoder）
# ═══════════════════════════════════════════════════════════
class MusicControlDataset(Dataset):
    def __init__(self, jsonl_path, processor, vocab, max_audio_len=250,
                 max_token_len=512, encodec_dir=None):
        self.processor = processor
        self.vocab = vocab
        self.max_audio_len = max_audio_len
        self.max_token_len = max_token_len
        self.encodec_dir = Path(encodec_dir) if encodec_dir else None
        with open(jsonl_path, encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f]
        # CE 模式：只保留有 EnCodec token 缓存的行
        if self.encodec_dir is not None:
            import os as _os
            self.valid = [i for i in range(len(self.data))
                          if _os.path.exists(str(self.encodec_dir / f"{i:04d}.npz"))]
            print(f"  [CE] 有效音频样本 {len(self.valid)}/{len(self.data)}")
        else:
            self.valid = list(range(len(self.data)))

    def __len__(self):
        return len(self.valid)

    def __getitem__(self, idx):
        real_idx = self.valid[idx]
        item = self.data[real_idx]
        text_enc = self.processor(
            text=[item["text"]], padding="max_length", max_length=128,
            truncation=True, return_tensors="pt")
        # token_ids（复用一次 encode 避免 class 内重复计算）
        tok_line = item.get("tokens") or item.get("token") or ""
        raw_ids = self.vocab.encode_token_line(tok_line)[:self.max_token_len]
        # Embedding 有 headroom；把超过当前表的 id clip 到 PAD，避免越界
        headroom = getattr(self.vocab, "headroom", 8192)
        raw_ids = [i if i < len(self.vocab) + headroom else 0 for i in raw_ids]
        token_ids = raw_ids + [0] * (self.max_token_len - len(raw_ids))
        token_mask = [1] * len(raw_ids) + [0] * (self.max_token_len - len(raw_ids))

        if self.encodec_dir is not None:
            # 真实 EnCodec codes: [4, T] -> input codes[:, :-1], labels codes[:, 1:]
            codes = np.load(str(self.encodec_dir / f"{real_idx:04d}.npz"))["codes"]
            if codes.ndim == 3:                              # [1, 4, T] -> [4, T]
                codes = codes[0]
            codes = torch.from_numpy(codes).long()           # [4, T]
            seq = codes.shape[1]
            audio_input = codes[:, :seq - 1]                 # [4, T-1]
            audio_labels = codes[:, 1:]                      # [4, T-1]（与 logits 行序 (b,c) 一致）
            audio_mask = torch.ones(seq - 1, dtype=torch.bool)
        else:
            # 占位（旧对齐训练路径）
            audio_input = torch.full((4, self.max_audio_len), 3, dtype=torch.long)
            audio_labels = torch.full((self.max_audio_len, 4), 3, dtype=torch.long)
            audio_mask = torch.ones(self.max_audio_len, dtype=torch.bool)
        return {
            "text_input_ids": text_enc["input_ids"].squeeze(0),
            "text_attention_mask": text_enc["attention_mask"].squeeze(0),
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "token_attention_mask": torch.tensor(token_mask, dtype=torch.bool),
            "decoder_input_ids": audio_input,
            "decoder_attention_mask": audio_mask,
            "labels": audio_labels,
        }


# ═══════════════════════════════════════════════════════════
# 5. 训练（显存优化 + 8-bit 优化器 + 自动 OOM 重试参数）
# ═══════════════════════════════════════════════════════════
def make_optimizer(model, lr):
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(model.token_encoder.parameters(), lr=lr)
        print("  [OK] 使用 bitsandbytes AdamW8bit")
        return opt
    except Exception as e:
        print(f"  [warn] bitsandbytes 不可用({e.__class__.__name__})，回退 AdamW")
        return torch.optim.AdamW(model.token_encoder.parameters(), lr=lr)


def train(args):
    from transformers import AutoProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    dtype = torch.bfloat16
    if args.amp == "fp16":
        dtype = torch.float16
    if device.type == "cpu":
        dtype = torch.float32
        print("  [warn] CPU 模式：禁用半精度")

    print(f"Device: {device} | dtype: {dtype} | gradient_checkpoint: {args.grad_checkpoint}")
    print("Loading MusicGen-medium (frozen backbone)...")
    model = ControlledMusicGen("facebook/musicgen-medium", dtype=dtype,
                               enable_grad_checkpoint=args.grad_checkpoint)

    if getattr(args, "force_cpu", False) or not torch.cuda.is_available():
        device = torch.device("cpu")
        dtype = torch.float32
        model.to("cpu")
        print("  [warn] CPU 训练（仅无 GPU 时）")
    else:
        # GPU 训练：5090/大显存直接整模型上卡（bf16 ~4GB）
        model.to(device)
        print(f"  [OK] GPU 训练：全模型上 {device}")

    trainable, leak, total, adapter = model.freeze_check()
    print(f"  trainable adapter params: {adapter:,}")
    print(f"  frozen backbone params: {total:,}")
    if leak:
        print(f"  [X] 发现未冻结的非 adapter 参数: {leak[:5]}")
        raise RuntimeError("冻结校验失败")
    print(f"  [OK] 严格冻结校验通过：仅 token_encoder 可训练")

    processor = AutoProcessor.from_pretrained("facebook/musicgen-medium")
    max_audio_len = int(args.max_audio_sec * 50)  # EnCodec 50Hz
    ce_mode = bool(args.encodec_dir)
    dataset = MusicControlDataset(args.data, processor, model.token_vocab,
                                  max_audio_len=max_audio_len,
                                  encodec_dir=args.encodec_dir)
    if ce_mode and args.batch_size > 1:
        print("  [CE] batch_size 强制为 1（码长逐样本对齐），用 grad_accum 累积")
        args.batch_size = 1
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, pin_memory=(device.type == "cuda"))
    print(f"Dataset: {len(dataset)} samples | max_audio_len: {max_audio_len} tokens | CE={ce_mode}")

    optimizer = make_optimizer(model, args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs * len(dataloader), 1))

    # bf16 不需要（也不支持）GradScaler，仅 fp16 启用
    use_amp = (device.type == "cuda" and dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model.train()
    global_step = 0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        n_batches = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        optimizer.zero_grad(set_to_none=True)
        for batch in pbar:
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=dtype,
                                enabled=(device.type == "cuda")):
                outputs = model(**batch, ce_loss=ce_mode)
                loss = outputs.loss / args.grad_accum
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            global_step += 1

            if global_step % args.grad_accum == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.token_encoder.parameters(), 1.0)
                if use_amp:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            epoch_loss += loss.item() * args.grad_accum
            n_batches += 1
            if n_batches % 20 == 0:
                pbar.set_postfix({"loss": f"{loss.item()*args.grad_accum:.4f}",
                                  "mem": cuda_mem()})

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f} | {cuda_mem()}")

        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "loss_type": "ce" if ce_mode else "align",
                "token_encoder": model.token_encoder.state_dict(),
                "vocab": model.token_vocab.token2id,
                "optimizer": optimizer.state_dict(),
                "loss": avg_loss,
            }, ckpt_path)
            print(f"  [SAVED] {ckpt_path}")

    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Music ControlNet Training (memory-optimized)")
    parser.add_argument("--data", default="C:/Deepseek/outputs/step2_triples.jsonl")
    parser.add_argument("--output_dir", default="C:/Deepseek/outputs/checkpoints")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--max_audio_sec", type=float, default=5.0)
    parser.add_argument("--encodec_dir", default=None,
                        help="EnCodec token 缓存目录（真实 CE 训练）；先跑 data.prepare_encodec_tokens")
    parser.add_argument("--force_cpu", action="store_true", help="强制 CPU（调试用）")
    parser.add_argument("--amp", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--grad_checkpoint", action="store_true", default=False,
                        help="对 MusicGen 主干开启 gradient checkpointing")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        # dry run：只加载模型 + 冻结校验，不训练
        train(args) if False else None
        from transformers import AutoProcessor
        print("DRY RUN (freeze check only)")
        model = ControlledMusicGen("facebook/musicgen-medium", dtype=torch.bfloat16)
        trainable, leak, total, adapter = model.freeze_check()
        print(f"  adapter params: {adapter:,} | backbone: {total:,} | leak: {len(leak)}")
        print("  OK" if not leak else f"  LEAK: {leak[:5]}")
    else:
        train(args)
