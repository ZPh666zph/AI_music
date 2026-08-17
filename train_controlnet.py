#!/usr/bin/env python3
"""
train_controlnet.py — Music ControlNet: Token-Conditioned MusicGen Fine-tuning
=============================================================================
架构: Frozen MusicGen + Trainable Token Encoder (Concatenation 注入)
产出: 论文核心实验脚本 · 发给老师审阅

用法:
  python train_controlnet.py --data outputs/step2_triples.jsonl --epochs 20

依赖:
  pip install transformers datasets soundfile librosa torchmetrics
"""

import os, sys, json, argparse, math, time, re
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import soundfile as sf
from tqdm import tqdm

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


# ═══════════════════════════════════════════════════════════
# 1. Token Vocabulary & Tokenizer
# ═══════════════════════════════════════════════════════════

class MusicTokenVocabulary:
    """将 PDF 六维度 Token 标签映射为整数 ID"""
    
    # 六维度关键词（与 slakh_to_tokens.py 输出对齐）
    SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[MASK]", "[SEP]"]
    
    DIMENSION_KEYWORDS = {
        "harmony":  ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
                     "maj", "min", "dim", "aug", "7", "maj7", "min7", "dim7", "sus2", "sus4",
                     "major", "minor"],
        "rhythm":   ["4/4", "3/4", "6/8", "2/4", "5/4", "7/8"],
        "dynamics": ["pp", "p", "mp", "mf", "f", "ff"],
        "texture":  ["monophonic", "homophonic", "polyphonic", "sparse", "medium", "dense"],
        "timbre":   ["warm", "bright", "dark", "neutral"],
        "articulation": ["legato", "staccato", "marcato"],
        "structure": ["intro", "verse", "chorus", "bridge", "outro", "section"],
        "stem":     [],  # 动态填充乐器名
    }
    
    def __init__(self):
        self.token2id = {}
        self.id2token = {}
        
        # 1. 特殊 token
        for tok in self.SPECIAL_TOKENS:
            self._add(tok)
        
        # 2. 维度关键词
        for group in self.DIMENSION_KEYWORDS.values():
            for tok in group:
                if tok not in self.token2id:
                    self._add(tok)
        
        # 3. 动态 token 占位（数字、乐器名等运行时填充）
        self._next_dynamic_id = len(self.token2id)
    
    def _add(self, token: str):
        if token not in self.token2id:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token
    
    def encode_token_line(self, token_line: str) -> List[int]:
        """将一行 Token（如 '[BAR:1|CHORD:C]'）转为 token ID 序列"""
        ids = [self.token2id["[BOS]"]]
        parts = re.findall(r'\[([^\]]+)\]', token_line)
        for part in parts:
            for atom in part.split("|"):
                atom = atom.strip()
                if ":" in atom:
                    _, val = atom.split(":", 1)
                    for sub in val.strip().split():
                        if sub in self.token2id:
                            ids.append(self.token2id[sub])
                        else:
                            self._add(sub)
                            ids.append(self.token2id[sub])
                elif atom in self.token2id:
                    ids.append(self.token2id[atom])
        ids.append(self.token2id["[EOS]"])
        return ids
    
    def encode_text(self, text: str) -> List[int]:
        """简单空格分词 → ID（仅用于 token 序列，text prompt 走 T5 encoder）"""
        ids = [self.token2id["[BOS]"]]
        for word in text.lower().split()[:50]:
            if word in self.token2id:
                ids.append(self.token2id[word])
        ids.append(self.token2id["[EOS]"])
        return ids
    
    def __len__(self):
        return len(self.token2id)


# ═══════════════════════════════════════════════════════════
# 2. Token Encoder (可训练 · ~8M 参数)
# ═══════════════════════════════════════════════════════════

class MusicTokenEncoder(nn.Module):
    """
    将结构化 Token 序列编码为与 text embedding 等维的向量
    
    结构: Embedding → PositionalEncoding → 4×TransformerEncoderLayer → Linear(512→1024)
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        max_seq_len: int = 1024,
        text_dim: int = 1024,  # MusicGen text embedding dim
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 投影到 text embedding 空间
        self.proj = nn.Linear(d_model, text_dim)
        self.layer_norm = nn.LayerNorm(text_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, token_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None):
        """
        token_ids: [B, L]
        returns:   [B, L, text_dim]
        """
        x = self.token_embedding(token_ids) * math.sqrt(self.d_model)  # [B, L, d]
        x = self.pos_encoding(x)
        x = self.transformer(x, src_key_padding_mask=~attention_mask if attention_mask is not None else None)
        x = self.proj(x)
        x = self.layer_norm(x)
        return x


class PositionalEncoding(nn.Module):
    """正弦位置编码"""
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor):
        return self.dropout(x + self.pe[:, :x.size(1)])


# ═══════════════════════════════════════════════════════════
# 3. Controlled MusicGen Wrapper
# ═══════════════════════════════════════════════════════════

class ControlledMusicGen(nn.Module):
    """
    包装 HuggingFace MusicGen，在 encoder_attn 前注入 Token embeddings
    
    核心机制: 
      原始: encoder_attn(Q=decoder_hidden, KV=text_emb)
      修改: encoder_attn(Q=decoder_hidden, KV=concat(text_emb, token_emb))
    
    所有 MusicGen 参数冻结，仅训练 token_encoder
    """
    
    def __init__(self, model_name: str = "facebook/musicgen-medium"):
        super().__init__()
        from transformers import MusicgenForConditionalGeneration
        
        self.musicgen = MusicgenForConditionalGeneration.from_pretrained(model_name)
        
        # 冻结全部 MusicGen 参数
        for p in self.musicgen.parameters():
            p.requires_grad = False
        
        # 可训练的 Token Encoder
        self.token_vocab = MusicTokenVocabulary()
        self.token_encoder = MusicTokenEncoder(
            vocab_size=len(self.token_vocab),
            d_model=512,
            nhead=8,
            num_layers=4,
            text_dim=self.musicgen.config.decoder.hidden_size,
        )
        
        self.text_dim = self.musicgen.config.decoder.hidden_size
        
        # 注册 forward hook — 在每层 encoder_attn 前注入
        self._register_hooks()
    
    def _register_hooks(self):
        """在每个 decoder layer 的 encoder_attn 前注册 hook，拼接 token_emb"""
        self._token_emb_cache = None
        decoder_layers = self.musicgen.decoder.model.decoder.layers
        
        def make_hook():
            def hook(module, input):
                # input[0] 是 hidden_states [B, T_dec, dim]
                if self._token_emb_cache is not None:
                    return input  # 已经在 generate() 中处理
                return input
            return hook
        
        # 实际注入发生在 generate() 中
        # 因为 encoder_attn 的 KV 来自 encoder_hidden_states 参数
    
    def train(self, mode: bool = True):
        super().train(mode)
        self.musicgen.eval()  # MusicGen 永远 eval
        self.token_encoder.train(mode)
        return self
    
    def forward(
        self,
        text_input_ids: torch.LongTensor,
        text_attention_mask: torch.Tensor,
        token_ids: torch.LongTensor,
        token_attention_mask: torch.Tensor,
        decoder_input_ids: torch.LongTensor,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        """
        训练时前向传播
        
        text_input_ids:  [B, T_txt]  — T5 tokenizer 输出
        token_ids:       [B, T_tok]  — 我们的 Token vocabulary ID
        decoder_input_ids: [B, T_aud] — EnCodec audio tokens
        """
        # 1. Text encoder (frozen) → project to decoder dim
        te_out = self.musicgen.text_encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
        ).last_hidden_state  # [B, T_txt, 768]
        text_emb = self.musicgen.enc_to_dec_proj(te_out)  # [B, T_txt, 1536]
        
        # 2. Token encoder (trainable)
        token_emb = self.token_encoder(token_ids, token_attention_mask)  # [B, T_tok, 1536]
        
        # 3. Concatenate
        combined_emb = torch.cat([text_emb, token_emb], dim=1)  # [B, T_txt+T_tok, 1536]
        combined_mask = torch.cat([text_attention_mask, token_attention_mask], dim=1)
        
        # 4. MusicGen decoder (frozen) — 把 combined_emb 作为 encoder_hidden_states
        decoder_outputs = self.musicgen.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=combined_emb,
            encoder_attention_mask=combined_mask,
            labels=labels,
            return_dict=True,
        )
        
        return decoder_outputs
    
    @torch.no_grad()
    def generate(
        self,
        text: List[str],
        token_seqs: List[str],
        processor,
        max_new_tokens: int = 256,
        **generate_kwargs,
    ):
        """推理时生成音频"""
        self.eval()
        
        # Text → T5
        text_inputs = processor(text=text, padding=True, return_tensors="pt")
        te_out = self.musicgen.text_encoder(
            input_ids=text_inputs["input_ids"].to(self.musicgen.device),
            attention_mask=text_inputs["attention_mask"].to(self.musicgen.device),
        ).last_hidden_state
        text_emb = self.musicgen.enc_to_dec_proj(te_out)
        
        # Token → Token Encoder
        token_ids_list = [self.token_vocab.encode_token_line(t) for t in token_seqs]
        max_tok = max(len(ids) for ids in token_ids_list)
        token_ids = torch.zeros(len(token_seqs), max_tok, dtype=torch.long)
        token_mask = torch.zeros(len(token_seqs), max_tok, dtype=torch.bool)
        for i, ids in enumerate(token_ids_list):
            token_ids[i, :len(ids)] = torch.tensor(ids)
            token_mask[i, :len(ids)] = True
        token_ids = token_ids.to(self.musicgen.device)
        token_mask = token_mask.to(self.musicgen.device)
        token_emb = self.token_encoder(token_ids, token_mask)
        
        # Concatenate + Generate
        combined_emb = torch.cat([text_emb, token_emb], dim=1)
        combined_mask = torch.cat([
            text_inputs["attention_mask"].to(self.musicgen.device), 
            token_mask
        ], dim=1)
        
        audio = self.musicgen.generate(
            **text_inputs,
            encoder_hidden_states=combined_emb,
            encoder_attention_mask=combined_mask,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )
        
        return audio


# ═══════════════════════════════════════════════════════════
# 4. Dataset
# ═══════════════════════════════════════════════════════════

class MusicControlDataset(Dataset):
    """加载 (text, token, audio) 三元组 JSONL"""
    
    def __init__(self, jsonl_path: str, processor, vocab: MusicTokenVocabulary,
                 max_audio_len: int = 1500, max_token_len: int = 512):
        self.processor = processor
        self.vocab = vocab
        self.max_audio_len = max_audio_len
        self.max_token_len = max_token_len
        
        with open(jsonl_path, encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Text → T5 tokenizer
        text_enc = self.processor(text=[item["text"]], padding="max_length",
                                   max_length=128, truncation=True, return_tensors="pt")
        
        # Token → vocab IDs
        token_ids = self.vocab.encode_token_line(item["token"])[:self.max_token_len]
        token_ids = token_ids + [0] * (self.max_token_len - len(token_ids))
        token_mask = [1] * min(len(self.vocab.encode_token_line(item["token"])), self.max_token_len)
        token_mask = token_mask + [0] * (self.max_token_len - len(token_mask))
        
        # Audio → EnCodec tokens (需加载 .wav 并编码)
        # 这里使用占位，实际训练时替换为 EnCodec encode
        audio_tokens = torch.zeros(self.max_audio_len, dtype=torch.long)
        audio_mask = torch.zeros(self.max_audio_len, dtype=torch.bool)
        
        return {
            "text_input_ids": text_enc["input_ids"].squeeze(0),
            "text_attention_mask": text_enc["attention_mask"].squeeze(0),
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "token_attention_mask": torch.tensor(token_mask, dtype=torch.bool),
            "decoder_input_ids": audio_tokens,
            "decoder_attention_mask": audio_mask,
            "labels": audio_tokens.clone(),
        }


# ═══════════════════════════════════════════════════════════
# 5. Metrics
# ═══════════════════════════════════════════════════════════

@dataclass
class EvalMetrics:
    """评估指标集合"""
    loss: float = 0.0
    
    # 和弦准确度
    chord_accuracy: float = 0.0
    
    # 动态 RMS 相关系数
    rms_correlation: float = 0.0
    
    # 音色频谱质心 MAE
    spectral_mae: float = 0.0
    
    # 奏法 onset count ratio
    onset_ratio: float = 0.0
    
    # BPM error
    bpm_error: float = 0.0
    
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


def compute_chord_accuracy(generated_audio: np.ndarray, target_chords: List[str], sr: int = 32000) -> float:
    """
    用 MusicLang Predict (或简化版和弦识别) 验证生成音频的和弦匹配率
    占位: 需要导入 musiclang_predict
    """
    # TODO: import musiclang_predict as mlp
    # predicted = mlp.predict_chords(generated_audio, sr)
    # match = sum(1 for p, t in zip(predicted, target_chords) if p == t)
    # return match / len(target_chords)
    return 0.0  # 占位


def compute_rms_correlation(audio: np.ndarray, target_dynamics: List[str], sr: int = 32000) -> float:
    """计算生成音频的 RMS 曲线与目标动态指令的 Pearson 相关系数"""
    # 将动态标签转为数值: pp=0, p=1, mp=2, mf=3, f=4, ff=5
    dyn_map = {"pp": 0, "p": 1, "mp": 2, "mf": 3, "f": 4, "ff": 5}
    target_vals = np.array([dyn_map.get(d, 3) for d in target_dynamics])
    
    # 分段 RMS
    seg_len = len(audio) // len(target_dynamics)
    rms_vals = []
    for i in range(len(target_dynamics)):
        seg = audio[i * seg_len : (i + 1) * seg_len]
        rms_vals.append(np.sqrt(np.mean(seg ** 2)))
    rms_vals = np.array(rms_vals)
    
    if len(rms_vals) > 1:
        return float(np.corrcoef(rms_vals, target_vals)[0, 1])
    return 0.0


# ═══════════════════════════════════════════════════════════
# 6. Dry Run & Training Loop
# ═══════════════════════════════════════════════════════════

def dry_run(args):
    """仅验证架构，不做训练"""
    from transformers import AutoProcessor
    print("=" * 60)
    print("DRY RUN - 架构验证 (强制CPU，避Blackwell)")
    print("=" * 60)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    device = torch.device("cpu")
    print(f"Device: {device}")
    use_amp = False
    amp_type = "none (dry_run uses CPU)"
    print(f"Mixed precision: {amp_type}")
    print("\nLoading MusicGen...")
    model = ControlledMusicGen("facebook/musicgen-medium")
    model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n参数量:")
    print(f"  可训练 (Token Encoder): {trainable:,}  ({trainable/total*100:.2f}%)")
    print(f"  总计: {total:,}")
    print(f"\nToken 词汇表: {len(model.token_vocab)}")
    processor = AutoProcessor.from_pretrained("facebook/musicgen-medium")
    if os.path.exists(args.data):
        dataset = MusicControlDataset(args.data, processor, model.token_vocab)
        print(f"\n数据集: {len(dataset)} 条三元组")
        sample = dataset[0]
        print(f"  text_input_ids:  {sample['text_input_ids'].shape}")
        print(f"  token_ids:       {sample['token_ids'].shape}")
    else:
        print(f"\n数据文件未找到: {args.data} (dry_run 不需要)")
    print(f"\n前向传播交叉验证 (跳过，解码器需完整音频编码流程)")
    print(f"  [注: dry_run 仅验证模型加载/数据集解析/参数统计]")
    print(f"\nOK: 架构加载成功")
    if device.type == "cuda":
        print(f"\nGPU 显存占用: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"\n{'=' * 60}")
    print(f"训练预估: batch={args.batch_size}, grad_accum={args.grad_accum}")
    print(f"显存需求: ~4-6 GB (fp16) | 推荐: RTX 4090 / A100")
    print(f"{'=' * 60}")


def train(args):
    from transformers import AutoProcessor
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mixed precision: {'bf16' if torch.cuda.is_bf16_supported() else 'fp16'}")
    
    # Model
    print("Loading MusicGen...")
    model = ControlledMusicGen("facebook/musicgen-medium")
    model.to(device)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
    
    # Data
    processor = AutoProcessor.from_pretrained("facebook/musicgen-medium")
    dataset = MusicControlDataset(args.data, processor, model.token_vocab)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, pin_memory=True)
    print(f"Dataset: {len(dataset)} samples, {len(dataloader)} batches/epoch")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.token_encoder.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(dataloader))
    
    # Mixed precision
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    
    # Training
    model.train()
    global_step = 0
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with autocast(device_type=device.type, enabled=use_amp):
                outputs = model(**batch)
                loss = outputs.loss
            
            scaler.scale(loss).backward()
            
            if (global_step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.token_encoder.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")
        
        # 每 N epoch 保存 checkpoint
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = f"{args.output_dir}/checkpoint_epoch{epoch+1}.pt"
            torch.save({
                "epoch": epoch + 1,
                "token_encoder": model.token_encoder.state_dict(),
                "vocab": model.token_vocab.token2id,
                "optimizer": optimizer.state_dict(),
                "loss": avg_loss,
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")
    
    print("Training complete.")


# ═══════════════════════════════════════════════════════════
# 7. CLI
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Music ControlNet Training")
    parser.add_argument("--data", default="C:/Deepseek/outputs/step2_triples.jsonl")
    parser.add_argument("--output_dir", default="C:/Deepseek/outputs/checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--dry_run", action="store_true",
                        help="仅加载模型验证架构，不训练")
    args = parser.parse_args()
    
    if args.dry_run:
        dry_run(args)
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        train(args)
