#!/usr/bin/env python3
"""
train_moe_controlnet.py — MoE ControlNet 初次通电 (1 Epoch 冒烟)
===============================================================
架构: Frozen MusicGen (1.5B) + Trainable Token Encoder + MoE Router + 3 Experts
数据集: gufeng_moe_train.jsonl (string/wind/brass)
"""

import sys, os, json, random, math
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU 模式
if sys.stdout.encoding != "utf-8": sys.stdout.reconfigure(encoding="utf-8")

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

torch.manual_seed(42)
DEVICE = torch.device("cpu")

# ═══════════════════════════════════════════════════════════
# 1. Token 词汇表
# ═══════════════════════════════════════════════════════════
VOCAB_TOKENS = [
    "[PAD]", "[GLOBAL:STYLE:", "erhu]", "dizi]", "suona]",
    "[GLOBAL:EXPERT:", "string]", "wind]", "brass]",
    "[GLOBAL:TEMPO:", "[GLOBAL:KEY:", "[GLOBAL:TIME:",
    "[SEC:intro|BAR:", "[SEC:verse1|BAR:", "[SEC:chorus1|BAR:", "[SEC:outro|BAR:",
    "[STEM:erhu]", "[STEM:dizi]", "[STEM:suona]", "[/STEM]",
    "[DYN:p|ART:legato|", "[DYN:mp|ART:legato|", "[DYN:f|ART:marcato|",
    "[DYN:p|ART:marcato|", "TEX:sparse|", "TEX:medium|", "TEX:dense|",
    "TEMPO:", "]", "|", "\n", " ", "-",
    "60","70","80","90","100","110","120","130",
]
token_to_id = {t: i for i, t in enumerate(VOCAB_TOKENS)}
VOCAB_SIZE = len(VOCAB_TOKENS)

def tokenize(text: str, max_len=128) -> torch.Tensor:
    """简单 tokenize: 将 Token 字符串拆分为子 token, 查表"""
    ids = []
    # 用 split 按自然边界切分
    parts = text.replace("\n"," \n ").replace("|"," | ").replace("["," [").replace("]","] ").split()
    for p in parts:
        ids.append(token_to_id.get(p, 0))
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    return torch.tensor(ids[:max_len], dtype=torch.long)


# ═══════════════════════════════════════════════════════════
# 2. Dataset
# ═══════════════════════════════════════════════════════════
class MoEDataset(Dataset):
    def __init__(self, jsonl_path: str, max_samples=200):
        self.data = []
        with open(jsonl_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples: break
                self.data.append(json.loads(line))
        print(f"  Loaded {len(self.data)} samples")

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        r = self.data[idx]
        tokens = tokenize(r["tokens"])
        style = {"string":0,"wind":1,"brass":2}.get(r["style_expert"],0)
        return {
            "tokens": tokens,
            "style_expert": torch.tensor(style, dtype=torch.long),
            "text": r["text"],
        }


# ═══════════════════════════════════════════════════════════
# 3. MoE 组件
# ═══════════════════════════════════════════════════════════

class TokenEncoder(nn.Module):
    """4层 Transformer · 13.4M 参数"""
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=512, nhead=8, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=2048,
            batch_first=True, activation="gelu", dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.proj = nn.Linear(d_model, 1536)

    def forward(self, token_ids):
        x = self.embedding(token_ids)           # [B, T, 512]
        x = x + self.pos[:, :x.size(1), :]      # positional
        x = self.transformer(x)                  # [B, T, 512]
        x = self.proj(x)                         # [B, T, 1536]
        return x


class StyleRouter(nn.Module):
    """基于 [STYLE] Token 的显式路由 · 3 Expert"""
    def __init__(self, num_experts=3, hidden_dim=1536):
        super().__init__()
        self.num_experts = num_experts
        self.style_embed = nn.Embedding(10, 256)
        self.gate = nn.Sequential(
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, num_experts),
        )

    def forward(self, style_ids):
        s = self.style_embed(style_ids)   # [B, 256]
        logits = self.gate(s)              # [B, 3]
        return F.softmax(logits, dim=-1)


class MoEExperts(nn.Module):
    """3 个 FFN Expert"""
    def __init__(self, num_experts=3, hidden_dim=64, ffn_dim=256):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, hidden_dim))
            for _ in range(num_experts)
        ])

    def forward(self, x, weights):
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            out += weights[:, i:i+1] * expert(x)
        return out


class MoEControlNet(nn.Module):
    """完整的 MoE ControlNet 模型"""
    def __init__(self):
        super().__init__()
        self.token_encoder = TokenEncoder()
        self.router = StyleRouter()
        self.experts = MoEExperts(hidden_dim=64)  # 小规模测试
        self.fc = nn.Linear(64, 1)  # dummy output head

    def forward(self, token_ids, style_ids):
        # Token Encoder
        tok_emb = self.token_encoder(token_ids)    # [B, T, 1536]
        tok_avg = tok_emb.mean(dim=1)          # [B, 1536]

        # 降维到 64 做 MoE 测试
        tok_small = tok_avg[:, :64]             # [B, 64]

        # Router
        expert_w = self.router(style_ids)       # [B, 3]

        # Experts
        expert_out = self.experts(tok_small, expert_w)  # [B, 64]

        # Loss head (简化: 回归到 style_id)
        logits = self.fc(expert_out)             # [B, 1]
        return logits, expert_w


# ═══════════════════════════════════════════════════════════
# 4. 训练
# ═══════════════════════════════════════════════════════════
def train():
    print("=" * 60)
    print("MoE ControlNet — 初次通电 (1 Epoch CPU)")
    print("=" * 60)

    # Config
    BATCH_SIZE = 4
    EPOCHS = 1
    LR = 1e-4
    JSONL = "C:/Deepseek/data/gufeng_moe_train.jsonl"

    # Dataset
    ds = MoEDataset(JSONL, max_samples=100)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    # Model
    model = MoEControlNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n  模型参数: 可训练 {trainable:,} / 总计 {total:,} ({trainable/total*100:.2f}%)")
    print(f"  数据集:   {len(ds)} 条")
    print(f"  Batch:    {BATCH_SIZE}")
    print(f"  Device:   {DEVICE}")
    print()

    # Training loop
    model.train()
    pbar = tqdm(dl, desc=f"Epoch 1/{EPOCHS}", unit="batch")
    total_loss = 0.0

    for batch in pbar:
        tok = batch["tokens"].to(DEVICE)
        sty = batch["style_expert"].to(DEVICE)

        logits, expert_w = model(tok, sty)
        loss = criterion(logits.squeeze(-1), sty.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix_str(f"loss={loss.item():.4f}  w={expert_w[0].tolist()}")

    avg_loss = total_loss / len(dl)
    print(f"\n{'='*60}")
    print(f"训练完成!")
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  可训练参数:  {trainable:,}")
    print(f"{'='*60}")

    # 打印一次前向传播的 Router 输出
    print(f"\n  Router 激活样例 (batch[0]):")
    print(f"    输入 style={sty[0].item()} → expert_w={expert_w[0].tolist()}")
    print(f"    (0=string, 1=wind, 2=brass)")

    return avg_loss


if __name__ == "__main__":
    train()
