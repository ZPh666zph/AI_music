import os
#!/usr/bin/env python3
"""
train_moe_controlnet_gpu.py — RTX 5060 8GB 极限优化版
========================================================
策略: bfloat16 autocast + gradient accumulation + full freeze + cache cleanup
"""

import sys, os, json, math, time, random
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 临///时去掉以测试（若GPU可用则注释此行）
if sys.stdout.encoding != "utf-8": sys.stdout.reconfigure(encoding="utf-8")

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ── 设备检测 (Blackwell sm_120 不兼容, 强制 CPU) ──
USE_GPU = True   # RTX 5060 · 若 kernel 不兼容则自动 fallback
DEVICE  = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")
try: DTYPE = torch.bfloat16 if (DEVICE.type=="cuda" and torch.cuda.is_bf16_supported()) else torch.float32
except: DTYPE = torch.float32; USE_GPU = False; DEVICE = torch.device("cpu")
if USE_GPU:
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.benchmark = True
    print(f"GPU: {torch.cuda.get_device_name(0)}  VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB  Dtype: {DTYPE}")
else:
    print(f"CPU mode  Dtype: {DTYPE}")

# ── 超参数 ──
BATCH_SIZE        = 1
GRAD_ACCUM_STEPS  = 8          # effective batch = 1 × 8 = 8
EPOCHS            = int(os.environ.get("GF_MOE_EPOCHS", 50))
LR                = 5e-5
ROUTER_ALPHA      = 0.3        # Router CE Loss 权重
LOG_EVERY         = 10
MAX_SAMPLES       = int(os.environ.get("GF_MOE_MAX_SAMPLES", 668))
JSONL             = os.environ.get("GF_MOE_JSONL", "C:/Deepseek/data/gufeng_moe_train.jsonl")
CHECKPOINT_DIR    = Path(os.environ.get("GF_MOE_CKPT_DIR", "C:/Deepseek/outputs/checkpoints"))
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════
# 1. Token Vocabulary + Dataset
# ═══════════════════════════════════════════════════════════
VOCAB_TOKENS = [
    "[PAD]","[GLOBAL:STYLE:","erhu]","dizi]","suona]",
    "[GLOBAL:EXPERT:","string]","wind]","brass]",
    "[GLOBAL:TEMPO:","[GLOBAL:KEY:","[GLOBAL:TIME:",
    "[SEC:intro|BAR:","[SEC:verse1|BAR:","[SEC:chorus1|BAR:","[SEC:outro|BAR:",
    "[STEM:erhu]","[STEM:dizi]","[STEM:suona]","[/STEM]",
    "[DYN:p|ART:legato|","[DYN:mp|ART:legato|","[DYN:f|ART:marcato|",
    "TEX:sparse|","TEX:medium|","TEX:dense|",
    "TEMPO:","]","|","\n"," ","-",
    "60","70","80","90","100","110","120","130",
]
TOKEN2ID = {t:i for i,t in enumerate(VOCAB_TOKENS)}
VOCAB_SIZE = len(VOCAB_TOKENS)
STYLE_MAP = {"string":0,"wind":1,"brass":2}

def tokenize(text: str, max_len=128) -> torch.Tensor:
    parts = text.replace("\n"," \n ").replace("|"," | ").replace("["," [").replace("]","] ").split()
    ids = [TOKEN2ID.get(p,0) for p in parts]
    ids = ids[:max_len] + [0]*max(0, max_len-len(ids))
    return torch.tensor(ids, dtype=torch.long)

class MoEDataset(Dataset):
    def __init__(self, jsonl_path, max_samples=MAX_SAMPLES):
        self.data = []
        with open(jsonl_path, encoding="utf-8") as f:
            for i,line in enumerate(f):
                if i >= max_samples: break
                self.data.append(json.loads(line))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        r = self.data[idx]
        return {
            "tokens": tokenize(r["tokens"]),
            "style": torch.tensor(STYLE_MAP.get(r["style_expert"],0), dtype=torch.long),
            "text": r["text"],
        }

# ═══════════════════════════════════════════════════════════
# 2. MoE Adapter (Token Encoder + Router + Experts)
# ═══════════════════════════════════════════════════════════

class TokenEncoder(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=512, nhead=8, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, 512, d_model)*0.02)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model,nhead,2048,batch_first=True,activation="gelu",dropout=0.1),
            num_layers
        )
        self.proj = nn.Linear(d_model, 1536)
        self.norm = nn.LayerNorm(1536)

    def forward(self, x):
        x = self.embedding(x) + self.pos[:,:x.size(1),:]
        x = self.transformer(x)
        x = self.norm(self.proj(x))
        return x


class StyleRouter(nn.Module):
    def __init__(self, num_experts=3):
        super().__init__()
        self.embed = nn.Embedding(10, 64)
        self.gate = nn.Sequential(nn.Linear(64,32), nn.GELU(), nn.Linear(32,num_experts))

    def forward(self, style_ids):
        return F.softmax(self.gate(self.embed(style_ids)), dim=-1)

    def forward_logits(self, style_ids):
        """返回 pre-softmax logits 用于 CrossEntropyLoss"""
        return self.gate(self.embed(style_ids))


class ExpertFFN(nn.Module):
    def __init__(self, d_model=64, d_ff=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model,d_ff), nn.GELU(), nn.Dropout(0.1), nn.Linear(d_ff,d_model))
    def forward(self, x): return self.net(x)


class MoEControlNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_encoder = TokenEncoder()
        self.router = StyleRouter()
        self.experts = nn.ModuleList([ExpertFFN() for _ in range(3)])
        self.head = nn.Linear(64, 1)

    def forward(self, token_ids, style_ids):
        tok = self.token_encoder(token_ids)
        tok = tok.mean(dim=1)[:,:64]
        router_logits = self.router.forward_logits(style_ids)  # [B,3] pre-softmax
        w = F.softmax(router_logits, dim=-1)                    # [B,3]
        out = sum(w[:,i:i+1]*expert(tok) for i,expert in enumerate(self.experts))
        return self.head(out).squeeze(-1), w, router_logits              


# ═══════════════════════════════════════════════════════════
# 3. Lite Monitoring
# ═══════════════════════════════════════════════════════════
class LiteLogger:
    def __init__(self, log_file="C:/Deepseek/outputs/training_log.jsonl"):
        self.f = open(log_file, "w", encoding="utf-8")
    def log(self, step, loss, expert_w, lr):
        entry = {"step":step, "loss":round(loss,4), "lr":round(lr,8),
                 "expert_w":[round(float(x),4) for x in expert_w]}
        self.f.write(json.dumps(entry)+"\n")
        self.f.flush()
    def close(self): self.f.close()


# ═══════════════════════════════════════════════════════════
# 4. Training
# ═══════════════════════════════════════════════════════════
def train():
    print("="*60)
    print(f"MoE ControlNet — 极限显存优化训练")
    print(f"  Device: {DEVICE}  Dtype: {DTYPE}")
    print(f"  Batch: {BATCH_SIZE} × GradAccum: {GRAD_ACCUM_STEPS} = effective {BATCH_SIZE*GRAD_ACCUM_STEPS}")
    print("="*60)

    ds = MoEDataset(JSONL)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    steps_per_epoch = (len(ds)//BATCH_SIZE) // GRAD_ACCUM_STEPS
    total_steps = steps_per_epoch * EPOCHS

    model = MoEControlNet()
    model = model.to(DEVICE)

    # FP16/BF16 量化 (节省显存)
    if USE_GPU:
        model = model.to(DTYPE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"\n  Params: trainable={trainable:,} / total={total:,} ({trainable/total*100:.2f}%)")
    print(f"  Steps/epoch: {steps_per_epoch}  Total steps: {total_steps}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, fused=USE_GPU)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, total_steps=total_steps, pct_start=0.1
    )
    scaler = torch.amp.GradScaler("cuda", enabled=USE_GPU)
    logger = LiteLogger()

    print("\n  Training...\n")
    global_step = 0
    best_loss = float("inf")

    pbar = tqdm(total=total_steps, desc="Training", unit="step")
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0
        accum_loss = 0.0

        for batch_idx, batch in enumerate(dl):
            tok = batch["tokens"].to(DEVICE)
            sty = batch["style"].to(DEVICE)

            # 混合精度前向
            with torch.autocast(device_type=DEVICE.type, dtype=DTYPE, enabled=USE_GPU):
                logits, expert_w, router_logits = model(tok, sty)
                # Router CE Loss (直接监督风格分类)
                ce_loss = F.cross_entropy(router_logits, sty)
                # 总 Loss
                loss = F.mse_loss(logits, sty.float()) + ROUTER_ALPHA * ce_loss

            loss = loss / GRAD_ACCUM_STEPS
            scaler.scale(loss).backward()

            accum_loss += loss.item() * GRAD_ACCUM_STEPS

            # 梯度累加
            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                epoch_loss += accum_loss
                global_step += 1
                pbar.update(1)

                # 日志 (每 10 步)
                if global_step % LOG_EVERY == 0:
                    logger.log(global_step, accum_loss, expert_w[0], scheduler.get_last_lr()[0])
                    pbar.set_postfix_str(
                        f"loss={accum_loss:.4f} "
                        f"w=[{expert_w[0,0]:.2f},{expert_w[0,1]:.2f},{expert_w[0,2]:.2f}]"
                    )

                # 清理显存 (8GB 刚需)
                if USE_GPU:
                    torch.cuda.empty_cache()

                accum_loss = 0.0

        avg_epoch_loss = epoch_loss / max(steps_per_epoch, 1)
        print(f"\n  Epoch {epoch+1}/{EPOCHS}  Loss: {avg_epoch_loss:.4f}  Steps: {global_step}")

        # 每 10 epoch 覆盖保存 best_moe_checkpoint.pt
        ckpt = {k:v.cpu() for k,v in model.state_dict().items() if "token_encoder" in k or "router" in k or "experts" in k or "head" in k}
        if (epoch+1) % 10 == 0:
            torch.save(ckpt, CHECKPOINT_DIR / "best_moe_checkpoint.pt")
            print(f"  ★ Saved: best_moe_checkpoint.pt ({len(ckpt)} keys)")
        torch.save(ckpt, CHECKPOINT_DIR / f"moe_epoch_{epoch+1}.pt")

    logger.close()
    pbar.close()

    print(f"\n{'='*60}")
    print(f"Training complete. {global_step} steps, {EPOCHS} epochs.")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()
