#!/usr/bin/env python3
"""train_moe_expand.py — 热插拔验证: 3→5 Experts, 仅训练新专家"""
import sys, os, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
if sys.stdout.encoding != "utf-8": sys.stdout.reconfigure(encoding="utf-8")

import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

torch.manual_seed(42)
DEVICE = torch.device("cpu")
CKPT_OLD = Path("C:/Deepseek/outputs/checkpoints/best_moe_checkpoint.pt")
OUT_CKPT = Path("C:/Deepseek/outputs/checkpoints/moe_5expert.pt")

# ── 共享组件 ──
sys.path.insert(0, ".")
from train_moe_controlnet_gpu import (
    TokenEncoder, StyleRouter, ExpertFFN, MoEControlNet, tokenize, TOKEN2ID,
    VOCAB_SIZE, STYLE_MAP
)

# ═══════════════════════════════════════════════════════════
# 1. 扩展 Router: 3→5 experts
# ═══════════════════════════════════════════════════════════
class ExpandedMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_encoder = TokenEncoder(vocab_size=VOCAB_SIZE, num_layers=4)
        self.router = StyleRouter(num_experts=5)  # 5 experts!
        self.experts = nn.ModuleList([ExpertFFN() for _ in range(5)])
        self.head = nn.Linear(64, 1)

    def forward(self, token_ids, style_ids):
        tok = self.token_encoder(token_ids)
        tok = tok.mean(dim=1)[:, :64]
        router_logits = self.router.forward_logits(style_ids)
        w = F.softmax(router_logits, dim=-1)
        out = sum(w[:, i:i+1] * exp(tok) for i, exp in enumerate(self.experts))
        return self.head(out).squeeze(-1), w, router_logits

# ═══════════════════════════════════════════════════════════
# 2. 加载旧权重 + snapshot
# ═══════════════════════════════════════════════════════════
model = ExpandedMoE()
old_state = torch.load(CKPT_OLD, map_location="cpu", weights_only=True)
model.load_state_dict(old_state, strict=False)  # Expert 3,4 随机初始化

# 快照旧专家权重
snapshot = {}
for k, v in model.state_dict().items():
    if any(f"experts.{i}" in k for i in range(3)):
        snapshot[k] = v.clone()
    if "token_encoder" in k or "head" in k:
        snapshot[k] = v.clone()

# 冻结旧专家 (0,1,2) + token_encoder + head
for name, param in model.named_parameters():
    if any(f"experts.{i}" in name for i in range(3)):
        param.requires_grad = False
    elif "token_encoder" in name or "head" in name:
        param.requires_grad = False
    else:
        param.requires_grad = True  # 仅 Expert 3,4 + Router 可训练

# ═══════════════════════════════════════════════════════════
# 3. Dataset
# ═══════════════════════════════════════════════════════════
STYLE_MAP_EXP = {"string":0,"wind":1,"brass":2,"electronic":3,"piano":4}

class ModernDataset:
    def __init__(self, jsonl_path, max_samples=100):
        self.data = []
        with open(jsonl_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_samples: break
                self.data.append(json.loads(line))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        r = self.data[idx]
        return {
            "tokens": tokenize(r["tokens"]),
            "style": torch.tensor(STYLE_MAP_EXP.get(r["style_expert"], 0), dtype=torch.long),
        }

# ═══════════════════════════════════════════════════════════
# 4. Training
# ═══════════════════════════════════════════════════════════
ds = ModernDataset("C:/Deepseek/data/modern_styles_train.jsonl", max_samples=100)
dl = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

print("="*60)
print("MoE Hot-Plug: 3→5 Experts (Freeze old, Train new)")
print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"  Frozen:    {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")
print(f"  Samples:   {len(ds)}")
print("="*60)

EPOCHS = 5
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch in pbar:
        tok = batch["tokens"].to(DEVICE)
        sty = batch["style"].to(DEVICE)
        _, _, router_logits = model(tok, sty)
        loss = F.cross_entropy(router_logits, sty)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix_str(f"loss={loss.item():.4f}")
    print(f"  Epoch {epoch+1}: avg loss={total_loss/len(dl):.4f}")

# ═══════════════════════════════════════════════════════════
# 5. 验证：旧专家权重未变化
# ═══════════════════════════════════════════════════════════
print("\n--- 热插拔完整性验证 ---")
for k, v_old in snapshot.items():
    v_new = model.state_dict()[k]
    if not torch.allclose(v_old, v_new, atol=1e-6):
        print(f"  ⚠ CHANGED: {k}")
        break
else:
    print("  ✓ 全部 3 个旧专家权重未变化 — Zero Forgetting 验证通过!")

# 保存
ckpt = {k:v for k,v in model.state_dict().items() if any(
    x in k for x in ["token_encoder","router","experts","head"])}
torch.save(ckpt, OUT_CKPT)
print(f"\n  Saved: {OUT_CKPT} ({len(ckpt)} keys)")

# Router 测试
print("\n--- 5-Expert Router Test ---")
for style in ["string","wind","brass","electronic","piano"]:
    sid = torch.tensor([STYLE_MAP_EXP[style]])
    with torch.no_grad():
        _, w, _ = model(tokenize(ds[0]["tokens"]).unsqueeze(0), sid)
    eid = torch.argmax(w, dim=1).item()
    names = ["string","wind","brass","electronic","piano"]
    print(f"  {style:12s} → Expert {eid} ({names[eid]})  w=[{w[0,0]:.3f},{w[0,1]:.3f},{w[0,2]:.3f},{w[0,3]:.3f},{w[0,4]:.3f}]")
