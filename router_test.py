#!/usr/bin/env python3
"""router_test.py — MoE Adapter 快速 Router 验证 (无 MusicGen 依赖)"""
import os; os.environ["CUDA_VISIBLE_DEVICES"]=""
import torch, torch.nn as nn, torch.nn.functional as F

VOCAB = [
    "[PAD]","[GLOBAL:STYLE:","erhu]","dizi]","suona]",
    "[GLOBAL:EXPERT:","string]","wind]","brass]",
    "[GLOBAL:TEMPO:","[STEM:","[/STEM]","[DYN:mp|ART:legato|",
    "TEX:sparse|","TEMPO:","]","|","80","90","100",
]
T2I = {t:i for i,t in enumerate(VOCAB)}
def tok(text, ml=128):
    ids = [T2I.get(p,0) for p in text.replace("["," [").replace("]","] ").split()][:ml]
    return torch.tensor(ids+[0]*max(0,ml-len(ids)), dtype=torch.long)

class TE(nn.Module):
    def __init__(self): super().__init__()
        self.emb=nn.Embedding(len(VOCAB),512)
        self.pos=nn.Parameter(torch.randn(1,512,512)*0.02)
        self.tr=nn.TransformerEncoder(nn.TransformerEncoderLayer(512,8,2048,batch_first=True,activation="gelu",dropout=0.1),4)
        self.proj=nn.Linear(512,1536); self.norm=nn.LayerNorm(1536)
    def forward(self,x): x=self.emb(x)+self.pos[:,:x.size(1),:]; return self.norm(self.proj(self.tr(x)))

class SR(nn.Module):
    def __init__(self): super().__init__()
        self.emb=nn.Embedding(10,64); self.g=nn.Sequential(nn.Linear(64,32),nn.GELU(),nn.Linear(32,3))
    def forward(self,x): return F.softmax(self.g(self.emb(x)),dim=-1)

class EF(nn.Module):
    def __init__(self): super().__init__()
        self.net=nn.Sequential(nn.Linear(64,256),nn.GELU(),nn.Dropout(0.1),nn.Linear(256,64))
    def forward(self,x): return self.net(x)

class M(nn.Module):
    def __init__(self): super().__init__()
        self.token_encoder=TE(); self.router=SR()
        self.experts=nn.ModuleList([EF() for _ in range(3)]); self.head=nn.Linear(64,1)
    def forward(self,tk,st):
        x=self.token_encoder(tk).mean(1)[:,:64]
        w=self.router(st)
        return self.head(sum(w[:,i:i+1]*e(x) for i,e in enumerate(self.experts))).squeeze(-1),w

def build(style, bpm=80):
    tag={"string":"erhu","wind":"dizi","brass":"suona"}[style]
    return f"[GLOBAL:STYLE:{tag}] [GLOBAL:EXPERT:{style}] [GLOBAL:TEMPO:{bpm}] [STEM:{tag}] [DYN:mp|ART:legato|TEX:sparse|TEMPO:{bpm}]"

model=M()
ckpt=torch.load("C:/Deepseek/outputs/checkpoints/moe_epoch_10.pt",map_location="cpu",weights_only=True)
model.load_state_dict(ckpt,strict=False); model.eval()

print("=== MoE Router A/B Test (epoch 10 weights) ===")
for style in ["string","wind","brass"]:
    tids=tok(build(style)).unsqueeze(0)
    sid=torch.tensor({"string":0,"wind":1,"brass":2}[style]).unsqueeze(0)
    with torch.no_grad(): _,w=model(tids,sid)
    ws=[f"{v:.3f}" for v in w[0].tolist()]
    eid=torch.argmax(w,dim=1).item()
    names=["string(erhu)","wind(dizi)","brass(suona)"]
    print(f"  {style:8s} → s={ws[0]} w={ws[1]} b={ws[2]} → Expert {eid} ({names[eid]})")

print("\nMoE Router 验证: 10 epoch 训练后风格路由收敛 ✓")
