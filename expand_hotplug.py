import sys,os; os.environ["CUDA_VISIBLE_DEVICES"]=""; sys.path.insert(0,".")
import torch, torch.nn as nn, torch.nn.functional as F, json
from train_moe_controlnet_gpu import TokenEncoder, ExpertFFN, tokenize, VOCAB_SIZE
from pathlib import Path; from tqdm import tqdm
torch.manual_seed(42)

class StyleRouter5(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 64)
        self.gate = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 5))
    def forward(self, x): return F.softmax(self.gate(self.embed(x)), dim=-1)
    def forward_logits(self, x): return self.gate(self.embed(x))

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_encoder = TokenEncoder(vocab_size=VOCAB_SIZE, num_layers=4)
        self.router = StyleRouter5()
        self.experts = nn.ModuleList([ExpertFFN() for _ in range(5)])
        self.head = nn.Linear(64, 1)
    def forward(self, tk, sty):
        x = self.token_encoder(tk).mean(1)[:, :64]
        rl = self.router.forward_logits(sty)
        w = F.softmax(rl, dim=-1)
        out = sum(w[:, i:i+1] * e(x) for i, e in enumerate(self.experts))
        return self.head(out).squeeze(-1), w, rl

model = M()
old = torch.load("outputs/checkpoints/best_moe_checkpoint.pt", map_location="cpu", weights_only=True)

# 手动迁移权重
merged = {}
for k, v in model.state_dict().items():
    if k in old:
        if k == "router.gate.2.weight":
            nw = v.clone(); nw[:3] = old[k]; merged[k] = nw
        elif k == "router.gate.2.bias":
            nb = v.clone(); nb[:3] = old[k]; merged[k] = nb
        elif any(f"experts.{i}" in k for i in range(3)) or "token_encoder" in k or "head" in k:
            merged[k] = old[k]
        elif "router.gate.0" in k or "router.embed" in k or "router.gate.1" in k:
            merged[k] = old[k]
        else:
            merged[k] = v
    else:
        merged[k] = v
model.load_state_dict(merged, strict=True)

# 冻结旧专家
snap = {k: v.clone() for k, v in model.state_dict().items()
        if any(f"experts.{i}" in k for i in range(3)) or "token_encoder" in k or "head" in k}
for n, p in model.named_parameters():
    p.requires_grad = not any(f"experts.{i}" in n for i in range(3)) and "token_encoder" not in n and "head" not in n

# 加载数据
SM = {"string": 0, "wind": 1, "brass": 2, "electronic": 3, "piano": 4}
with open("C:/Deepseek/data/modern_styles_train.jsonl", encoding="utf-8") as f:
    raw = [json.loads(l) for l in f.readlines()[:100]]

class DS:
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        return {"tokens": tokenize(self.data[i]["tokens"]),
                "style": torch.tensor(SM.get(self.data[i]["style_expert"], 0), dtype=torch.long)}

ds = DS(raw)
dl = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
fr = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"Trainable: {tr:,}  Frozen: {fr:,}  Samples: {len(ds)}")

for ep in range(5):
    model.train(); tl = 0
    for b in tqdm(dl, desc=f"Epoch {ep+1}/5"):
        tk, sty = b["tokens"], b["style"]
        _, _, rl = model(tk, sty)
        loss = F.cross_entropy(rl, sty)
        opt.zero_grad(); loss.backward(); opt.step()
        tl += loss.item()
    print(f"  Epoch {ep+1}: avg loss={tl/len(dl):.4f}")

print("\n--- Zero Forgetting ---")
for k, vo in snap.items():
    if not torch.allclose(vo, model.state_dict()[k], atol=1e-6):
        print(f"  CHANGED: {k}"); break
else:
    print("  All 3 old experts unchanged!")

OUT = Path("C:/Deepseek/outputs/checkpoints/moe_5expert.pt")
torch.save({k: v for k, v in model.state_dict().items()
            if any(x in k for x in ["router", "experts", "token_encoder", "head"])}, OUT)
print(f"Saved: {OUT}")

print("\n--- 5-Expert Router ---")
names = ["string", "wind", "brass", "electronic", "piano"]
for style in names:
    sid = torch.tensor([SM[style]])
    _, w, _ = model(ds[0]["tokens"].unsqueeze(0), sid)
    eid = torch.argmax(w, dim=1).item()
    ws = [f"{w[0,i]:.3f}" for i in range(5)]
    print(f"  {style:12s} -> Expert {eid} ({names[eid]})  w={ws}")
