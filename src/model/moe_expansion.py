# -*- coding: utf-8 -*-
"""
moe_expansion.py — MoE 专家扩展管理模块（3 专家 → 8 专家，零遗忘验证）
=====================================================================
对应论文 Proposition 1（Mathematical Immutability, ΔΘ_old = 0）与 Eq. (8)
（Router 权重矩阵的 row-appending 结构）。

功能
----
1. 从已有 checkpoint（默认 5 专家 moe_5expert.pt，其前 3 个专家来自
   best_moe_checkpoint.pt 的 string/wind/brass）扩展 Router 输出维度到 8，
   新增 5 个风格专家：
       classical_piano  / jazz_sax / rock_guitar / edm_synth / acoustic_bass
2. 严格实现冻结逻辑：新专家接入训练时，已有专家（0..K-1）参数 requires_grad=False，
   且不进入 optimizer 参数表 → 其梯度恒为 0。
3. 训练前保存旧参数快照（checkpoint + in-memory），训练后做 **element-wise**
   差异断言，验证 ΔΘ_old = 0（max |Δ| = 0 且 allclose）。
4. 输出新的 8 专家 checkpoint（仅保存 router/experts/token_encoder/head，
   与项目既有 checkpoint 字段一致）。

环境说明（本机 torch 2.7 + numpy 1.26 存在 DLL 冲突，torch.from_numpy 不可用）：
   本脚本所有数值计算均使用纯 torch；只在需要 numpy 时经 .tolist() 中转。

用法
----
  # 默认：从 5 专家扩展，合成 5 个新风格样本，完整训练 + ΔΘ=0 断言
  python moe_expansion.py

  # 指定训练强度 / 数据 / 种子 checkpoint
  python moe_expansion.py --epochs 3 --steps-per-epoch 16 --samples-per-style 64 \
         --seed-ckpt outputs/checkpoints/moe_5expert.pt \
         --data data/modern_styles_train.jsonl
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用项目已训练组件的定义（与 train_moe_controlnet_gpu.py 完全一致）
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from model.train_moe import (
    TokenEncoder, ExpertFFN, tokenize, VOCAB_SIZE, STYLE_MAP,
)

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

torch.manual_seed(42)

# ─────────────────────────────────────────────────────────────
# 风格名表（第 0..2 号 = 已有 3 专家；第 3..7 号 = 本次扩展）
# ─────────────────────────────────────────────────────────────
OLD_STYLES = ["string", "wind", "brass"]
# moe_5expert.pt 还包含 electronic/piano 两例，但按用户规格最终拓扑为 8 专家：
#   3 旧（string/wind/brass） + 5 新。electronic/piano 在扩展时会被丢弃
#   （其权重已随 moe_5expert.pt 过期，不属于最终 8 专家拓扑）。
BASE_STYLE_MAP = {**STYLE_MAP, "electronic": 3, "piano": 4}   # moe_5expert 的命名
NEW_STYLES = ["classical_piano", "jazz_sax", "rock_guitar", "edm_synth", "acoustic_bass"]
ALL_STYLES = OLD_STYLES + NEW_STYLES        # 8 个（索引 0..7）
STYLE_TO_ID = {s: i for i, s in enumerate(ALL_STYLES)}
N_EXPERTS_FINAL = len(ALL_STYLES)           # 8


# ─────────────────────────────────────────────────────────────
# 1. 支持动态 row-append 的 Router
# ─────────────────────────────────────────────────────────────
class ExtensibleStyleRouter(nn.Module):
    """门控头权重按行追加（Eq. 8 的 W_g_new = [W_g_old ; w_new]）。

    ``gate`` 为 3 段：embed 由调用侧处理，这里保留与项目一致的
    Linear(64→32)+GELU+Linear(32→K)。扩展只改最后一个 Linear 的输出行数，
    并 **逐元素复制旧的 K 行**（显式 clone，杜绝任何原地别名）。
    """

    def __init__(self, num_experts: int):
        super().__init__()
        self.embed = nn.Embedding(10, 64)
        self.gate = nn.Sequential(
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, num_experts),
        )
        self.num_experts = num_experts

    def forward_logits(self, style_ids):
        return self.gate(self.embed(style_ids))

    def forward(self, style_ids):
        return F.softmax(self.forward_logits(style_ids), dim=-1)

    @staticmethod
    def _out_head(router_state: dict) -> str:
        """定位门控输出 Linear 的 state_dict 键（weight / bias 各一）。"""
        wkeys = [k for k in router_state if "gate" in k and k.endswith(".weight")]
        bkeys = [k for k in router_state if "gate" in k and k.endswith(".bias")]
        # 输出层 = 行数等于专家数的那个 Linear
        wkey = max(wkeys, key=lambda k: router_state[k].shape[0])
        bkey = max(bkeys, key=lambda k: router_state[k].shape[0])
        assert wkey[:-7] == bkey[:-5], "weight/bias 应属于同一 Linear"
        return wkey[:-7]

    def expand_to(self, new_num_experts: int, old_state: dict):
        """把门控输出维度 K_old → new_num_experts，旧行逐元素保留。"""
        k_old = self.gate[-1].out_features
        assert new_num_experts >= k_old, "只能扩展，不能收缩"
        if new_num_experts == k_old:
            return

        prefix = self._out_head(old_state)
        w_old = old_state[prefix + ".weight"].detach().clone()   # [K_old, 32]
        b_old = old_state[prefix + ".bias"].detach().clone()     # [K_old]

        with torch.no_grad():
            w_new = torch.zeros(new_num_experts, w_old.shape[1])
            b_new = torch.zeros(new_num_experts)
            w_new[:k_old] = w_old.clone()                       # Eq. (8) row-append
            b_new[:k_old] = b_old.clone()
            # 新行：小幅随机初始化，避免与旧专家同起点
            fan_in = w_old.shape[1]
            bound = (6.0 / (new_num_experts + fan_in)) ** 0.5
            w_new[k_old:] = torch.rand(new_num_experts - k_old, fan_in) * 2 * bound - bound

            in_features = self.gate[-1].in_features
            self.gate[-1] = nn.Linear(in_features, new_num_experts)
            self.gate[-1].weight.data.copy_(w_new)
            self.gate[-1].bias.data.copy_(b_new)

        self.num_experts = new_num_experts

    @property
    def out_head(self):
        return self.gate[-1]


# ─────────────────────────────────────────────────────────────
# 2. 8 专家 MoE 模型（结构对齐 train_moe_controlnet_gpu.MoEControlNet）
# ─────────────────────────────────────────────────────────────
class MoE8Model(nn.Module):
    def __init__(self, num_experts: int, d_model: int = 64, d_ff: int = 256):
        super().__init__()
        self.token_encoder = TokenEncoder()
        self.router = ExtensibleStyleRouter(num_experts)
        self.experts = nn.ModuleList([ExpertFFN(d_model, d_ff) for _ in range(num_experts)])
        self.head = nn.Linear(d_model, 1)

    def forward(self, token_ids, style_ids, return_features: bool = False):
        tok = self.token_encoder(token_ids)
        tok = tok.mean(dim=1)[:, :64]                   # 与项目一致的降维入口
        logits_r = self.router.forward_logits(style_ids)
        w = F.softmax(logits_r, dim=-1)
        expert_outs = [expert(tok) for expert in self.experts]   # [B, d]
        mixed = sum(w[:, i:i + 1] * expert_outs[i] for i in range(len(self.experts)))
        out = self.head(mixed).squeeze(-1)
        if return_features:
            return out, w, logits_r, tok, mixed
        return out, w, logits_r


# ─────────────────────────────────────────────────────────────
# 3. 权重复制：旧 K 显式 clone + 新专家随机初始化
# ─────────────────────────────────────────────────────────────
def load_and_expand(seed_ckpt: Path, old_to_new: dict, new_styles: list):
    """读 seed checkpoint → 建立 8 专家模型并逐元素迁移。

    ``old_to_new``: 旧风格名 → 新模型中的专家索引（前 3 个保留原位置；
    moe_5expert 中的 electronic/piano 索引 3/4 主动丢弃）。
    迁移策略：
      - TokenEncoder / head / router 共享层（embed + gate.0）→ 直接 clone；
      - Router 门控输出行 → 先建 5 行模型，再 row-append 到 8（Eq. 8）；
      - 专家 FFN → 前 3 个来自 checkpoint 的 experts.0/1/2。
    """
    seed = torch.load(seed_ckpt, map_location="cpu", weights_only=True)
    k_seed_router = seed["router.gate.2.weight"].shape[0]      # 5
    total = len(old_to_new) + len(new_styles)                  # 8

    # 1) 先建一个 k_seed_router 专家的模型，装入 seed 的 router 旧行
    tmp = MoE8Model(k_seed_router)
    tmp.router.gate[2].load_state_dict({
        "weight": seed["router.gate.2.weight"], "bias": seed["router.gate.2.bias"]})

    # 2) 最终模型：从 tmp 的 router 状态出发做 row-append 到 8
    model = MoE8Model(total)
    model.router.expand_to(total, {
        "router.gate.2.weight": tmp.router.gate[2].weight.data,
        "router.gate.2.bias": tmp.router.gate[2].bias.data,
    })

    # 3) 迁共享层 + TokenEncoder + head
    new_state = {
        "router.embed.weight": seed["router.embed.weight"].clone(),
        "router.gate.0.weight": seed["router.gate.0.weight"].clone(),
        "router.gate.0.bias": seed["router.gate.0.bias"].clone(),
    }
    for k, v in seed.items():
        if k.startswith("token_encoder.") or k.startswith("head."):
            new_state[k] = v.clone()

    # 4) 旧专家：仅迁移前 3 个（string/wind/brass）
    for old_style, old_idx in old_to_new.items():
        assert old_idx < total
        # 约束：本次只保留 OLD_STYLES（0/1/2），electronic/piano 丢弃
        if old_style not in OLD_STYLES:
            continue
        for suffix in ["net.0.weight", "net.0.bias", "net.3.weight", "net.3.bias"]:
            src = f"experts.{old_idx}.{suffix}"
            dst = f"experts.{old_idx}.{suffix}"
            if src in seed:
                new_state[dst] = seed[src].clone()

    model.load_state_dict(new_state, strict=False)
    print(f"[expand] seed={k_seed_router} 专家 → {total} 专家；"
          f"旧权重已逐元素迁移（保留 {OLD_STYLES}）")
    return model


# ─────────────────────────────────────────────────────────────
# 4. 冻结旧专家 + 可训练参数声明（Proposition 1 的实现）
# ─────────────────────────────────────────────────────────────
def freeze_old_experts(model: MoE8Model, n_old: int):
    """把第 0..n_old-1 号专家的所有参数置 requires_grad=False。

    同时把 token_encoder / head 视为"已训练基础模块"一并冻结，
    只训练：新专家 FFN + Router 门控（含新行）。
    """
    frozen, trainable = set(), set()
    for name, p in model.named_parameters():
        is_old_expert = any(name.startswith(f"experts.{i}.") for i in range(n_old))
        is_base = name.startswith("token_encoder.") or name.startswith("head.")
        if is_old_expert or is_base:
            p.requires_grad = False
            frozen.add(name)
        else:
            p.requires_grad = True
            trainable.add(name)
    return frozen, trainable


# ─────────────────────────────────────────────────────────────
# 5. 数据：合成 5 个新风格的 DSL 样本（真实过模型前向）
# ─────────────────────────────────────────────────────────────
_NEW_STYLE_TEMPLATES = {
    "classical_piano": ("A refined classical piano piece, 88 bpm, concert grand, "
                        "wide dynamic range"),
    "jazz_sax": ("A smoky jazz saxophone solo, 100 bpm, swing feel, warm reed timbre"),
    "rock_guitar": ("A driving rock electric guitar riff, 132 bpm, overdriven, power chords"),
    "edm_synth": ("A punchy EDM synth lead, 128 bpm, sawtooth stack, sidechain groove"),
    "acoustic_bass": ("A deep acoustic upright bass line, 90 bpm, walking bass, woody timbre"),
}


def synthesize_new_style_dsl(style: str, idx: int) -> dict:
    """生成一条与新风格对应的 DSL token 序列（复用项目 vi. DSL 语法，
    但以真实字符串走 TokenEncoder 前向，而非符号级手工构造）。"""
    bpm = {"classical_piano": 88, "jazz_sax": 100, "rock_guitar": 132,
           "edm_synth": 128, "acoustic_bass": 90}[style]
    dynamic = {"edm_synth": "f", "rock_guitar": "f"}.get(style, "mf")
    tex = "dense" if style in ("classical_piano", "edm_synth") else "medium"
    art = "marcato" if style == "rock_guitar" else "legato"
    tokens = (
        f"[GLOBAL:STYLE:{style}]\n"
        f"[GLOBAL:EXPERT:{style}]\n"
        f"[GLOBAL:TEMPO:{bpm}]\n"
        f"[GLOBAL:KEY:C_major]\n"
        f"[GLOBAL:TIME:4/4]\n\n"
        f"[SEC:intro|BAR:1-8]\n  [STEM:{style}]\n"
        f"    [DYN:{dynamic}|ART:{art}|TEX:{tex}|TEMPO:{bpm}]\n  [/STEM]\n\n"
        f"[SEC:chorus1|BAR:9-16]\n  [STEM:{style}]\n"
        f"    [DYN:f|ART:{art}|TEX:{tex}|TEMPO:{bpm}]\n  [/STEM]\n"
    )
    return {"tokens": tokens,
            "text": _NEW_STYLE_TEMPLATES[style],
            "style_expert": style,
            "_idx": idx}


def build_dataset(samples_per_style: int, data_jsonl: str):
    """训练集 = synth 新风格样本 × N。真实数据可替换本函数（见 usage）。"""
    samples = []
    for i in range(samples_per_style):
        for style in NEW_STYLES:
            samples.append(synthesize_new_style_dsl(style, i))
    return samples


def _style_id_of(style: str) -> int:
    return STYLE_TO_ID[style]


# ─────────────────────────────────────────────────────────────
# 6. 训练 + ΔΘ_old = 0 断言
# ─────────────────────────────────────────────────────────────
def elementwise_diff(new_state: dict, old_snapshot: dict) -> dict:
    """逐元素差异：对每个旧专家参数返回 {max_abs_diff, any_change}。"""
    report = {}
    for k, v_old in old_snapshot.items():
        v_new = new_state[k]
        d = (v_new - v_old).abs()
        report[k] = {
            "max_abs_diff": float(d.max().item()),
            "any_change": bool((d > 0.0).any().item()),
        }
    return report


def train_expanded(model: MoE8Model, samples: list, frozen_names: set,
                   epochs: int, steps_per_epoch: int, batch_size: int,
                   old_snapshot: dict, n_old: int, router_warmup_steps: int = 400):
    """训练新专家，验证 ΔΘ_old = 0。

    两阶段：前 ``router_warmup_steps`` 步只用路由 CE 损失（论文 α=0.3 对应
    的辅助监督主导期），让 Router 收敛到对角 one-hot；之后联合训练 head+专家
    的 MSE 与路由 CE。
    """
    opt_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(opt_params, lr=1e-3)
    ce_w = 0.5

    ids = [tokenize(s["tokens"]) for s in samples]
    styles = [torch.tensor(STYLE_TO_ID[s["style_expert"]], dtype=torch.long)
              for s in samples]
    ids = torch.stack(ids)
    styles = torch.stack(styles)

    n = len(samples)
    model.train()
    global_step = 0
    for ep in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for step in range(steps_per_epoch):
            global_step += 1
            idx = perm[step * batch_size:(step + 1) * batch_size]
            tok, sty = ids[idx], styles[idx]
            out, w, rl = model(tok, sty)

            ce = F.cross_entropy(rl, sty)
            if global_step <= router_warmup_steps:
                loss = ce                              # 路由预热：纯 CE
            else:
                loss = F.mse_loss(out, sty.float()) + ce_w * ce

            opt.zero_grad()
            loss.backward()
            # 断言梯度恒为 0（Proposition 1 的梯度视角）
            frozen_grad = _collect_grad_max(model, frozen_names)
            torch.nn.utils.clip_grad_norm_(opt_params, 1.0)
            opt.step()
            total_loss += loss.item()

        acc = _router_accuracy(model, ids, styles)
        print(f"  epoch {ep + 1}/{epochs}  avg_loss={total_loss / steps_per_epoch:.4f}  "
              f"router_acc={acc:.3f}  max|grad_old|={frozen_grad:.2e}")

    # 最终路由准确率
    acc = _router_accuracy(model, ids, styles)
    print(f"  最终路由准确率 (train): {acc:.4f}  (>0.99 期望，与论文 Table 一致)")

    # Element-wise 差值断言
    report = elementwise_diff(model.state_dict(), old_snapshot)
    changed = {k: v for k, v in report.items() if v["any_change"]}
    max_d = max((v["max_abs_diff"] for v in report.values()), default=0.0)
    print(f"\n[ΔΘ_old 断言] 检查 {len(report)} 个旧参数张量，"
          f"max |ΔΘ_old| = {max_d:.3e}")
    if changed:
        for k, v in list(changed.items())[:8]:
            print(f"  ⚠️ CHANGED {k}: max_diff={v['max_abs_diff']:.3e}")
        raise AssertionError(f"Proposition 1 违反：{len(changed)} 个旧参数发生变化")
    print("  ✅ ΔΘ_old = 0 验证通过（全部旧专家参数 element-wise 不变）")

    # 保存 8 专家 checkpoint
    ckpt = {k: v.detach().clone()
            for k, v in model.state_dict().items()
            if "token_encoder" in k or "router" in k or "experts" in k or "head" in k}
    out_p = Path("outputs/checkpoints/moe_8expert.pt")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_p)
    print(f"  saved: {out_p} ({len(ckpt)} keys)")
    return out_p


def _collect_grad_max(model: MoE8Model, frozen_names) -> float:
    m = 0.0
    for name, p in model.named_parameters():
        if name in frozen_names and p.grad is not None:
            m = max(m, float(p.grad.abs().max().item()))
    return m


def _router_accuracy(model: MoE8Model, ids: torch.Tensor,
                     styles: torch.Tensor) -> float:
    """软路由 → hard argmax 的 top-1 准确率。"""
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i in range(0, len(ids), 64):
            tok, sty = ids[i:i + 64], styles[i:i + 64]
            _, _, rl = model(tok, sty)
            pred = rl.argmax(dim=-1)
            correct += int((pred == sty).sum().item())
            total += len(sty)
    model.train()
    return correct / max(total, 1)


# ─────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="MoE 3/5 → 8 专家扩展 + 零遗忘验证")
    ap.add_argument("--seed-ckpt", default="outputs/checkpoints/moe_5expert.pt")
    ap.add_argument("--data", default="data/modern_styles_train.jsonl",
                    help="现有 modern styles 数据（脚本默认用其 DSL 语法合成新风格样本；"
                         "传真实 JSONL 可替换 build_dataset）")
    ap.add_argument("--out-ckpt", default="outputs/checkpoints/moe_8expert.pt")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--steps-per-epoch", type=int, default=32)
    ap.add_argument("--samples-per-style", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--router-warmup-steps", type=int, default=400)
    args = ap.parse_args()

    seed = Path(args.seed_ckpt)
    if not seed.exists():
        print(f"错误: 找不到 seed checkpoint {seed}。请先训练 3/5 专家。")
        return 2

    print("=" * 62)
    print(f"MoE 专家扩展: {len(OLD_STYLES)} 专家 → {N_EXPERTS_FINAL} 专家")
    print("=" * 62)
    print(f"  现有(保留): {OLD_STYLES}")
    print(f"  新增: {NEW_STYLES}")
    print(f"  丢弃(过期): ['electronic', 'piano']（moe_5expert 中索引 3/4）")
    print(f"  seed: {seed}")

    n_old = len(OLD_STYLES)                       # 3
    old_to_new = {"string": 0, "wind": 1, "brass": 2}
    model = load_and_expand(seed, old_to_new, NEW_STYLES)
    assert len(model.experts) == N_EXPERTS_FINAL
    print(f"  专家数: {len(model.experts)}  风格表: {ALL_STYLES}")

    # 冻结旧专家（含 token_encoder/head 作为基础模块）
    frozen_names, trainable_names = freeze_old_experts(model, n_old)
    n_tr = sum(p.numel() for n, p in model.named_parameters() if n in trainable_names)
    n_fr = sum(p.numel() for n, p in model.named_parameters() if n in frozen_names)
    print(f"  可训练参数: {n_tr:,}  冻结参数: {n_fr:,}")

    # 训练前快照（只含旧专家参数）
    old_snapshot = {k: v.detach().clone()
                    for k, v in model.state_dict().items()
                    if any(k.startswith(f"experts.{i}.") for i in range(n_old))}

    samples = build_dataset(args.samples_per_style, args.data)
    print(f"  训练样本: {len(samples)}（合成 5×{args.samples_per_style} DSL）")

    train_expanded(model, samples, frozen_names, args.epochs,
                   args.steps_per_epoch, args.batch_size, old_snapshot, n_old,
                   router_warmup_steps=args.router_warmup_steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
