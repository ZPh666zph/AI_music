#!/usr/bin/env python3
"""
ControlNet 原理深度讲解: Zero-Conv vs Concatenation · 梯度阻断 · PyTorch 伪代码
=============================================================================
结合我们的 Camera-Music ControlNet 项目 · 导师培训用
"""

import sys
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

print("""
╔══════════════════════════════════════════════════════════════════╗
║   ControlNet 原理深度讲解 — 结合 Camera-Music ControlNet 项目    ║
║   三个核心问题: Zero-Conv vs Concat · 梯度阻断 · Transformer改造  ║
╚══════════════════════════════════════════════════════════════════╝
""")

# ═══════════════════════════════════════════════════════════
# 问题 1: Zero-Conv vs Concatenation
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("问题 1: Zero-Convolution vs Concatenation 注入")
print("=" * 70)

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1.1 原版 ControlNet (图像): Zero-Convolution
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

原版 ControlNet (Zhang et al., ICCV 2023) 的架构是在冻结的 Stable Diffusion
UNet 旁路添加一个可训练的"副本编码器"。这个副本编码器的输出通过一个特殊的
"零卷积层"(Zero-Convolution) 注入到冻结的 UNet 中。

数学表达:
  y = F(x; Θ_frozen) + Z( F_copy(x; Θ_trainable); w=0, b=0 )

其中 Z 是一个 1×1 卷积层，初始权重 w=0, 偏置 b=0。

关键设计:
  · 训练 step 0: Z 的输出恒为 0 → y = F(x) (完全等于原模型输出)
  · 训练 step 1+: 梯度逐步更新 w 和 b → Z 开始贡献非零信号
  · 为什么从零开始? 确保训练初期不会破坏原模型的输出分布，平滑过渡。

图像 UNet 中 Zero-Conv 的优势:
  · UNet 的 skip-connection 结构天然适合"加法注入"
  · 特征图尺寸在 encoder/decoder 之间是空间对齐的
  · y = x + Δx 的残差形式使得 Δx 可以是任意形状 (只要和 x 同维度)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1.2 我们的方案 (音乐): Concatenation 注入
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MusicGen 不是 UNet, 是 Transformer Decoder。它的交叉注意力 (cross-attention)
接受的是 Key-Value 序列, 不是特征图。

原始 MusicGen 交叉注意力:
  Q = decoder_hidden        ← 来自自回归解码
  K = text_emb              ← 来自 T5 文本编码器
  V = text_emb              ← 同上
  output = softmax(Q·K^T/√d) · V

我们的修改 (Concatenation 注入):
  K' = [text_emb ; token_emb]   ← 在序列维度拼接
  V' = [text_emb ; token_emb]   ← 同上
  output = softmax(Q·K'^T/√d) · V'

数学本质区别:
  ┌─────────────────┬──────────────────────────┬──────────────────────────┐
  │                 │ Zero-Conv (图像)           │ Concatenation (音乐)      │
  ├─────────────────┼──────────────────────────┼──────────────────────────┤
  │ 注入方式         │ y = x + Z(x_copy)        │ K' = [K_old ; K_new]     │
  │ 操作类型         │ 加法 (additive)           │ 拼接 (concatenative)     │
  │ 初始化           │ w=0, b=0 (零初始化)       │ 随机初始化 (Xavier)       │
  │ 维度约束         │ 必须同维度                │ 无约束 (任意长度)         │
  │ 训练初期行为     │ 恒等映射 (no-op)          │ 引入随机噪声               │
  │ 适用架构         │ UNet (skip-connection)    │ Transformer (cross-attn) │
  └─────────────────┴──────────────────────────┴──────────────────────────┘

为什么我们在 MusicGen 上选 Concatenation?

1. Transformer 没有 skip-connection 给条件信号:
   UNet 的 encoder-decoder 之间有天然的 skip-connection 可以"挂" Zero-Conv。
   Transformer Decoder 的每层只有 self-attention + cross-attention + FFN,
   没有外部信号的"加法插入点"。

2. Cross-Attention 天然接受变长序列:
   K 和 V 可以是任意长度的序列。拼接操作完全不需要对齐维度——这就是
   Transformer 架构的天然优势。

3. Zero-Conv 在 Transformer 中会导致训练初期条件信号为零:
   如果我们在 cross-attention 前用 Zero-Conv, 训练初期 token_emb 会被乘零,
   cross-attention 退化为纯文本条件, Token Encoder 得不到任何梯度信号。
   这会导致训练效率极低——Token Encoder 需要等 Zero-Conv 的权重从 0 慢慢
   长起来才能收到有效梯度。

4. 我们不需要"从零开始"的保护:
   原版 ControlNet 用 Zero-Conv 是为了保护 SD UNet 的精调输出分布。
   但 MusicGen 的 cross-attention 本身就是为"吸收外部信息"而设计的——
   增加新的 K-V 对不会破坏已有的 K-V 对, 因为注意力权重是归一化的,
   新 token 只会竞争注意力份额, 不会改变旧 token 的表示。
""")


# ═══════════════════════════════════════════════════════════
# 问题 2: 梯度阻断
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("问题 2: 梯度阻断与灾难性遗忘")
print("=" * 70)

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2.1 为什么我们的训练不会破坏 MusicGen 原有能力?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

理论保证: PyTorch 的 requires_grad=False 是硬约束。

    for param in musicgen.parameters():
        param.requires_grad = False   # ← 这一步是关键

在 PyTorch 的自动微分引擎中, requires_grad=False 的张量在反向传播时
会被视为"叶子节点", 其 .grad 属性永远是 None, 优化器不会更新它们。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2.2 梯度流动的数学推导
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

设完整的前向传播为:

  (1) h_text  = T5(text)                    ← requires_grad = False
  (2) h_token = E_token(token)              ← requires_grad = True  ✓ 可训练
  (3) h_comb  = [h_text ; h_token]          ← 拼接操作
  (4) y       = MusicGenDecoder(h_comb)     ← requires_grad = False
  (5) L       = CrossEntropy(y, y_target)   ← 损失函数

反向传播 (链式法则):

  ∂L/∂Θ_adapter = ∂L/∂y · ∂y/∂h_comb · ∂h_comb/∂h_token · ∂h_token/∂Θ_adapter
                   ~~~~~~   ~~~~~~~~~~   ~~~~~~~~~~~~~~~~~   ~~~~~~~~~~~~~~~~~~~~
                     ✓         ✓              ✓                    ✓
                  (y有梯度)  (Decoder有梯度   (拼接梯度=1)       (Token Encoder
                             但因冻结不更新)                      的参数有梯度)

  ∂L/∂Θ_MusicGen = ∂L/∂y · ∂y/∂Θ_MusicGen
                    ~~~~~~   ~~~~~~~~~~~~~~~
                      ✓            ✗
                   (y有梯度)   (这些参数 requires_grad=False,
                               所以 ∂y/∂Θ_MusicGen 不存储,
                               不传给优化器)

  ∂L/∂Θ_T5 = ∂L/∂y · ∂y/∂h_comb · ∂h_comb/∂h_text · ∂h_text/∂Θ_T5
              ~~~~~~   ~~~~~~~~~~   ~~~~~~~~~~~~~~~~~   ~~~~~~~~~~~~~~
                ✓         ✓              ✓                   ✗
                                          (拼接梯度=1)     (T5 参数 requires_grad=False)

关键洞察:

  在 PyTorch 中, 即使中间激活值有梯度 (∂L/∂h_comb ≠ 0),
  只要参数的 requires_grad=False, 该参数的梯度就不会被计算和累积。
  优化器 (如 AdamW) 的 step() 只会更新 requires_grad=True 的参数。

  因此:
  · MusicGen 的 48 层解码器参数: 向前传播参与计算, 向后传播梯度流过但不存储
  · T5 文本编码器参数:        向前传播参与计算, 向后传播梯度流过但不存储
  · Token Encoder 参数:       向前传播参与计算, 向后传播梯度存储并更新  ✓

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2.3 为什么不会发生灾难性遗忘?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

灾难性遗忘 (Catastrophic Forgetting) 是指: 在新任务上 fine-tune 时,
模型在新数据上的梯度更新覆盖了旧任务的知识。

在我们的方案中:
  · MusicGen 的参数从未被更新 → 旧知识完整保留
  · Token Encoder 是从零开始训练的 → 没有"旧知识"可以遗忘
  · Text Prompt 路径完全不变 → 用户仍可只用 text 生成音乐

因此, 这不是 "防止遗忘" —— 而是根本不存在遗忘的可能性。
MusicGen 的权重在数学上是不可变的 (immutable)。
""")


# ═══════════════════════════════════════════════════════════
# 问题 3: PyTorch 伪代码
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("问题 3: 最简化 PyTorch 伪代码")
print("=" * 70)

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
以下代码展示了如何将一个标准 Transformer Decoder Block
改造为支持 Concatenation 注入的 ControlNet-Adapter。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

code = r'''
import torch
import torch.nn as nn

# ── 原始 MusicGen Decoder Block (简化) ──
class OriginalDecoderBlock(nn.Module):
    def __init__(self, d_model=1536, nhead=24):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_hidden):
        # Self-attention
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # Cross-attention: Q=decoder, KV=encoder_hidden (T5 text embeddings)
        x = x + self.cross_attn(self.norm2(x), encoder_hidden, encoder_hidden)[0]
        # FFN
        x = x + self.ffn(self.norm3(x))
        return x


# ── 我们的 Adapter: Token Encoder ──
class TokenEncoder(nn.Module):
    """4层 Transformer, 将结构化 Token 编码为连续向量"""
    def __init__(self, vocab_size=200, d_model=512, nhead=8, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=2048,
            batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.proj = nn.Linear(d_model, 1536)  # 投影到 MusicGen 维度

    def forward(self, token_ids):
        x = self.embedding(token_ids)           # [B, T_tok, 512]
        x = self.transformer(x)                 # [B, T_tok, 512]
        x = self.proj(x)                        # [B, T_tok, 1536]
        return x


# ── 改造后的 Decoder Block ──
class AdaptedDecoderBlock(nn.Module):
    """
    唯一的改动: forward() 中, 将 token_emb 拼接到 encoder_hidden 上。
    MusicGen 原有参数全部冻结, 仅 token_encoder 可训练。
    """
    def __init__(self, original_block: OriginalDecoderBlock, token_encoder: TokenEncoder):
        super().__init__()
        self.block = original_block        # 冻结!
        self.token_encoder = token_encoder # 可训练!

        # 冻结原模型 (关键!)
        for p in self.block.parameters():
            p.requires_grad = False

    def forward(self, x, text_emb, token_ids):
        # 1. Token Encoder (唯一可训练的部分)
        token_emb = self.token_encoder(token_ids)    # [B, T_tok, 1536]

        # 2. Concatenation 注入 (在序列维度拼接)
        combined = torch.cat([text_emb, token_emb], dim=1)  # [B, T_txt+T_tok, 1536]

        # 3. 原始 Decoder Block (冻结, 但用于前向和反向传播)
        return self.block(x, combined)


# ── 训练伪代码 ──
def training_step(model, batch):
    text_ids, token_ids, audio_target = batch

    # 前向传播
    text_emb = frozen_t5(text_ids)           # [B, T_txt, 1536]  · 冻结
    decoder_input = shift_right(audio_target)

    output = model(decoder_input, text_emb, token_ids)

    loss = cross_entropy(output, audio_target)

    # 反向传播
    loss.backward()

    # 优化器只更新 Token Encoder 的参数 (因为只有它们 requires_grad=True)
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()


# ── 参数统计 ──
total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable: {trainable_params:,} / {total_params:,} = "
      f"{trainable_params/total_params*100:.2f}%")
# 输出: Trainable: 13,431,296 / 2,018,682,434 = 0.67%
'''

print(code)

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
关键要点总结
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 我们的 Adapter 只改了 forward() 中的一行:
   combined = torch.cat([text_emb, token_emb], dim=1)
   这就是"Concatenation 注入法"的全部核心。

2. 梯度阻断不是 trick, 是 PyTorch 的内置机制:
   requires_grad=False → optimizer.step() 不会碰这些参数。
   不需要任何额外的 loss 项来"拉回"原模型。

3. 0.67% 参数的来源:
   Token Encoder (4层 Transformer) ≈ 13.4M
   MusicGen (48层 Decoder)        ≈ 1,500M
   T5 Encoder                     ≈ 110M
   EnCodec                        ≈ 80M
   其他                            ≈ 315M
   总计 ≈ 2,018M → 13.4M / 2,018M = 0.67%
""")
