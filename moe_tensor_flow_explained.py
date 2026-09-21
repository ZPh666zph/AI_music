#!/usr/bin/env python3
"""MoE ControlNet 张量流动 + Concatenation 控制原理 · 完整讲解"""
import sys
if sys.stdout.encoding != "utf-8": sys.stdout.reconfigure(encoding="utf-8")

print("""
╔══════════════════════════════════════════════════════════════════╗
║  MoE ControlNet 硬核解析: 张量流动 + Concatenation 控制原理       ║
║  基于 train_moe_controlnet.py 的逐行解构                         ║
╚══════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
问题 1: 完整张量维度变换图
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [Token 字符串]
  "[STYLE:erhu] [GLOBAL:TEMPO:120] ..."
        │
        ▼ tokenize() ──────────────────────────────────────
  [B, T] = [4, 128]        dtype: torch.long
    B = batch_size (4)
    T = max_token_len (128, padded with 0)
        │
        ▼ TokenEncoder.forward()
  ┌─────────────────────────────────────────────────────────┐
  │ 1. self.embedding(token_ids)                           │
  │    [4, 128] → [4, 128, 512]   vocab_size=100, d_model=512│
  │                                                        │
  │ 2. + positional encoding                               │
  │    [4, 128, 512]    (add, not concat)                  │
  │                                                        │
  │ 3. self.transformer (4-layer TransformerEncoder)       │
  │    [4, 128, 512] → [4, 128, 512]                       │
  │    每层: self-attn + FFN, batch_first=True              │
  │                                                        │
  │ 4. self.proj (Linear 512→1536)                         │
  │    [4, 128, 512] → [4, 128, 1536]                      │
  └─────────────────────────────────────────────────────────┘
        │
        ▼ .mean(dim=1)   (平均池化, 压缩时间维度)
  [B, D] = [4, 1536]
        │
        ▼ 切片 [:,:64]  (降维到 64 做小规模 MoE 测试)
  [B, d] = [4, 64]       ← 进入 MoE 的向量
        │
        ├──────────────────────────────────────────────────┐
        │                                                  │
        ▼ StyleRouter.forward(style_ids)                   │
  ┌──────────────────────────────────────────────┐         │
  │ style_ids: [4] = [0, 1, 2, 0]               │         │
  │   (0=string, 1=wind, 2=brass)                │         │
  │                                              │         │
  │ 1. self.style_embed(style_ids)               │         │
  │    [4] → [4, 256]                            │         │
  │                                              │         │
  │ 2. self.gate (Linear(256→128) + GELU         │         │
  │             + Linear(128→3))                 │         │
  │    [4, 256] → [4, 128] → [4, 3]             │         │
  │                                              │         │
  │ 3. softmax(dim=-1)                           │         │
  │    [4, 3]  (每行之和=1.0)                     │         │
  │    例: [[0.40, 0.24, 0.36],                  │         │
  │         [0.32, 0.28, 0.40],                  │         │
  │         [0.29, 0.33, 0.38],                  │         │
  │         [0.42, 0.22, 0.36]]                  │         │
  └──────────────────────────────────────────────┘         │
        │                                                  │
        │ expert_w: [4, 3]                                 │
        │                                                  │
        ▼ MoEExperts.forward(x=[4,64], weights=[4,3])     │
  ┌──────────────────────────────────────────────┐         │
  │ for i in [0,1,2]:  (遍历 3 个 Expert)        │         │
  │                                              │         │
  │   expert_i(x) → [4, 64]                      │         │
  │   weight_i = weights[:, i] → [4]             │         │
  │   weight_i.unsqueeze(1) → [4, 1]             │         │
  │   contribution = [4, 1] * [4, 64]            │         │
  │                = [4, 64]   ← 正确广播!        │         │
  │                                              │         │
  │   out += contribution                        │         │
  │                                              │         │
  │ 最终 out = w0*E0(x) + w1*E1(x) + w2*E2(x)    │         │
  │         = [4, 64]                             │         │
  └──────────────────────────────────────────────┘         │
        │                                                  │
        ▼ fc (Linear 64→1)                                 │
  [B, 1] = [4, 1]                                          │
        │                                                  │
        ▼ CrossEntropyLoss(logits, style_ids.float())      │
  scalar loss                                              │
                                                           │
  ═══════════════════════════════════════════════════════════
  │                                                        │
  │  ★ 广播 Bug 分析 ★                                      │
  │                                                        │
  │  错误写法: weights[:, i:i+1, None] * expert(x)          │
  │            [4, 1, 1] * [4, 64]                         │
  │            → PyTorch 广播: [4, 1, 1] 的第1维(=1)       │
  │              与 [4, 64] 的第0维(=4) 都匹配 batch=4     │
  │              但 [4,1,1] dim=2(=1) 匹配 [4,64] dim=1(=64)│
  │              结果: [4, 4, 64]  ← 多出一个 batch 维度!   │
  │                                                        │
  │  正确写法: weights[:, i:i+1] * expert(x)                │
  │            [4, 1] * [4, 64] = [4, 64]  ✓               │
  │                                                        │
  │  规则: PyTorch 广播从最后一维往前对齐。                  │
  │        [4,1,1] 和 [4,64] → 右对齐: (1,1) vs (4,64)     │
  │        → 1可以广播到4和64, 所以变成 [4,4,64] — 炸了。   │
  │                                                        │
  ═══════════════════════════════════════════════════════════


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
问题 2: Concatenation 为什么能控制 MusicGen?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  要理解 Concatenation 的控制作用, 必须先理解 Transformer 的
  Cross-Attention 机制。

  2.1 原始 MusicGen 的 Cross-Attention (无 Adapter)

    Q = decoder_hidden       ← shape: [B, T_dec, 1536]
    K = T5(text_prompt)      ← shape: [B, T_txt, 1536]
    V = T5(text_prompt)      ← shape: [B, T_txt, 1536]

    Attention(Q, K, V) = softmax(Q·K^T / √d) · V

    这里的 K 和 V 是 T5 文本编码器的输出。文本 prompt 中的每个 token
    (如 "folk", "ballad", "warm") 都被映射为一个 1536 维向量。

    假设 text_prompt = "folk ballad, warm male vocals, 80 bpm"
    → T_txt ≈ 12 tokens

    解码器的每个位置 (Q) 会关注这 12 个 text token,
    从中提取风格和情绪信息来引导音频 token 的生成。

    问题: 这 12 个 text token 只能表达粗粒度的风格信息。
    "轻柔" 在 T5 的语义空间中是一个模糊的连续区域,
    模型无法精确区分 p (极弱) vs mp (中弱)。

  2.2 加入 Adapter 后的 Cross-Attention

    token_emb = TokenEncoder("[DYN:p|ART:legato|TEX:sparse]")
    → shape: [B, T_tok, 1536],  T_tok ≈ 50 tokens

    K' = concat(T5(text_prompt), token_emb)
       = [B, T_txt + T_tok, 1536]

    V' = concat(T5(text_prompt), token_emb)
       = [B, T_txt + T_tok, 1536]

    Attention(Q, K', V') = softmax(Q·K'^T / √d) · V'

    现在 K 和 V 从 12 个 token 扩展到了 62 个 token。

  2.3 控制信号的机制

    Cross-Attention 的核心是: 解码器的每个位置根据 Q 和 K 的相似度
    来决定从 V 中"读取"多少信息。

    加入 Token 后发生了两件事:

    (a) 注意力竞争:
        softmax 的归一化是所有 K token 共享的。
        Token 向量会"竞争"原本属于 Text 向量的注意力份额。
        例如: [DYN:p] 的 K 向量如果和某个解码位置的 Q 相似,
        解码器会分配 20% 的注意力给 "[DYN:p]",
        从而在 V 中读取出"弱音量"的声学特征。

    (b) 信息注入:
        V 中包含了 Token Encoder 学到的连续表示。
        这些表示是在训练中被优化为"能影响最终音频输出"的向量。
        例如: [ART:staccato] 的 V 向量可能被训练为
        "让解码器生成短促、分离的音符"。

  2.4 为什么 Concatenation 能工作 (数学直觉)

    将 Cross-Attention 展开:

    attn_weights = softmax(
      [Q·K_text^T | Q·K_token^T] / √d
    )

    这是一个 (T_txt + T_tok) 维的向量, 每维对应一个 K 向量的权重。

    输出 = Σ(attn_text_i · V_text_i) + Σ(attn_token_j · V_token_j)

    第一项来自 Text prompt (风格/情绪), 第二项来自 Token (精确控制)。

    关键: Token 部分的梯///度只会更新 Token Encoder,
    不会影响 T5 或 MusicGen 的参数。但 Token Encoder 可以通过
    学习让 K_token 和 V_token 的表示对解码器的输出产生预期的影响。

  2.5 和 Zero-Conv (原版 ControlNet) 的对比

    Zero-Conv:  y = x + Z(x_copy)
    → 是"加法注入", 适合 UNet 的 skip-connection
    → 需要同维度对齐

    Concatenation:  K' = [K_text; K_token]
    → 是"序列拼接注入", 适合 Transformer 的 cross-attention
    → 不需要维度对齐, 任意长度都可以

    原版 ControlNet 用 Zero-Conv 是因为 UNet 的特征图在
    encoder-decoder 之间是空间对齐的, 加法是自然的。
    但 Transformer 没有这种对齐结构——它的 cross-attention
    天然接受任意长度的 K-V 序列, 所以拼接是更自然的选择。

    更重要的是: 在 cross-attention 中, 新增的 K-V 对不会
    改变已有的 K-V 对。它们只是在注意力分布中占据了一部分份额。
    这意味着 Text prompt 的控制路径完全不受影响——
    我们只是增加了一条并行的、更精确的控制路径。

  ═══════════════════════════════════════════════════════════
  │                                                        │
  │  ★ 一句话总结 ★                                          │
  │                                                        │
  │  Concatenation 之所以能控制 MusicGen, 是因为            │
  │  Cross-Attention 的 Q·K'^T 计算允许解码器的每个位置     │
  │  自由选择"关注 Text (风格) 还是 Token (结构)"。         │
  │  Token Encoder 通过反向传播学到让 K_token 和 V_token    │
  │  能触发"正确的音乐行为"。整个过程不修改 MusicGen 的一   │
  │  个参数——我们只是在给一个已经会唱歌的模型增加了一个      │
  │  "更精确的指挥手势"。                                    │
  │                                                        │
  ═══════════════════════════════════════════════════════════
""")
