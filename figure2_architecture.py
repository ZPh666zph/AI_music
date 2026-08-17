#!/usr/bin/env python3
"""
figure2_architecture.py — ControlNet-Adapter 架构图 (Graphviz)
===============================================================
输出: C:/Deepseek/outputs/figures/figure2_architecture.pdf
嵌入: \includegraphics[width=\columnwidth]{figures/figure2_architecture.pdf}
"""

import os, sys
sys.stdout.reconfigure(encoding='utf-8')

try:
    import graphviz
except ImportError:
    print("pip install graphviz (Python) + https://graphviz.org/download/ (system)")
    sys.exit(1)

dot = graphviz.Digraph("ControlNet-Adapter", format="pdf",
    graph_attr={
        "rankdir": "TB", "splines": "ortho", "nodesep": "0.5", "ranksep": "0.7",
        "fontname": "Times New Roman", "fontsize": "11", "bgcolor": "white",
        "label": "Figure 2: Camera-Music ControlNet-Adapter Architecture",
        "labelloc": "t", "labeljust": "l",
    },
    node_attr={"fontname": "Times New Roman", "fontsize": "9", "shape": "box"},
    edge_attr={"fontname": "Times New Roman", "fontsize": "7"},
)

with dot.subgraph(name="cluster_input") as c:
    c.attr(label="Input", style="rounded,filled", fillcolor="#f0f4ff", fontsize="10")
    c.node("text_prompt", "Text Prompt\n\"folk ballad, warm male vocals\"", shape="note", fillcolor="#e3f2fd")
    c.node("token_seq", "Token Sequence\n[SEC:verse][STEM:erhu]\n[CHORD:Am][DYN:mp]...", shape="note", fillcolor="#fce4ec")

with dot.subgraph(name="cluster_encoders") as c:
    c.attr(label="Encoders", style="rounded,filled", fillcolor="#fff3e0", fontsize="10")
    c.node("t5", "T5 Encoder\n(Frozen, 110M)", style="filled", fillcolor="#bbdefb")
    c.node("tok_enc", "Token Encoder\n(4× Transformer)\n512→1536 Projection\n(Trainable, 13.4M)", style="filled", fillcolor="#f48fb1")
    c.node("concat", "Concat + Position\n[B, Tt+Tk, 1536]", shape="parallelogram", style="filled", fillcolor="#c8e6c9")

with dot.subgraph(name="cluster_decoder") as c:
    c.attr(label="MusicGen Decoder (48 Layers, Frozen, 1.5B)", style="rounded,filled", fillcolor="#e8eaf6", fontsize="10")
    c.node("shared_1", "Shared L1-L12\n(通用音乐理解)", style="filled", fillcolor="#c5cae9")
    c.node("moe_box", "MoE Layers L13-L36\n(Style Router + 3 Experts)", shape="box3d", style="filled", fillcolor="#ffccbc")
    c.node("shared_2", "Shared L37-L48\n(解码到音频)", style="filled", fillcolor="#c5cae9")

with dot.subgraph(name="cluster_moe_detail") as c:
    c.attr(label="MoE Detail", style="rounded,filled", fillcolor="#fff8e1", fontsize="9")
    c.node("router", "Style Router\nW = [K×d]", shape="diamond", style="filled", fillcolor="#ffe0b2")
    c.node("e0", "E0: Pop\n(原版 FFN)", style="filled", fillcolor="#e0e0e0")
    c.node("e1", "E1: Gufeng\n+五声音阶 bias\n+民乐 embed", style="filled", fillcolor="#ffcdd2")
    c.node("e2", "E2: Folk\n+简化织体", style="filled", fillcolor="#c8e6c9")

c.node("audio_enc", "EnCodec Decoder\n(Frozen)", style="filled", fillcolor="#b39ddb")
c.node("output", "Audio Output\n(.wav)", shape="cds", style="filled", fillcolor="#a5d6a7")

# ── Edges ──
dot.edge("text_prompt", "t5", "Text Tokenizer")
dot.edge("token_seq", "tok_enc", "Vocab Embedding")
dot.edge("t5", "concat", "[B, Tt, 768] → [B, Tt, 1536]")
dot.edge("tok_enc", "concat", "[B, Tk, 1536]")
dot.edge("concat", "shared_1", "combined_emb")
dot.edge("shared_1", "moe_box")
dot.edge("moe_box", "shared_2")
dot.edge("shared_2", "audio_enc")
dot.edge("audio_enc", "output")

# MoE internal
dot.edge("router", "e0", "w0")
dot.edge("router", "e1", "w1")
dot.edge("router", "e2", "w2")
dot.edge("e0", "moe_box", style="dashed", arrowhead="none")
dot.edge("e1", "moe_box", style="dashed", arrowhead="none")
dot.edge("e2", "moe_box", style="dashed", arrowhead="none")

# Training flag
dot.node("trainable_flag", "TRAINABLE (0.67%)\nToken Encoder +\nRouter + Experts", 
         shape="note", fillcolor="#ffeb3b", fontsize="8", color="red")
dot.node("frozen_flag", "FROZEN (99.33%)\nT5 + MusicGen +\nEnCodec",
         shape="note", fillcolor="#e0e0e0", fontsize="8", color="gray")
dot.edge("trainable_flag", "tok_enc", style="dotted", color="red")
dot.edge("frozen_flag", "t5", style="dotted", color="gray")

out_dir = "C:/Deepseek/outputs/figures"
os.makedirs(out_dir, exist_ok=True)
dot.render(f"{out_dir}/figure2_architecture", cleanup=True)
print(f"Figure 2: {out_dir}/figure2_architecture.pdf")
