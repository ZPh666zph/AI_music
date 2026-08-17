# 🎼 Camera-Music ControlNet

**Multi-Viewpoint Structural Prompting for Controllable Music Generation via Hot-Pluggable Mixture-of-Experts**

[![arXiv](https://img.shields.io/badge/arXiv-24XX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/24XX.XXXXX)
[![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-ffcc00.svg)](https://huggingface.co/facebook/musicgen-medium)
[![Paper](https://img.shields.io/badge/PDF-main_v16-0066cc.svg)](final_paper/main_v16.pdf)
[![Checkpoint](https://img.shields.io/badge/Weights-moe_5expert.pt-green.svg)](outputs/checkpoints/moe_5expert.pt)

---

## Abstract

Current AI music generation models suffer from a *prompt-engineering gap* compared to video generation: cinematographic language (close-up, zoom-in, color grading) enables fine-grained multi-axis control over visual outputs, while music generation remains confined to coarse tags (genre, mood, BPM). **Camera-Music ControlNet** bridges this gap by systematically transplanting cinematographic language methodology to the music domain through:

- 🎯 A **six-dimensional Music-Prompt DSL** (Harmony, Rhythm, Texture, Dynamics, Timbre, Articulation) with nine cross-modal correspondences
- ⚡ A lightweight **ControlNet-Adapter** requiring only **0.67%** trainable parameters on frozen MusicGen (1.5B)
- 🔌 A **hot-pluggable Mixture-of-Experts** architecture where new style experts are appended with **zero forgetting** of previously trained experts (mathematically proven)
- 🏭 A **zero-cost automated Chinese traditional music data factory** producing 668 annotated training triples (Erhu, Dizi, Suona)

| Metric | Value |
|--------|:---:|
| Router Accuracy (3 experts) | **99.2%** |
| Timbre Spectral Spread | **+26% / +28%** |
| Chord Accuracy Drop (w/o Harmony) | **34.6%** ($p<0.001$) |
| LLM-as-a-Judge Mean Score | **3.90/5** (87% $\geq$ 4) |
| Zero-Forgetting Verification | **0 parameter changes** across 3 experts |

---

## Project Structure

```
Camera-Music-ControlNet/
├── final_paper/                    # LaTeX source & compiled PDF
│   ├── main_v16.tex               # Current preprint (NeurIPS format)
│   ├── main_v16.pdf               # Compiled 8-page manuscript
│   └── figures/                   # All 8 publication-quality PDF figures
│
├── outputs/
│   ├── checkpoints/               # Trained MoE weights
│   │   ├── best_moe_checkpoint.pt # 3-expert Router (Erhu/Dizi/Suona)
│   │   └── moe_5expert.pt         # 5-expert Router (+Electronic/+Piano)
│   ├── eval_audio/                # 30 × 10s evaluation clips
│   ├── inference/                 # Demo WAVs (Erhu/Dizi/Suona)
│   ├── figures/                   # Raw figure generation output
│   └── llm_eval_results.jsonl     # LLM-as-Judge scores
│
├── data/
│   ├── raw_midis/                 # 226 Chinese traditional MIDI files
│   ├── gufeng_moe_train.jsonl     # 668 MoE training triples
│   ├── modern_styles_train.jsonl  # 100 modern genre triples
│   └── Slakh_Sample/              # Slakh2100 validation subset
│
├── scripts/
│   ├── guofeng_data_factory.py    # MIDI + SF2 → WAV batch renderer
│   ├── guofeng_data_factory_v3.py # Multi-instrument parallel pipeline
│   ├── build_moe_dataset.py       # WAV + meta → JSONL dataset packer
│   ├── batch_rename_midi.py       # MIDI filename cleaner
│   ├── batch_inference_cpu.py     # CPU marathon inference (30 tracks)
│   ├── inference_moe.py           # Single-track MoE inference
│   ├── llm_evaluator.py           # LLM-as-a-Judge automated scoring
│   └── evaluate_metrics.py        # Objective metrics (Chord Acc, BPM, centroid)
│
├── training/
│   ├── train_moe_controlnet.py    # 3-expert MoE training (50 epochs)
│   ├── train_moe_controlnet_gpu.py# GPU-optimized version (bf16 + grad accum)
│   ├── train_moe_expand.py        # Hot-plug expansion (3→5 experts)
│   └── expand_hotplug.py          # Zero-forgetting verification script
│
├── visualization/
│   ├── regenerate_figures.py      # Figures 3-7 (matplotlib + SciencePlots)
│   ├── plot_real_training_figs.py # Real-data training curve + router heatmap
│   └── plot_timbre_distribution.py # Timbre violin plot
│
├── analysis/
│   ├── controlnet_deep_dive.py    # ControlNet principle explainer
│   ├── moe_tensor_flow_explained.py # MoE tensor flow visualization
│   └── moe_hotplug_proof.py       # Mathematical proof + code verification
│
└── setup/
    ├── install_fluidsynth.py       # Auto-download FluidSynth for Windows
    └── inspect_mco.py              # SF2 preset inspector
```

---

## 🎧 Audio Demos (Listen Before You Download)

Same text prompt — *"A beautiful Chinese traditional melody, 90 bpm, minor key"* — routed to three different MoE experts:

| Style Expert | Instrument | Audio | Router Activation |
|:---:|:---:|:---:|:---:|
| string | Erhu (二胡) | [▶️ Listen](assets/audio/demo_erhu.mp3) | 0.990 |
| wind | Dizi (笛子) | [▶️ Listen](assets/audio/demo_dizi.mp3) | 0.995 |
| brass | Suona (唢呐) | [▶️ Listen](assets/audio/demo_suona.mp3) | 0.992 |

> **Listening tip:** the three tracks share an identical text prompt — the only variable is the `[STYLE]` token. The perceptible timbre differences (spectral centroid shifts of +26%/+28%) demonstrate physical-level style disentanglement achieved by the MoE Router.

---

## 📥 Download Pretrained Checkpoints

### Option A — Hugging Face Hub (recommended)

```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='ZPh666zph/AI_music', local_dir='./checkpoints')"
```

| Checkpoint | Experts | Size | Description |
|:---|:---:|:---:|:---|
| `best_moe_checkpoint.pt` | 3 (string/wind/brass) | 55 MB | Best 3-expert Router (99% style accuracy) |
| `moe_5expert.pt` | 5 (+electronic/piano) | 55 MB | Hot-plugged 5-expert Router (zero-forgetting verified) |

### Option B — GitHub Releases

Download from the [Releases page](https://github.com/ZPh666zph/AI_music/releases) and place under `outputs/checkpoints/`.

---

## Quick Start

### Prerequisites

```bash
pip install torch torchaudio transformers datasets soundfile librosa pretty_midi pyfluidsynth tqdm matplotlib scienceplots
```

Windows users: FluidSynth is auto-installed via `python setup/install_fluidsynth.py`.

### 30-Second Demo

```python
import torch, soundfile as sf
from transformers import MusicgenForConditionalGeneration, AutoProcessor

# Load base model
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")

# Generate with style prompt
prompt = "A beautiful Chinese traditional melody played by erhu, 90 bpm, minor key"
inputs = processor(text=[prompt], padding=True, return_tensors="pt")

with torch.no_grad():
    audio = model.generate(**inputs, max_new_tokens=256)

sf.write("demo_erhu.wav", audio[0, 0].numpy(), 32000)
print("Saved: demo_erhu.wav")
```

### Run Full Inference Pipeline

```bash
# 1. Batch render MIDI → WAV (3 instruments × N tracks)
python scripts/guofeng_data_factory_v3.py

# 2. Pack into training dataset
python scripts/build_moe_dataset.py

# 3. Train 3-expert MoE Router
python training/train_moe_controlnet.py

# 4. Expand to 5 experts (zero-forgetting)
python training/expand_hotplug.py

# 5. Generate evaluation audio
python scripts/batch_inference_cpu.py

# 6. Run LLM-as-a-Judge scoring
python scripts/llm_evaluator.py
```

---

## Key Design Decisions

### Why Concatenation Injection Instead of Zero-Convolution?

| Zero-Conv (Image ControlNet) | Concatenation (Ours) |
|---|---|
| Additive: $y = x + Z(x_{\text{copy}})$ | Concatenative: $K' = [K_{\text{text}}; K_{\text{token}}]$ |
| Requires same-dimensional alignment | No dimension constraint |
| Suited for UNet skip-connections | Suited for Transformer cross-attention |
| Zero-initialized for gradual warm-up | Xavier-initialized, no warm-up needed |

MusicGen's Transformer decoder naturally accepts variable-length Key-Value sequences in its cross-attention layers. Concatenation is the structurally natural choice; additive injection has no architectural anchor point in a 48-layer decoder stack.

### Why Explicit Routing Instead of Learned Gating?

Traditional MoE uses learned gating ($w = \text{softmax}(W_g \cdot h)$) where the router selects experts based on hidden states. We use **explicit routing** ($w = \text{softmax}(W_g \cdot \text{MLP}(s))$) where the router selects experts based on the `[STYLE]` token embedding $s$. This has three advantages:

1. **Deterministic inference:** $[STYLE:gufeng] \rightarrow$ one-hot routing to Expert 1, zero computation overhead
2. **Hot-plug semantics:** new expert = append one row to $W_g$, old rows unchanged by construction
3. **Style-level interpretability:** each expert corresponds to exactly one instrument family (string/wind/brass)

---

## Citation

```bibtex
@article{cameramusic2024,
  title={Multi-Viewpoint Structural Prompting for Controllable Music Generation via Hot-Pluggable Mixture-of-Experts},
  author={Author One and Author Two and Author Three},
  journal={arXiv preprint arXiv:24XX.XXXXX},
  year={2024}
}
```

---

## License

This project is released under the [CC BY-NC 4.0](LICENSE) license. The MusicGen base model is subject to Meta's [CC BY-NC 4.0](https://github.com/facebookresearch/audiocraft/blob/main/LICENSE) license.

---

<p align="center">
  <sub>Built with ❤️ for NeurIPS 2026 · Camera-Music ControlNet Team</sub>
</p>
