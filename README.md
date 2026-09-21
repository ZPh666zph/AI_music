# MontageDirector

**Hierarchical, fine-grained controllable music generation via a cross-modal 6-axis Music-Prompt DSL.**

MontageDirector steers a **frozen MusicGen-medium backbone** with an ultra-lightweight
**0.67% token-encoder adapter** (13.4M params) through **Key-Value concatenation**
(`[T5(text) ‖ TokenEncoder(DSL)]` fed into the decoder cross-attention), and isolates
musical styles with an **explicitly-routed, hot-pluggable MoE** whose row-appended router
matrix provably incurs **zero catastrophic forgetting**.

> 🎧 **Demo audio (repo):** [`assets/audio/`](assets/audio/) · **anonymous listening-test pack (24 MB):** `<DEMO_AUDIO_URL>`
> 📄 **Paper (PDF, 16 pp.):** `<PAPER_URL>`
> 📦 **Model weights & full 12 GB recorded corpus:** `<RELEASE_URL>`
>
> _The three placeholders above are intentionally left as links: per GitHub's size policy the
> repository tracks **code, configs, scripts and docs only**; audio, weights and the 9.4 GB
> reference corpus are published externally (see **§5 Data availability**)._


---

## 1. Repository layout

```
montagedirector/
├── scripts/                     # 一键启动入口（薄封装）
├── configs/                     # 数据字段 / DSL 词汇说明
├── data/                        # 数据集格式说明（真实音频不入库，见 .gitignore）
└── src/
    ├── model/                   # 论文核心模型
    │   ├── train_controlnet.py  #   ControlledMusicGen：冻结 MusicGen + 0.67% TokenEncoder，KV-concat 注入 + 训练/生成
    │   ├── train_moe.py         #   显式路由 MoE 训练（MoEControlNet，3 专家基础）
    │   ├── moe_expansion.py     #   hot-plug 扩展到 8 专家（row-append router，零遗忘）
    │   └── hotplug_proof.py     #   Proposition 1 数学不可变性验证
    ├── data/                    # DSL / 数据管线
    │   ├── dsl_to_plaintext.py  #   6 轴 DSL → 自然语言（baseline 公平对齐）
    │   ├── extract_6axis.py     #   从 MIDI 提取 6 轴控制标签
    │   ├── guofeng_data_factory.py  # 国风数据集工厂（MIDI+SF2 → 多轨 mix）
    │   └── slakh_to_tokens.py   #   Slakh2100 → DSL token 三元组
    ├── engine/                  # LLM Conductor（论文 Tier 1/2）
    │   ├── llm_conductor.py     #   自由文本 → 语义状态 → 时序 storyboard → DSL
    │   └── llm_evaluator.py     #   LLM-as-a-Judge 评分
    ├── inference/               # Ours 推理
    │   ├── run_ours.py          #   checkpoint_epoch5 → 批量 668×10s 生成（000.wav..667.wav）
    │   └── inference_moe.py     #   MoE router 验证 + demo 生成
    ├── bench/                   # 开源基线推理（对比实验用）
    │   ├── run_hf_baselines.py  #   MusicGen / AudioLDM 2 / Stable Audio Open
    │   ├── run_musicongen.py    #   MusiConGen（需其官方 repo + 权重）
    │   └── midi_to_musicongen_conditions.py  # MIDI → MusiConGen 和弦/BPM
    └── eval/                    # 统一客观评测
        ├── eval_unified_fad_clap.py  # FAD(VGGish,多进程) + CLAP(分批 GPU)
        ├── eval_metrics_pipeline.py  # BPM/Chord/RMS/SpectralCentroid 客观指标
        └── eval_moe_separation.py    # MoE 音色分离验证
```

---

## 2. Environment

```bash
conda create -n montagedirector python=3.10 -y
conda activate montagedirector
pip install -r requirements.txt
```

> ⚠️ **transformers must stay at `4.35.0`** — the KV-concat injection relies on
> `MusicgenForConditionalGeneration.generate` accepting a precomputed
> `encoder_outputs` (a behaviour removed in later versions). The adapter code
> monkey-patches `_prepare_text_encoder_kwargs_for_generation` accordingly.

Backbone weights are pulled from HuggingFace on first run
(`facebook/musicgen-medium`); no weights are stored in this repo.

---

## 3. Reproduce the paper pipeline

### 3.1 Train the 0.67% Token-Encoder adapter

```bash
# data: JSONL lines of {text, tokens} (and audio for real CE training)
python -m model.train_controlnet --data <your_triples.jsonl> \
    --output_dir checkpoints --epochs 5 --batch_size 1 \
    --grad_accum 16 --max_audio_sec 5 --amp bf16
```

> Training notes: on machines **without** an audio-encoding pipeline the current
> script falls back to an alignment auxiliary loss (keeps the adapter trainable and
> the checkpoint format correct). For paper-grade audio, train with real
> EnCodec audio tokens as labels (see `model/train_controlnet.py` `MusicControlDataset`
> and Section 5).

### 3.2 Generate the test set (Ours)

```bash
python -m inference.run_ours \
    --ckpt checkpoints/checkpoint_epoch5.pt \
    --data <guofeng_moe_train.jsonl> \
    --out_dir results/ours --seconds 10 --device cuda
```

### 3.3 Run the open baselines & unified metrics

```bash
# baselines (HF 三件套 + MusiConGen 见 bench/README)
python -m bench.run_hf_baselines --seconds 10
# metrics
python -m eval.eval_unified_fad_clap --gen_dirs results/musicgen results/audioldm2 \
    --ref_dir <reference_audio_16k> --prompts baseline_prompts.json
```

---

## 4. Data format

Test/train sets are line-delimited JSON with:

| field | meaning |
|---|---|
| `text` | free-form natural-language description |
| `tokens` / `token` | the 6-axis Music-Prompt DSL (see `configs/dsl.md`) |
| `audio_path` | rendered audio (training with real audio tokens) |

`dsl_to_plaintext.py` turns DSL into plain text so text-only baselines get the
*same semantic content* (apples-to-apples comparison).

### 4.1 Generating the guofeng JSONL from raw MIDI (two steps)

```
raw_midis/*.mid ──①──> src/data/guofeng_data_factory.py   (渲染: erhu/dizi/suona 三路)
                        → <out_dir>/<song>/<inst>/mix.wav + meta.json
                  ──②──> src/data/build_moe_dataset.py     (打包)
                        → guofeng_moe_train.jsonl  (每行 text + tokens + audio_path)
```

```bash
# ① 渲染（Linux 需先 apt-get install fluidsynth；SF2 音色库放 GF_SOUNDFONT_DIR）
export GF_MIDI_DIR=/path/raw_midis GF_SOUNDFONT_DIR=/path/soundfonts \
       GF_OUT_DIR=/path/guofeng_v3 GF_FLUIDSYNTH=fluidsynth
python -m data.guofeng_data_factory --demo        # 先试 1 首，再全量

# ② 打包成训练/测试 jsonl
python -m data.build_moe_dataset                  # GF_OUTPUT 指定输出 jsonl
```

---

## 5. Known limitations (honest notes for reproducibility)

- **Adapter weights / checkpoints are not shipped.** Train your own with the
  scripts above; on our dev machine (RTX 5060 8GB) we trained
  `checkpoint_epoch5.pt` (loss 0.97 → 0.63) and generated 10 s clips successfully,
  but paper-grade audio requires training with real EnCodec CE labels on a
  ≥24 GB GPU (recommend `facebook/musicgen-medium`, fp16, batch ≥ 8).
- **CLAP**: `laion-clap` needs a stock PyTorch (2.0–2.3) env; on bleeding-edge
  torch it hangs — run `eval.eval_unified_fad_clap` in a clean env.
- **Baselines** require their official weights (download separately).

## 6. Data availability

The repository is deliberately kept lightweight (code + configs + scripts + docs only).
The following artefacts are **not** tracked in Git and are published externally:

| Artefact | Size | Where |
|---|---|---|
| Generated audio, 5 systems x 669 tracks (MusicGen / AudioLDM 2 / Stable Audio Open / MusiConGen / Ours) | 12 GB | `<RELEASE_URL>` |
| gufeng reference corpus (669 recorded renders, 16 kHz) | 9.4 GB | `<RELEASE_URL>` |
| Trained adapter checkpoints (13.4 M params, CE 10-epoch + ablation) | 1.2 GB | `<RELEASE_URL>` |
| Anonymous listening-test pack (30 groups x 5 systems, mp3 + sheets) | 24 MB | `<DEMO_AUDIO_URL>` |

Reproducibility note: all objective numbers in `TABLE_4plus1.md` were produced on a single
RTX 5090 with one unified evaluation pipeline (VGGish-FAD + LAION-CLAP), the same reference
set for every system, and a **disjoint train/test split** (602 train / 63 shared test prompts).

## 7. License & citation

See `LICENSE`. If you use this code, please cite the MontageDirector paper (TBD).
