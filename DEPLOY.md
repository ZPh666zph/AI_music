# DEPLOY.md — AutoDL(Linux) 迁移说明书：上传 → 数据产线 → 开始训练

> 目标：把本仓库部署到 AutoDL 等 Linux GPU 服务器，用 `raw_midis` 重新生成数据，
> 然后开始两种训练：**① 0.67% Token-Encoder 适配器（MontageDirector 核心）② 显式路由 MoE**。
> 全程命令均为 **Linux bash**，请逐段执行。

---

## 0. 本机打包与上传

```bash
# —— 在本机（Windows Git Bash / PowerShell）执行 ——
# ① 精简仓库（含本轮已补的数据产线）
tar -czf montagedirector.tar.gz montagedirector

# ② MIDI 原料 + SoundFont 音色库
tar -czf raw_midis.tar.gz  data/raw_midis
tar -czf soundfonts.tar.gz soundfonts

# ③ 上传（端口与主机换成 AutoDL 实例给的）
scp -P <端口> montagedirector.tar.gz raw_midis.tar.gz soundfonts.tar.gz \
    root@<region>.autodl.com:/root/autodl-fs/
```

> 建议都放 `/root/autodl-fs/`（持久盘，换实例不丢）。

---

## 1. 服务器登录与目录

```bash
ssh -p <端口> root@<region>.autodl.com
mkdir -p /root/autodl-fs/checkpoints /root/autodl-fs/data
cd /root/autodl-fs
tar -xzf montagedirector.tar.gz
tar -xzf raw_midis.tar.gz      # 解出 raw_midis/
tar -xzf soundfonts.tar.gz     # 解出 soundfonts/
ls montagedirector/src/model   # 应含 train_controlnet.py / train_moe.py ...
nvidia-smi                     # 确认 GPU 可见
```

---

## 2. 系统与 Python 环境

```bash
# 系统依赖：FluidSynth（MIDI→WAV 渲染必需）
apt-get update && apt-get install -y fluidsynth
fluidsynth --version

# Python 环境（AutoDL 镜像通常自带 conda；建议建独立环境）
conda create -n montagedirector python=3.10 -y
conda activate montagedirector

# 项目依赖
cd /root/autodl-fs/montagedirector
pip install -r requirements.txt
pip install pretty_midi music21    # 数据产线需要（requirements 已含则跳过）
```

> ⚠️ `requirements.txt` 会安装 `transformers==4.35.0` 并锁定：
> KV-concat 注入只在该版本验证过，**不要升级**。
> 权重（`facebook/musicgen-medium`）首次训练/推理会自动从 HuggingFace 下载。

---

## 3. 数据产线：raw_midis → guofeng jsonl

```bash
cd /root/autodl-fs/montagedirector
export PYTHONPATH=$PWD/src

# 配置路径（Linux）
export GF_MIDI_DIR=/root/autodl-fs/raw_midis          # 你上传解压的 MIDI 目录
export GF_SOUNDFONT_DIR=/root/autodl-fs/soundfonts    # 4 个 .sf2 所在目录
export GF_OUT_DIR=/root/autodl-fs/guofeng_v3          # 渲染输出（mix.wav + meta.json）
export GF_TEMP_DIR=/root/autodl-fs/guofeng_temp
export GF_FLUIDSYNTH=fluidsynth

# ① 先渲染 1 首验证（强烈建议）
python -m data.guofeng_data_factory --demo
ls /root/autodl-fs/guofeng_v3/*/*/mix.wav | head        # 应有 wav

# ② 全量渲染（226 首 × 3 乐器 ≈ 678 条；可后台跑）
nohup python -m data.guofeng_data_factory > render.log 2>&1 &
tail -f render.log

# ③ 打包成 jsonl（text + tokens + audio_path + style_expert）
export GF_OUTPUT=/root/autodl-fs/data/gufeng_train.jsonl
python -m data.build_moe_dataset
wc -l /root/autodl-fs/data/gufeng_train.jsonl            # 期望 ~668-678
```

### 3.1 划分 train / test（规范实验必需）

```bash
cd /root/autodl-fs/data
# 按行随机切分：90% 训练 / 10% 测试（同一随机种子，保证可复现）
shuf --random-source=<(yes 42) gufeng_train.jsonl \
     > _shuf.jsonl
total=$(wc -l < _shuf.jsonl)
head -n $((total * 9 / 10)) _shuf.jsonl > gufeng_train_split.jsonl
tail -n $((total - total * 9 / 10)) _shuf.jsonl > gufeng_test_split.jsonl
rm _shuf.jsonl
wc -l gufeng_train_split.jsonl gufeng_test_split.jsonl
```

---

## 4. 训练 ①：0.67% Token-Encoder（MontageDirector 核心）

```bash
cd /root/autodl-fs/montagedirector
conda activate montagedirector
export PYTHONPATH=$PWD/src

# 4.0 先 dry-run：验证加载/冻结（不训练，秒回）
python -m model.train_controlnet \
    --data /root/autodl-fs/data/gufeng_train_split.jsonl \
    --output_dir /root/autodl-fs/checkpoints --dry_run

# 4.1 正式训练（24G+ 显存建议 batch 可调高；8G 保持 batch=1 + grad_accum）
python -m model.train_controlnet \
    --data /root/autodl-fs/data/gufeng_train_split.jsonl \
    --output_dir /root/autodl-fs/checkpoints \
    --epochs 30 \
    --batch_size 4 \
    --grad_accum 16 \
    --lr 1e-4 \
    --save_every 5 \
    --max_audio_sec 8 \
    --amp bf16 \
    --grad_checkpoint
# → 每 5 epoch 保存 checkpoints/checkpoint_epoch{N}.pt（含 token_encoder + vocab）
```

参数参考：
| 参数 | 说明 |
|---|---|
| `--epochs` | 收敛建议 30–50 |
| `--batch_size` | 24G 卡可 4–8；8G 卡保持 1 |
| `--grad_accum` | 有效 batch = batch_size × grad_accum，保持 ≥8 |
| `--amp bf16` | Ampere+ 用 bf16；老卡用 fp16 |
| `--grad_checkpoint` | 显存紧张时开启（慢一点）|

> **关于"真实 CE 音频训练"**：当前 `train_controlnet` 在无音频编码数据时使用对齐辅助 loss 训练
> （可收敛、可出 checkpoint）。若需论文级指挥效果，需把 jsonl 补充真实音频的 EnCodec tokens
> 作为 labels（本项目 `MusicControlDataset` 已预留，属后续增强）。

---

## 5. 训练 ②：显式路由 MoE（3 专家 → 8 专家）

```bash
cd /root/autodl-fs/montagedirector
conda activate montagedirector
export PYTHONPATH=$PWD/src

# 用同一份 jsonl；MoE 需要 style_expert 字段
export GF_MOE_JSONL=/root/autodl-fs/data/gufeng_train_split.jsonl
export GF_MOE_CKPT_DIR=/root/autodl-fs/checkpoints
export GF_MOE_EPOCHS=50
export GF_MOE_MAX_SAMPLES=668

python -m model.train_moe
# → checkpoints/best_moe_checkpoint.pt + moe_epoch_{N}.pt

# 8 专家 hot-plug 扩展（3→8，零遗忘验证）
python -m model.moe_expansion
```

---

## 6. 常见问题速查

| 问题 | 处理 |
|---|---|
| `transformers` 被升级导致注入失败 | `pip install transformers==4.35.0 huggingface-hub<1 tokenizers<0.15` |
| HuggingFace 下载慢/失败 | `export HF_ENDPOINT=https://hf-mirror.com` 后重试 |
| CUDA OOM | 降 `--batch_size`、开 `--grad_checkpoint`、`--max_audio_sec 5` |
| `fluidsynth` 找不到 | `which fluidsynth`；没装则 `apt-get install -y fluidsynth` |
| 渲染全是 FAIL | 先 `python -m data.guofeng_data_factory --demo` 单首看报错；确认 SF2 路径 |
| 想多卡/换 GPU | 数据与 checkpoint 都在 `/root/autodl-fs`，重建实例后重挂即可 |

---

*到"开始训练"即止。评测（FAD/CLAP）、推理生成（run_ours）与论文级真实 CE 训练见 README.md。*
