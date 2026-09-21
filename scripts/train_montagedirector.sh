#!/usr/bin/env bash
# 训练 0.67% Token-Encoder 适配器（ControlledMusicGen）
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m model.train_controlnet \
    --data "${DATA:-data/guofeng_train.jsonl}" \
    --output_dir "${OUT:-checkpoints}" \
    --epochs "${EPOCHS:-5}" \
    --batch_size 1 --grad_accum "${GRAD_ACCUM:-16}" \
    --max_audio_sec 5 --amp bf16
