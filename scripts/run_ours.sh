#!/usr/bin/env bash
# 用已训练 adapter 批量生成 668 条测试音频（Ours）
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m inference.run_ours \
    --ckpt "${CKPT:-checkpoints/checkpoint_epoch5.pt}" \
    --data "${DATA:-data/guofeng_test.jsonl}" \
    --out_dir "${OUT:-results/ours}" \
    --seconds "${SECONDS_LEN:-10}" \
    --device "${DEVICE:-cuda}"
