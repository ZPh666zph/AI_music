#!/usr/bin/env bash
# 统一 FAD + CLAP（模型音频目录 + 参考集 + prompts）
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m eval.eval_unified_fad_clap \
    --gen_dirs "${GEN_DIRS:-results/musicgen results/audioldm2}" \
    --ref_dir "${REF_DIR:-data/ref_audio}" \
    --prompts "${PROMPTS:-baseline_prompts.json}"
