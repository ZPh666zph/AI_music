#!/usr/bin/env bash
# 训练显式路由 MoE（3 专家基础）；扩展 8 专家见 src/model/moe_expansion.py
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m model.train_moe
