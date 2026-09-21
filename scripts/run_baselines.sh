#!/usr/bin/env bash
# 开源基线推理：MusicGen / AudioLDM 2 / Stable Audio Open
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m bench.run_hf_baselines --seconds "${SECONDS_LEN:-10}" --prompts "${PROMPTS:-baseline_prompts.json}"
