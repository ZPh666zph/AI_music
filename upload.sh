#!/usr/bin/env bash
# upload.sh — Camera-Music ControlNet 一键上传
# 用法: bash upload.sh

set -e
REPO="https://github.com/ZPh666zph/AI_music.git"
HF_REPO="ZPh666zph/AI_music"

echo "=============================================="
echo "Camera-Music ControlNet 上传脚本"
echo "=============================================="

# ── Step 1: 上传模型权重到 Hugging Face ──
echo ""
echo "[1/2] 上传模型权重到 Hugging Face Hub..."
if command -v huggingface-cli &> /dev/null; then
    huggingface-cli upload "$HF_REPO" \
        outputs/checkpoints/best_moe_checkpoint.pt \
        outputs/checkpoints/moe_5expert.pt \
        outputs/checkpoints/moe_epoch_50.pt
    echo "  ✓ 权重已上传到 $HF_REPO"
else
    echo "  ⚠ 未安装 huggingface-cli, 跳过 (pip install huggingface_hub)"
fi

# ── Step 2: 上传代码 + 音频到 GitHub ──
echo ""
echo "[2/2] 上传代码 + 音频到 GitHub..."
git init
git add -A
git commit -m "Camera-Music ControlNet: code, demos, and paper" || echo "  (无变更可提交)"
git branch -M main
git remote remove origin 2>/dev/null || true
git remote add origin "$REPO"
git push -u origin main

echo ""
echo "=============================================="
echo "完成!"
echo "  代码+音频: $REPO"
echo "  模型权重:  https://huggingface.co/$HF_REPO"
echo "=============================================="
