# data/ — 说明（真实数据不入库）

本仓库**不随附**音频与 JSONL 数据（体积大且常含未发布资源）。
请按以下字段自行准备，或联系作者获取 guofeng 测试集（N=668）。

## 测试/训练 JSONL 格式（每行一个 JSON）

```json
{
  "track_name": "Angel前奏",
  "style_expert": "wind",
  "text": "Chinese dizi (bamboo flute) performance of gufeng music, ...",
  "tokens": "[GLOBAL:STYLE:dizi]...[SEC:intro|BAR:1-8]...",
  "audio_path": "path/to/mix.wav"
}
```

- `tokens`：6 轴 Music-Prompt DSL（见 `configs/dsl.md`）
- `audio_path`：渲染/真值音频（供真实 CE 训练与 FAD 参考集）

## 建议目录

```
data/
├── guofeng_train.jsonl        # 训练三元组（text+tokens+audio_path）
├── guofeng_test.jsonl         # 测试集（与 baseline_prompts id 0..N-1 对齐）
└── ref_audio/000.wav ...      # FAD 参考（真值，16k 单声道）
```

## 生成 baseline 对齐文本

```bash
python -m data.dsl_to_plaintext --input data/guofeng_test.jsonl \
    --output baseline_prompts.json --jsonl
```
