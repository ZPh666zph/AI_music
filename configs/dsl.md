# Music-Prompt DSL（6 轴）词汇说明

论文提出的六轴 DSL 由一个 axis token + value 组成：`[AXIS:value]`。
运行时由 `src/model/train_controlnet.py` 的 `MusicTokenVocabulary` 编码为整数序列，
并可在轴间以 `|` 组合，或按时序（`[SEC:...|BAR:a-b]`）组织成 storyboard。

## 六轴表

| 轴 | token | 取值 |
|---|---|---|
| Harmony 和声 | `[CHORD:]` / `KEY:` | C, Am, F#:min7, … ; KEY: A_minor |
| Rhythm 节奏 | `[TEMPO:]` / `TIME:` | 60–160 BPM; 4/4, 3/4, 6/8… |
| Texture 织体 | `[TEX:]` / `[VOICES:]` | sparse/medium/dense/polyphonic; 1–2, 8+ |
| Dynamics 力度 | `[DYN:]` | pp, p, mp, mf, f, ff, CRESC |
| Timbre 音色 | `[COLOR:]` | warm, bright, cold, dark |
| Articulation 运音法 | `[ART:]` | legato, staccato, marcato |

## 结构 token（storyboard 组织，不进 baseline prompt）

`[GLOBAL:...]` 全局属性；`[SEC:intro|BAR:1-8]` 分段；`[STEM:erhu]` 乐器轨道；
`[PATTERN:auto|DYN:p|ART:legato|TEX:sparse]` 轨道级模式。

## 示例（guofeng 测试集单条）

```
[GLOBAL:STYLE:erhu_solo][GLOBAL:TEMPO:120][GLOBAL:KEY:G_major][GLOBAL:TIME:4/4]
[SEC:intro|BAR:1-8][STEM:erhu][PATTERN:auto|DYN:p|ART:legato|TEX:sparse|COLOR:warm]
```

翻译到自然语言见 `src/data/dsl_to_plaintext.py`（baseline 公平对齐用）。
