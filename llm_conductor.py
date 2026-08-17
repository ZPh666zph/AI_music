#!/usr/bin/env python3
"""
llm_conductor.py — LLM「音乐指挥家」Agent Workflow 原型
=========================================================
输入: 用户大白话 (如 "写一首伤感的关于毕业的歌")
输出: 带控制 Token 的歌词文本 + 结构化 JSON
"""

import sys, json, re, os

# ── 第零层: Token 知识库 ──
# LLM 需要知道哪些 Token 可用、各自什么含义

TOKEN_KB = {
    "dynamics": {
        "pp": "极弱 · 独白感 · 耳语",
        "p":  "弱 · 私密 · 低语",
        "mp": "中弱 · 叙述 · 日常语气",
        "mf": "中强 · 坚定 · 正常歌唱",
        "f":  "强 · 爆发 · 呐喊",
        "ff": "极强 · 撕心裂肺 · 嘶吼",
    },
    "articulation": {
        "legato":   "连奏 · 流畅 · 不舍 · 绵长",
        "staccato":  "断奏 · 跳跃 · 决绝 · 短促",
        "marcato":   "强奏 · 强调 · 宣言式",
        "breathy":   "气声 · 脆弱 · 亲密",
    },
    "texture": {
        "sparse": "稀疏 · 一支吉他/一架钢琴 · 孤独",
        "medium": "中等 · 乐队进入 · 逐渐丰满",
        "dense":  "密集 · 全编制 · 情感巅峰",
    },
    "timbre": {
        "warm":   "温暖 · 木吉他 · 大提琴 · 回忆",
        "bright":  "明亮 · 钢琴高音区 · 希望",
        "dark":   "暗沉 · 低音提琴 · 沉重",
    },
    "harmony_hint": {
        "major": "明亮 · 希望 · 坚定 · 大调",
        "minor": "忧伤 · 思念 · 遗憾 · 小调",
        "tension": "紧张 · 不安 · 属七和弦",
    },
    "tempo": {
        "60-70":  "慢 · 沉思 · 挽歌",
        "70-90":  "中慢 · 民谣叙事",
        "90-110": "中速 · 流行 · 行走感",
        "110+":   "快速 · 激动 · 释放",
    },
}

# ── 情感→Token 映射表 ──
EMOTION_MAP = {
    "伤感":  {"key": "minor", "dyn": "mp", "art": "legato", "tex": "sparse", "color": "warm", "tempo": "70-80"},
    "怀念":  {"key": "minor", "dyn": "p",  "art": "legato", "tex": "sparse", "color": "warm", "tempo": "60-70"},
    "遗憾":  {"key": "minor", "dyn": "mp", "art": "legato", "tex": "sparse", "color": "dark", "tempo": "60-75"},
    "希望":  {"key": "major", "dyn": "mf", "art": "legato", "tex": "medium","color": "bright","tempo": "90-100"},
    "热血":  {"key": "major", "dyn": "f",  "art": "marcato","tex": "dense", "color": "bright","tempo": "110+"},
    "孤独":  {"key": "minor", "dyn": "p",  "art": "legato", "tex": "sparse", "color": "dark", "tempo": "60-70"},
    "释然":  {"key": "major", "dyn": "mf", "art": "legato", "tex": "medium","color": "warm", "tempo": "80-90"},
    "愤怒":  {"key": "minor", "dyn": "ff", "art": "marcato","tex": "dense", "color": "dark", "tempo": "120+"},
    "甜蜜":  {"key": "major", "dyn": "mp", "art": "legato", "tex": "sparse", "color": "bright","tempo": "80-95"},
    "思念":  {"key": "minor", "dyn": "p",  "art": "breathy","tex": "sparse", "color": "warm", "tempo": "65-75"},
}

# 歌曲段落→Token 映射
SECTION_MAP = {
    "intro":    {"dyn": "p",  "art": "legato",  "tex": "sparse", "desc": "引入 · 一个乐器 · 氛围铺垫"},
    "verse":    {"dyn": "mp", "art": "legato",  "tex": "sparse", "desc": "主歌 · 叙事 · 克制"},
    "pre-chorus":{"dyn": "mf","art": "legato",  "tex": "medium","desc": "预副歌 · 情绪爬升"},
    "chorus":   {"dyn": "f",  "art": "marcato", "tex": "dense",  "desc": "副歌 · 爆发 · 记忆点"},
    "bridge":   {"dyn": "mp", "art": "legato",  "tex": "medium","desc": "桥段 · 转折 · 反思"},
    "outro":    {"dyn": "p",  "art": "legato",  "tex": "sparse", "desc": "尾声 · 渐弱 · 余韵"},
}


# ══════════════════════════════════════════════════════════
# LLM Prompt 模板（核心交互逻辑）
# ══════════════════════════════════════════════════════════

SYSTEM_PROMPT = """你是一位 AI 音乐指挥家 (Music Conductor)。

你的任务分两步:
1. 根据用户的大白话说，创作一首中文歌词
2. 在歌词的每个段落开头，插入合适的控制 Token

你可以使用的控制维度:
- [DYN:值]    动态(音量/强度): pp p mp mf f ff
- [ART:值]    奏法: legato(连奏-流畅) staccato(断奏-跳跃) marcato(强奏-宣言) breathy(气声-脆弱)
- [TEX:值]    织体(配器密度): sparse(稀疏-孤独) medium(中等-饱满) dense(密集-巅峰)
- [COLOR:值]  音色: warm(温暖) bright(明亮) dark(暗沉)
- [KEY:值]    调性: major(大调-明亮) minor(小调-忧伤)
- [TEMPO:值]  速度: 60-70(慢) 70-90(中慢) 90-110(中速) 110+(快)
- [STEM:乐器] 主奏乐器: piano acoustic_guitar strings electric_guitar bass drums

Token 插入规则:
1. 每个段落[verse/chorus/bridge等]标签后紧随一行 Token 指令
2. Token 指令格式: [DYN:mp|ART:legato|TEX:sparse|COLOR:warm]
3. Token 选择要与歌词的情感走向一致
4. 副歌(chorus)的动态和织体要比主歌(verse)高 1-2 级
5. intro 使用最稀疏的配器，outro 回到稀疏

输出格式:
[section:verse]
[TOKEN:DYN:mp|ART:legato|TEX:sparse|COLOR:warm|TEMPO:80]
歌词行 1
歌词行 2
...

[section:chorus]
[TOKEN:DYN:f|ART:marcato|TEX:dense|COLOR:bright|TEMPO:100]
歌词行 1
歌词行 2
...

最后输出一个 JSON 全局配置:
```json
{"global_key": "minor", "global_tempo": 80, "mood": "wistful"}
```
"""

USER_PROMPT_TEMPLATE = """用户要求: {user_input}

请按照系统指令中的格式，创作这首歌曲的完整歌词和控制 Token。
注意:
1. 分析用户输入中的情感关键词，选择合适的 Token 值
2. 歌词至少包含: intro, verse1, chorus1, verse2, chorus2, bridge, outro
3. 每个段落都要有对应的 Token 指令行
4. 在 JSON 块中总结全局配置"""


# ══════════════════════════════════════════════════════════
# 示例输出（模拟 LLM 生成）
# ══════════════════════════════════════════════════════════

SAMPLE_OUTPUT = """[section:intro]
[TOKEN:DYN:p|ART:legato|TEX:sparse|COLOR:warm|TEMPO:72|STEM:acoustic_guitar]
（钢琴独奏，渐入）

[section:verse1]
[TOKEN:DYN:mp|ART:legato|TEX:sparse|COLOR:warm|TEMPO:75|STEM:acoustic_guitar]
教室的钟声 最后一次响起
黑板上还留着 没擦干净的公式
窗外梧桐叶 一片一片落地
我们就这样 各奔东西

[section:pre-chorus]
[TOKEN:DYN:mf|ART:legato|TEX:medium|COLOR:warm|TEMPO:78|STEM:piano+strings]
那些抄过的作业 借过的橡皮
变成了手机里 不敢翻的照片集

[section:chorus1]
[TOKEN:DYN:f|ART:marcato|TEX:dense|COLOR:bright|TEMPO:85|STEM:full_band]
再见了 亲爱的朋友
谢谢你 陪我这四年
青春是一趟 没有回程的列车
开走就不再 靠站

[section:verse2]
[TOKEN:DYN:mp|ART:legato|TEX:sparse|COLOR:warm|TEMPO:75|STEM:acoustic_guitar]
宿舍的卧谈 聊到凌晨三点
说好要一起 去看全世界的海面
可现在地图上 只有各自的航线
原来长大 就是学会说再见

[section:pre-chorus]
[TOKEN:DYN:mf|ART:legato|TEX:medium|COLOR:warm|TEMPO:78|STEM:piano+strings]
那件签满名字的班服 还压在箱底
穿不上了 也舍不得丢去

[section:chorus2]
[TOKEN:DYN:f|ART:marcato|TEX:dense|COLOR:bright|TEMPO:88|STEM:full_band]
再见了 亲爱的朋友
原谅我 转身时没回头
这校园的每个角落都有你
只是我们都 不能停留

[section:bridge]
[TOKEN:DYN:mp|ART:breathy|TEX:medium|COLOR:dark|TEMPO:70|STEM:strings]
若干年后 再回到这里
梧桐还是那棵 而你在哪里
那些没说出口的话 被风吹散
变成了这首歌 替我告诉你

[section:outro]
[TOKEN:DYN:p|ART:legato|TEX:sparse|COLOR:warm|TEMPO:65|STEM:acoustic_guitar]
（钢琴独奏，渐弱到无声）

```json
{
  "global_key": "C_minor",
  "global_tempo": 78,
  "global_time": "4/4",
  "mood": "wistful|nostalgic|bittersweet",
  "song_name": "梧桐那年"
}
```"""


# ══════════════════════════════════════════════════════════
# 解析器：将 LLM 输出转为结构化 JSON
# ══════════════════════════════════════════════════════════

def parse_llm_output(text: str) -> dict:
    """将 LLM 生成的带 Token 歌词解析为结构化 JSON"""
    sections = []
    current_section = None
    global_config = {}
    
    lines = text.strip().split("\n")
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 检测段落标签
        sec_match = re.match(r'\[section:(\w+)\]', line)
        if sec_match:
            if current_section:
                sections.append(current_section)
            current_section = {
                "section": sec_match.group(1),
                "tokens": {},
                "lyrics": [],
            }
            continue
        
        # 检测 Token 行
        if line.startswith("[TOKEN:") and current_section:
            token_str = line.replace("[TOKEN:", "").rstrip("]")
            for pair in token_str.split("|"):
                if ":" in pair:
                    k, v = pair.split(":", 1)
                    current_section["tokens"][k.strip()] = v.strip()
            continue
        
        # 检测 JSON 块
        if line == "```json":
            continue
        if line == "```":
            continue
        
        # 尝试解析 JSON
        if line.startswith("{") and current_section:
            try:
                global_config = json.loads(line)
            except json.JSONDecodeError:
                pass
            continue
        
        # 歌词行
        if current_section and not line.startswith("[") and not line.startswith("```"):
            # 跳过纯括号注释
            if not (line.startswith("（") and line.endswith("）")):
                current_section["lyrics"].append(line)
    
    if current_section:
        sections.append(current_section)
    
    return {
        "sections": sections,
        "global": global_config,
    }


# ══════════════════════════════════════════════════════════
# 架构图文字描述
# ══════════════════════════════════════════════════════════

ARCHITECTURE_DIAGRAM = r"""
╔══════════════════════════════════════════════════════════════╗
║           LLM 音乐指挥家 Agent Workflow                       ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  ┌─────────────────────┐                                     ║
║  │  用户大白话           │                                     ║
║  │  "写一首伤感的         │                                     ║
║  │   关于毕业的歌"        │                                     ║
║  └─────────┬───────────┘                                     ║
║            │                                                  ║
║  ┌─────────▼───────────┐                                     ║
║  │ LLM 第1步: 歌词创作   │  ← 标准 NLG 能力                    ║
║  │ · 分析情感关键词       │                                     ║
║  │ · 生成叙事结构          │                                     ║
║  │ · 创作分段歌词          │                                     ║
║  └─────────┬───────────┘                                     ║
║            │ 歌词文本                                          ║
║  ┌─────────▼───────────┐                                     ║
║  │ LLM 第2步: Token 注入 │  ← 核心创新点                        ║
║  │ · 情感→Token 映射表   │    伤感→minor+legato+mp              ║
║  │ · 段落→Token 规则库   │    verse→sparse, chorus→dense        ║
║  │ · 叙事弧线→动态曲线   │    intro(pp)→verse(mp)→chorus(f)     ║
║  │ · 意象→音色选择       │    回忆→warm, 希望→bright             ║
║  │ · 乐器→分轨配置       │    钢琴独奏→全乐队编制                ║
║  └─────────┬───────────┘                                     ║
║            │ 标注歌词                                           ║
║  ┌─────────▼───────────┐                                     ║
║  │ 带 Token 的歌词输出   │                                     ║
║  │                      │                                     ║
║  │ [section:verse1]     │                                     ║
║  │ [TOKEN:DYN:mp|       │                                     ║
║  │  ART:legato|          │                                     ║
║  │  TEX:sparse|          │                                     ║
║  │  COLOR:warm]          │                                     ║
║  │ 教室的钟声...         │                                     ║
║  │                      │                                     ║
║  │ [section:chorus1]    │                                     ║
║  │ [TOKEN:DYN:f|        │                                     ║
║  │  ART:marcato|         │                                     ║
║  │  TEX:dense|           │                                     ║
║  │  COLOR:bright]        │                                     ║
║  │ 再见了 亲爱的朋友...   │                                     ║
║  └─────────┬───────────┘                                     ║
║            │                                                  ║
║  ┌─────────▼───────────┐                                     ║
║  │ 下游消费              │                                     ║
║  │ → train_controlnet   │  训练时: Token行→TokenEncoder        ║
║  │ → MusicGen 推理       │  推理时: Token行→generate()条件     ║
║  │ → UI 展示             │  展示时: 歌词+Token可视化面板         ║
║  └─────────────────────┘                                     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""


# ══════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(ARCHITECTURE_DIAGRAM)
    
    print("\n" + "=" * 60)
    print("示例: 用户输入 → LLM 输出")
    print("=" * 60)
    print(f"\n用户输入: '写一首伤感的关于毕业的歌'\n")
    print(SAMPLE_OUTPUT)
    
    print("\n" + "=" * 60)
    print("解析后结构化 JSON")
    print("=" * 60)
    parsed = parse_llm_output(SAMPLE_OUTPUT)
    print(json.dumps(parsed, ensure_ascii=False, indent=2))
    
    print("\n" + "=" * 60)
    print("情感→Token 映射表 (部分)")
    print("=" * 60)
    for emotion, tokens in list(EMOTION_MAP.items())[:5]:
        print(f"  {emotion}: {tokens}")
    
    print("\n" + "=" * 60)
    print("段落→Token 映射表")
    print("=" * 60)
    for section, tokens in SECTION_MAP.items():
        print(f"  {section:12s}: {tokens}")
