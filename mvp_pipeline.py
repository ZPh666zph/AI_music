#!/usr/bin/env python3
"""
mvp_pipeline.py — 小闭环 MVP: 大白话 → LLM → Token → MusicGen → .wav
=====================================================================
用法:
  1. python mvp_pipeline.py                          # 交互式输入
  2. python mvp_pipeline.py --text "凄美古风离别曲"   # 命令行输入
  3. python mvp_pipeline.py --text "热血摇滚" --output rock.wav

API:
  DeepSeek API (免费额度) — OpenAI 兼容协议
  设置环境变量: set DEEPSEEK_API_KEY=sk-xxxx

注意:
  本地 MusicGen 使用 CPU 模式 (RTX 5060 Blackwell 不兼容)
  首次运行会自动下载模型权重 (~6GB)
"""

import os, sys, json, re, time, argparse
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# ═══════════════════════════════════════════════════════════
# 0. 配置
# ═══════════════════════════════════════════════════════════

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL_NAME = "deepseek-chat"       # DeepSeek-V3
MUSICGEN_MODEL = "facebook/musicgen-medium"
OUTPUT_DIR = "C:/Deepseek/outputs/mvp"

# ═══════════════════════════════════════════════════════════
# 1. LLM Prompt 模板 (复用 llm_conductor.py 的设计)
# ═══════════════════════════════════════════════════════════

SYSTEM_PROMPT = """你是一位 AI 音乐指挥家。根据用户的大白话需求，生成一首完整歌曲的结构化控制指令。

你掌握的六维 Token:
- DYN: pp(极弱) p(弱) mp(中弱) mf(中强) f(强) ff(极强)
- ART: legato(连奏-流畅) staccato(断奏-跳跃) marcato(强奏-宣言) breathy(气声)
- TEX: sparse(稀疏-孤独) medium(中等) dense(密集-巅峰)
- COLOR: warm(温暖) bright(明亮) dark(暗沉)
- STYLE: gufeng(古风) folk(民谣) pop(流行) rock(摇滚)
- TEMPO: 60-90(慢) 90-120(中) 120+(快)
- KEY: major(大调-明亮) minor(小调-忧伤)
- STEM: acoustic_guitar piano strings erhu guzheng pipa dizi full_band

规则:
1. intro 用最弱动态(p) + 最稀疏织体(sparse)
2. verse 用中弱(mp) + 稀疏(sparse) + legato
3. chorus 用强(f/ff) + 密集(dense) + marcato
4. 叙事弧线: intro(p)→verse(mp)→pre-chorus(mf)→chorus(f)→bridge(mp)→outro(p)
5. 古风必须用 minor 调式 + 五声音阶色彩, 民谣用 major + warm
6. 歌词用中文，每段 4 行

输出格式 (严格 JSON, 不要 markdown 代码块):
{
  "song_name": "...",
  "global_style": "gufeng",
  "global_key": "A_minor",
  "global_tempo": 72,
  "global_mood": "凄美, 哀而不伤",
  "text_prompt": "简短的自然语言全局描述(20字以内)",
  "sections": [
    {
      "section": "intro",
      "tokens": {"DYN":"p","ART":"legato","TEX":"sparse","COLOR":"warm","STEM":"guzheng"},
      "lyrics": ""
    },
    {
      "section": "verse1",
      "tokens": {"DYN":"mp","ART":"legato","TEX":"sparse","COLOR":"warm","STEM":"erhu"},
      "lyrics": "歌词行1\\n歌词行2\\n歌词行3\\n歌词行4"
    }
  ]
}
"""

USER_PROMPT_TEMPLATE = "用户需求: {user_input}\n\n请生成完整的歌曲结构 JSON。"


# ═══════════════════════════════════════════════════════════
# 2. LLM 调用 (DeepSeek API)
# ═══════════════════════════════════════════════════════════

def call_deepseek(user_input: str, api_key: str = None) -> dict:
    """调用 DeepSeek API, 返回结构化 JSON"""
    import urllib.request, urllib.error
    
    api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "请设置 DeepSeek API Key:\n"
            "  1. 访问 https://platform.deepseek.com 注册\n"
            "  2. 设置环境变量: set DEEPSEEK_API_KEY=sk-xxxx\n"
            "  3. 或传参: python mvp_pipeline.py --api-key sk-xxxx"
        )
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(user_input=user_input)},
        ],
        "temperature": 0.7,
        "max_tokens": 2048,
        "response_format": {"type": "json_object"},  # DeepSeek 支持 JSON mode
    }
    
    req = urllib.request.Request(
        DEEPSEEK_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    
    print(f"  [LLM] 调用 DeepSeek API ({MODEL_NAME})...")
    t0 = time.time()
    
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        raise RuntimeError(f"API 错误 ({e.code}): {body}")
    
    elapsed = time.time() - t0
    content = data["choices"][0]["message"]["content"]
    print(f"  [LLM] 完成, 耗时 {elapsed:.1f}s, {len(content)} 字符")
    
    # 解析 JSON
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        # 尝试提取 JSON 块
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            result = json.loads(match.group())
        else:
            raise RuntimeError(f"LLM 输出非 JSON:\n{content[:500]}")
    
    return result


# ═══════════════════════════════════════════════════════════
# 3. 本地 LLM fallback (离线模板 — 无 API 时使用)
# ═══════════════════════════════════════════════════════════

def offline_template(user_input: str) -> dict:
    """无需 API 的离线模板，基于关键词匹配"""
    # 情感关键词检测
    keywords = {
        "古风": {"style": "gufeng", "key": "A_minor", "stem": "guzheng", "tempo": 72},
        "凄美": {"color": "dark", "mood": "凄美, 哀而不伤"},
        "热血": {"style": "rock", "key": "E_major", "stem": "electric_guitar", "tempo": 140},
        "伤感": {"key": "C_minor", "mood": "伤感, 怀念"},
        "民谣": {"style": "folk", "key": "C_major", "stem": "acoustic_guitar", "tempo": 80},
        "毕业": {"mood": "青春, 怀念"},
        "离别": {"mood": "不舍, 祝福"},
        "摇滚": {"style": "rock", "key": "E_major", "stem": "electric_guitar", "tempo": 140},
    }
    
    style = "pop"
    key = "C_major"
    stem = "piano"
    tempo = 90
    mood = ""
    color = "warm"
    
    for kw, vals in keywords.items():
        if kw in user_input:
            for k, v in vals.items():
                if k == "style": style = v
                elif k == "key": key = v
                elif k == "stem": stem = v
                elif k == "tempo": tempo = v
                elif k == "mood": mood = v
                elif k == "color": color = v
    
    print(f"  [离线模式] 检测到: style={style}, key={key}, stem={stem}, tempo={tempo}")
    
    sections = [
        {"section": "intro", "tokens": {"DYN": "p", "ART": "legato", "TEX": "sparse", "COLOR": color, "TEMPO": str(tempo-5), "STEM": stem}, "lyrics": ""},
        {"section": "verse1", "tokens": {"DYN": "mp", "ART": "legato", "TEX": "sparse", "COLOR": color, "TEMPO": str(tempo), "STEM": stem}, "lyrics": "第一段歌词\\n第二行\\n第三行\\n第四行"},
        {"section": "pre-chorus", "tokens": {"DYN": "mf", "ART": "legato", "TEX": "medium", "COLOR": color, "TEMPO": str(tempo+3), "STEM": stem+"+strings"}, "lyrics": "预备副歌\\n情绪爬升"},
        {"section": "chorus1", "tokens": {"DYN": "f", "ART": "marcato", "TEX": "dense", "COLOR": "bright", "TEMPO": str(tempo+10), "STEM": "full_band"}, "lyrics": "副歌第一行\\n高潮\\n记忆点\\n重复"},
        {"section": "verse2", "tokens": {"DYN": "mp", "ART": "legato", "TEX": "sparse", "COLOR": color, "TEMPO": str(tempo), "STEM": stem}, "lyrics": "第二段主歌\\n叙述继续"},
        {"section": "bridge", "tokens": {"DYN": "mp", "ART": "breathy", "TEX": "medium", "COLOR": "dark", "TEMPO": str(tempo-5), "STEM": "strings"}, "lyrics": "桥段转折\\n反思"},
        {"section": "outro", "tokens": {"DYN": "p", "ART": "legato", "TEX": "sparse", "COLOR": color, "TEMPO": str(tempo-10), "STEM": stem}, "lyrics": ""},
    ]
    
    return {
        "song_name": f"{user_input[:10]}...",
        "global_style": style,
        "global_key": key,
        "global_tempo": tempo,
        "global_mood": mood or user_input,
        "text_prompt": f"{style} style, {key}, {tempo}bpm, {stem}",
        "sections": sections,
    }


# ═══════════════════════════════════════════════════════════
# 4. Token 序列组装
# ═══════════════════════════════════════════════════════════

def assemble_token_from_json(llm_output: dict) -> str:
    """将 LLM JSON 转为 PDF 策略 B 格式的 Token 序列"""
    token_lines = []
    
    # 全局
    g = llm_output
    token_lines.append(f"[GLOBAL:STYLE:{g.get('global_style','pop')}]")
    token_lines.append(f"[GLOBAL:KEY:{g.get('global_key','C_major')}]")
    token_lines.append(f"[GLOBAL:TEMPO:{g.get('global_tempo',90)}]")
    token_lines.append(f"[GLOBAL:MOOD:{g.get('global_mood','')}]")
    token_lines.append("")
    
    # 段落
    bar = 1
    for sec in g.get("sections", []):
        tokens = sec.get("tokens", {})
        token_lines.append(f"[SEC:{sec['section']}|BAR:{bar}-{bar+7}]")
        
        stem = tokens.get("STEM", "piano")
        token_lines.append(f"  [STEM:{stem}]")
        
        # 组装 PATTERN 行
        dyn = tokens.get("DYN", "mf")
        art = tokens.get("ART", "legato")
        tex = tokens.get("TEX", "medium")
        color = tokens.get("COLOR", "warm")
        tempo = tokens.get("TEMPO", "90")
        
        pattern = f"[PATTERN:auto|DYN:{dyn}|ART:{art}|TEX:{tex}|COLOR:{color}|TEMPO:{tempo}]"
        token_lines.append(f"    {pattern}")
        token_lines.append(f"  [/STEM]")
        token_lines.append("")
        
        bar += 8
    
    return "\n".join(token_lines)


# ═══════════════════════════════════════════════════════════
# 5. MusicGen 生成
# ═══════════════════════════════════════════════════════════

def generate_music(text_prompt: str, output_path: str, duration_sec: int = 15) -> str:
    """使用本地 MusicGen-medium 生成音频 (CPU 模式)"""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 强制 CPU
    
    import torch
    import soundfile as sf
    from transformers import MusicgenForConditionalGeneration, AutoProcessor
    
    print(f"  [MusicGen] 加载模型 {MUSICGEN_MODEL}...")
    t0 = time.time()
    
    model = MusicgenForConditionalGeneration.from_pretrained(MUSICGEN_MODEL)
    processor = AutoProcessor.from_pretrained(MUSICGEN_MODEL)
    
    print(f"  [MusicGen] 模型就绪, 耗时 {time.time()-t0:.0f}s")
    
    # Token 估算: 1 token ≈ 0.02s, duration_sec 秒 ≈ duration_sec * 50 tokens
    max_new_tokens = int(duration_sec * 50)
    
    print(f"  [MusicGen] 生成中, 目标 {duration_sec}s...")
    t0 = time.time()
    
    inputs = processor(text=[text_prompt], padding=True, return_tensors="pt")
    with torch.no_grad():
        audio = model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    elapsed = time.time() - t0
    actual_dur = audio.shape[2] / 32000
    print(f"  [MusicGen] 完成, 耗时 {elapsed:.0f}s ({actual_dur:.1f}s 音频)")
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sf.write(output_path, audio[0, 0].numpy(), 32000)
    print(f"  [MusicGen] 保存: {output_path}")
    
    return output_path


# ═══════════════════════════════════════════════════════════
# 6. 主流程
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MVP: 大白话 → 音乐")
    parser.add_argument("--text", type=str, help="用户大白话输入")
    parser.add_argument("--output", type=str, default=None, help="输出 .wav 路径")
    parser.add_argument("--api-key", type=str, default=None, help="DeepSeek API Key")
    parser.add_argument("--offline", action="store_true", help="离线模板模式 (无需 API)")
    parser.add_argument("--duration", type=int, default=15, help="生成音频时长(秒,默认15)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("MVP Pipeline: 大白话 → LLM → Token → MusicGen → .wav")
    print("=" * 60)
    
    # ── 1. 用户输入 ──
    user_input = args.text
    if not user_input:
        print("\n输入你的音乐需求 (如: 写一首凄美的古风离别曲):")
        user_input = input("> ").strip()
        if not user_input:
            user_input = "写一首凄美的古风离别曲"
            print(f"使用默认: {user_input}")
    
    print(f"\n用户输入: {user_input}")
    
    # ── 2. LLM 生成 Token ──
    print(f"\n[Step 1/3] LLM 生成 Token...")
    
    if args.offline:
        llm_output = offline_template(user_input)
    else:
        try:
            llm_output = call_deepseek(user_input, args.api_key)
        except RuntimeError as e:
            print(f"  ⚠️  API 不可用: {e}")
            print(f"  → 切换离线模板模式")
            llm_output = offline_template(user_input)
    
    print(f"\n  LLM 输出:")
    print(f"    歌名:    {llm_output.get('song_name', '')}")
    print(f"    风格:    {llm_output.get('global_style', '')}")
    print(f"    调性:    {llm_output.get('global_key', '')}  BPM:{llm_output.get('global_tempo', '')}")
    print(f"    段落数:  {len(llm_output.get('sections', []))}")
    
    # ── 3. Token 组装 ──
    print(f"\n[Step 2/3] Token 组装...")
    token_seq = assemble_token_from_json(llm_output)
    
    print(f"\n  Token 序列 (前 500 字符):")
    for line in token_seq.split("\n")[:15]:
        print(f"    {line}")
    print(f"    ... (共 {len(token_seq)} 字符)")
    
    # ── 4. 保存 Token JSON ──
    output_dir = os.path.dirname(args.output) or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    song_name = re.sub(r'[\\/:*?"<>|]', '_', llm_output.get("song_name", "mvp"))
    token_json_path = os.path.join(output_dir, f"{song_name}_tokens.json")
    with open(token_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "user_input": user_input,
            "llm_output": llm_output,
            "token_sequence": token_seq,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  Token JSON 已保存: {token_json_path}")
    
    # ── 5. MusicGen 生成 ──
    print(f"\n[Step 3/3] MusicGen 生成...")
    output_path = args.output or os.path.join(output_dir, f"{song_name}.wav")
    
    text_prompt = llm_output.get(
        "text_prompt",
        f"{llm_output.get('global_style','')}, {llm_output.get('global_key','')}, {llm_output.get('global_tempo',90)}bpm"
    )
    print(f"  Text prompt: {text_prompt}")
    
    generate_music(text_prompt, output_path, args.duration)
    
    # ── 6. 完成 ──
    print(f"\n{'=' * 60}")
    print(f"MVP Pipeline 完成!")
    print(f"  Token JSON: {token_json_path}")
    print(f"  音频文件:   {output_path}")
    print(f"{'=' * 60}")


# ═══════════════════════════════════════════════════════════
# 附录: 低成本获取高质量国风多轨语料的三个策略
# ═══════════════════════════════════════════════════════════

DATA_STRATEGIES = r"""
╔══════════════════════════════════════════════════════════════════╗
║   三个极具操作性的低成本国风多轨语料获取策略                        ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  策略 1: MIDI + 民乐 SF2 合成 (零成本)                            ║
║  ─────────────────────────────────────                            ║
║  工具:   MuseScore (开源) + FluidSynth + 民乐 SoundFont           ║
║  数据源: midi.org, MuseScore 社区, Bilibili MIDI 翻奏             ║
║  操作:                                                        ║
║    1) 搜集古风/民乐 MIDI (500+ 首社区资源)                        ║
║    2) 用 FluidSynth + 古筝/琵琶/二胡/笛子/箫 SF2 音源渲染         ║
║    3) 每轨单独渲染 → 天然获得分轨音频                              ║
║    4) 五声音阶过滤器: 移除非宫商角徵羽的音符                        ║
║    5) 经 slakh_to_tokens.py 批量转换为 Token                       ║
║  产出:   500 组 (Text + Token + 分轨 Audio) 三元组                 ║
║  成本:   零 (全部开源工具)                                         ║
║  时间:   2-3 天 (自动化脚本 + 人工抽检)                             ║
║                                                                  ║
║  策略 2: 腾讯 SongPrep-7B + Bilibili 爬取 (近乎零成本)              ║
║  ─────────────────────────────────────────────                      ║
║  工具:   SongPrep-7B (腾讯混元开源 7B 音乐预处理模型)               ║
║  数据源: Bilibili/抖音 民乐翻奏/独奏视频 (youtube-dl/you-get)      ║
║  操作:                                                        ║
║    1) 爬取古筝/琵琶/二胡独奏或合奏视频 (严格标注 CC 许可)          ║
║    2) 用 SongPrep-7B 自动提取:                                     ║
║       · 歌曲结构 (intro/verse/chorus/bridge)                      ║
║       · 和弦进行                                                    ║
║       · 声源分离 (人声/鼓/贝斯/其他)  → 自动分轨                     ║
║    3) LLM 辅助生成古风风格文本描述                                   ║
║  产出:   200-500 组真实演奏的三元组                                   ║
║  成本:   零 (模型免费 + 视频公开)                                    ║
║  时间:   1 周 (爬取 + 标注 + 清洗)                                   ║
║  注意:   仅用于学术研究，需尊重原视频版权                              ║
║                                                                  ║
║  策略 3: 现有 MIDI 数据集 + 乐器重映射 + 数据增强 (零成本)           ║
║  ───────────────────────────────────────────────────                ║
║  工具:   pretty_midi + FluidSynth + audiomentations               ║
║  数据源: Lakh MIDI (176K MIDI, CC-BY), MAESTRO (钢琴, CC-BY-NC)    ║
║  操作:                                                        ║
║    1) 从 Lakh MIDI 筛选符合 4/4拍 + 中慢速 (60-100 BPM) 的条目      ║
║    2) 乐器重映射: Piano→Guzheng, Violin→Erhu, Flute→Dizi          ║
║    3) 五声音阶约束 + 中国调式 bias (将 MIDI 约束到宫商角徵羽)        ║
║    4) 数据增强 × 3: pitch shift ±3半音, time stretch 0.9-1.1x     ║
║    5) LLM 批量生成古风中文描述 (用模板: "{乐器}独奏, {调式}, 古风")  ║
║  产出:   1000+ 组高质量古风三元组                                     ║
║  成本:   零                                                         ║
║  时间:   3-4 天                                                      ║
║                                                                  ║
║  综合建议:                                                      ║
║    策略 1 (MIDI+SF2) 做首选 → 质量可控, 分轨天然                    ║
║    策略 3 (Lakh MIDI 重映射) 做规模 → 数据量最大                     ║
║    策略 2 (SongPrep-7B) 做真实性 → 引入真人演奏质感                   ║
║    三策略组合 → 1000-2000 组高质量国风多轨语料                        ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""


if __name__ == "__main__":
    print(DATA_STRATEGIES)
    main()
