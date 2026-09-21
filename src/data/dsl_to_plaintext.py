# -*- coding: utf-8 -*-
"""
dsl_to_plaintext.py — 将含 6 轴 Music-Prompt DSL 的测试集 Prompt 翻译为纯自然语言，
供不支持 DSL 的 Baseline（MusicGen / AudioLDM 2 / Stable Audio Open）使用。

用法：
    python dsl_to_plaintext.py --input guofeng_test.json --output baseline_prompts.json
    python dsl_to_plaintext.py --input guofeng_test.csv  --output baseline_prompts.json
    python dsl_to_plaintext.py --input guofeng_test.json --output baseline_prompts.json --jsonl

输入格式（自动识别）：
    JSON  : 一个 list，或形如 {"samples": [...]} / {"data": [...]} 的包一层 list 的对象；
            每条记录中含 prompt 的字段按优先级自动探测：
            prompt > dsl > tokens > text > input > instruction
    JSONL : 每行一个 JSON 对象
    CSV   : 第一行为表头（必须包含 prompt 相关列）

翻译示例：
    输入 : "[TEX:sparse] | [DYN:p] | [COLOR:warm] A solo erhu playing a sad and slow melody."
    输出 : "Sparse texture, soft dynamics, warm timbre. A solo erhu playing a sad and slow melody."

规则：
    - 所有能翻译的 [AXIS:value] token 被替换为自然语言短语，按出现顺序用 ", " 连接；
    - 复合 token（如 [PATTERN:auto|DYN:ff|ART:staccato|TEX:polyphonic|VOICES:4]）
      会被拆成多个 AXIS:value 分别翻译；
    - 无法翻译的方括号 token（如 [SEC:intro|BAR:1-8]）会被丢弃，保证 baseline
      不接触任何 DSL 符号；
    - 剩余自然语言原样保留，并清理多余的空白与分隔符。

输出：
    baseline_prompts.json —— list，每条包含：
      {"id": 序号,
       "baseline_prompt": 翻译后的纯自然语言,
       "text": 同 baseline_prompt（兼容现有 baseline_inference.py 的 "text" 字段）,
       "original_prompt": 翻译前原文,
       ... 以及输入记录里的其他字段}
    若加 --jsonl，同时输出 baseline_prompts.jsonl（每行 {"text": ...}），
    可直接被现有 baseline_inference.py 读取。
"""
import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ─────────────────────────────────────────────────────────────
# 6 轴 DSL → 自然语言词典（值全部使用句子首字母大写短语）
# ─────────────────────────────────────────────────────────────
TEX_GLOSS = {
    "sparse": "sparse texture",
    "dense": "dense texture",
    "medium": "medium texture",
    "polyphonic": "polyphonic texture",
    "monophonic": "monophonic texture",
}

DYN_GLOSS = {
    "pp": "very soft dynamics",
    "p": "soft dynamics",
    "mp": "moderately soft dynamics",
    "mf": "moderately loud dynamics",
    "f": "loud dynamics",
    "ff": "very loud dynamics",
    "cresc": "a gradual crescendo",
    "dim": "a gradual decrescendo",
}

COLOR_GLOSS = {
    "warm": "warm timbre",
    "bright": "bright timbre",
    "cold": "cold timbre",
    "dark": "dark timbre",
}

ART_GLOSS = {
    "legato": "legato articulation",
    "staccato": "staccato articulation",
    "marcato": "marcato articulation",
    "portato": "portato articulation",
}

CHORD_QUALITY = {
    "maj": "major",
    "min": "minor",
    "min7": "minor seventh",
    "maj7": "major seventh",
    "7": "dominant seventh",
    "dim": "diminished",
    "aug": "augmented",
    "sus2": "suspended second",
    "sus4": "suspended fourth",
}


def _note_name(root: str) -> str:
    """把 C / C# / Db 之类写成可读拼写。"""
    root = root.strip()
    if len(root) >= 2 and root[1] == "#":
        return f"{root[0]}-sharp"
    if len(root) >= 2 and root[1] == "b":
        return f"{root[0]}-flat"
    return root


def _translate_axis(axis: str, value: str):
    """把单个 AXIS:value 翻译成英文短语；无法翻译时返回 None。"""
    axis = axis.strip().upper()
    value = value.strip()

    if axis == "TEX":
        return TEX_GLOSS.get(value.lower(), f"{value} texture")

    if axis == "VOICES":
        v = value.lower()
        if v == "1-2":
            return "one to two voices"
        if v == "8+":
            return "eight or more voices"
        if v.isdigit():
            return f"{int(v)} voices"
        return f"{value} voices"

    if axis == "DYN":
        return DYN_GLOSS.get(value.lower(), f"{value} dynamics")

    if axis == "COLOR":
        return COLOR_GLOSS.get(value.lower(), f"{value} timbre")

    if axis == "TEMPO":
        v = value.lower()
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", v)
        if m:
            return f"a tempo between {m.group(1)} and {m.group(2)} BPM"
        if v.isdigit():
            return f"a tempo around {int(v)} BPM"
        return f"a tempo of {value}"

    if axis == "ART":
        return ART_GLOSS.get(value.lower(), f"{value} articulation")

    if axis == "CHORD":
        # 支持 C / Am / F#:min7 / Db:maj 等
        m = re.fullmatch(r"([A-Ga-g][#b]?)(?::(\w+))?", value)
        if m:
            root, qual = m.group(1), (m.group(2) or "maj")
            qual_name = CHORD_QUALITY.get(qual.lower(), qual.lower())
            return f"{_note_name(root)} {qual_name} harmony"
        return f"{value} harmony"

    if axis == "KEY":
        m = re.fullmatch(r"([A-Ga-g][#b]?)_(major|minor)", value)
        if m:
            return f"{_note_name(m.group(1))} {m.group(2)} key"
        return f"{value.replace('_', ' ')} key"

    if axis == "STYLE":
        return f"{value} style"

    if axis == "STEM":
        return f"featuring {value}"

    if axis == "MOOD":
        return f"a {value} mood"

    if axis == "REG":
        v = value.lower()
        return "a low register" if v == "low" else ("a high register" if v == "high" else f"a {value} register")

    if axis in ("TIME", "METER"):
        return f"{value} time signature"

    # PATTERN / SEC / BAR / GLOBAL / EXPERT 等结构性 token 不进 baseline prompt
    return None


def compile_prompt(raw: str) -> str:
    """把含 DSL 的 prompt 翻译为纯自然语言。"""
    if not isinstance(raw, str):
        return raw
    raw = raw.strip()

    token_re = re.compile(r"\[([^\[\]]+)\]")
    glosses = []
    remainder = token_re.sub("", raw)  # 先去掉所有方括号 token

    for match in token_re.finditer(raw):
        content = match.group(1)
        for part in content.split("|"):
            part = part.strip()
            if ":" not in part:
                continue
            axis, value = part.split(":", 1)
            gloss = _translate_axis(axis, value)
            if gloss:
                glosses.append(gloss)

    # 清理剩余文本：去掉 token 移除后残留的 "|"、";"、","、句点与多余空格
    remainder = re.sub(r"^[\s\|\;,\.]+", "", remainder)
    remainder = re.sub(r"\s+", " ", remainder).strip()

    if not glosses:
        return remainder

    gloss_text = ", ".join(glosses)
    gloss_text = gloss_text[0].upper() + gloss_text[1:]
    if remainder:
        return f"{gloss_text}. {remainder}"
    return f"{gloss_text}."


PROMPT_FIELD_PRIORITY = ["prompt", "dsl", "tokens", "text", "input", "instruction"]


def compose_baseline_prompt(record: dict):
    """返回 (baseline_prompt, original_prompt, translated_bool)。

    规则（用于 guofeng 测试集这类 {"tokens": DSL, "text": 自然语言} 的记录）：
    - 若同时存在 tokens/dsl 与 text，则 baseline_prompt = DSL 翻译出的 glosses + 自然语言 text，
      两者内容互补，任何 baseline 都不丢失语义；
    - 否则按 PROMPT_FIELD_PRIORITY 取单一字段走 compile_prompt。
    """
    has_dsl = any(k in record and isinstance(record[k], str) and record[k].strip()
                  for k in ("tokens", "dsl"))
    natural = record.get("text")
    natural_ok = isinstance(natural, str) and natural.strip()

    if has_dsl and not natural_ok:
        field, raw = find_prompt_field(record)
        out = compile_prompt(raw)
        return out, raw, (out != raw)

    if has_dsl and natural_ok:
        dsl_field = "tokens" if "tokens" in record else "dsl"
        gloss_text = _glosses_only(record[dsl_field])
        combined = gloss_text + natural.strip() if gloss_text else natural.strip()
        return combined, record[dsl_field], True

    field, raw = find_prompt_field(record)
    out = compile_prompt(raw)
    return out, raw, (out != raw)


def _glosses_only(dsl: str) -> str:
    """只提取 DSL 里的可翻译 token，拼成首字母大写的短语串（句号结尾，空则返回 ""）。"""
    token_re = re.compile(r"\[([^\[\]]+)\]")
    glosses = []
    for match in token_re.finditer(dsl):
        content = match.group(1)
        for part in content.split("|"):
            part = part.strip()
            if ":" not in part:
                continue
            axis, value = part.split(":", 1)
            gloss = _translate_axis(axis, value)
            if gloss:
                glosses.append(gloss)
    if not glosses:
        return ""
    text = ", ".join(glosses)
    return text[0].upper() + text[1:] + ". "


def find_prompt_field(record: dict):
    """按优先级返回记录中的 prompt 字段名与内容。"""
    for key in PROMPT_FIELD_PRIORITY:
        if key in record and isinstance(record[key], str) and record[key].strip():
            return key, record[key]
    # 兜底：任意看起来像 prompt 的字符串字段
    for key, val in record.items():
        if isinstance(val, str) and len(val) > 3:
            return key, val
    return None, None


def load_records(path: str):
    """读 JSON / JSONL / CSV，统一返回 list[dict]。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"找不到输入文件: {path}")

    if p.suffix.lower() == ".csv":
        with open(p, encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))

    text = p.read_text(encoding="utf-8")

    # .jsonl：逐行解析（优先于整体 JSON 解析，避免 Extra data 报错）
    if p.suffix.lower() == ".jsonl":
        records = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records

    stripped = text.lstrip()
    if stripped.startswith("["):
        return json.loads(text)
    if stripped.startswith("{"):
        # 可能是单个对象，也可能是包了一层的 {"samples": [...]}
        obj = json.loads(text)
        if isinstance(obj, dict):
            for key in ("samples", "data", "items", "prompts", "test"):
                if isinstance(obj.get(key), list):
                    return obj[key]
            # 任意 list-of-dict 值都当作样本列表
            for val in obj.values():
                if isinstance(val, list) and val and isinstance(val[0], dict):
                    return val
            return [obj]
        return obj

    # JSONL
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def main():
    ap = argparse.ArgumentParser(description="6 轴 DSL prompt → baseline 自然语言 prompt")
    ap.add_argument("--input", default="guofeng_test.json", help="测试集 JSON/JSONL/CSV 路径")
    ap.add_argument("--output", default="baseline_prompts.json", help="输出 JSON 路径")
    ap.add_argument("--jsonl", action="store_true", help="同时输出 baseline_prompts.jsonl（兼容 baseline_inference.py）")
    args = ap.parse_args()

    records = load_records(args.input)
    if not records:
        print(f"警告: {args.input} 中没有读取到任何记录")
        return

    out_records = []
    n_translated = 0
    for i, rec in enumerate(records):
        baseline_prompt, raw, translated = compose_baseline_prompt(rec)
        if not baseline_prompt:
            print(f"警告: 第 {i} 条记录没有可用的 prompt 字段，跳过")
            continue
        if translated:
            n_translated += 1
        out = dict(rec)                      # 保留 track_name / bpm 等元数据
        out["id"] = i
        out["original_prompt"] = raw
        out["baseline_prompt"] = baseline_prompt
        out["text"] = baseline_prompt        # 兼容 baseline_inference.py
        out_records.append(out)

    out_path = Path(args.output)
    out_path.write_text(
        json.dumps(out_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"共处理 {len(out_records)} 条，其中 {n_translated} 条含 DSL 已翻译")
    print(f"已写出: {out_path}")

    if args.jsonl:
        jl_path = out_path.with_suffix(".jsonl")
        with open(jl_path, "w", encoding="utf-8") as f:
            for rec in out_records:
                f.write(json.dumps({"text": rec["baseline_prompt"]}, ensure_ascii=False) + "\n")
        print(f"已写出: {jl_path}（每行 {{\"text\": ...}}，可直接喂 baseline_inference.py）")


if __name__ == "__main__":
    main()
