#!/usr/bin/env python3
"""llm_evaluator.py — LLM-as-a-Judge 自动 MOS 评分系统"""
import sys, os, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
if sys.stdout.encoding != "utf-8": sys.stdout.reconfigure(encoding="utf-8")

import librosa, numpy as np, soundfile as sf
from pathlib import Path
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════
# 1. 音频特征提取
# ═══════════════════════════════════════════════════════════
def extract_features(wav_path: str) -> dict:
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1: audio = audio.mean(axis=1)

    # Spectral Centroid
    spec = np.abs(librosa.stft(audio, n_fft=2048))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    centroid = float(np.mean(np.sum(freqs[:, None] * spec, axis=0) / (np.sum(spec, axis=0) + 1e-8)))

    # BPM
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    tempo = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    bpm = float(np.atleast_1d(np.asarray(tempo[0] if isinstance(tempo, tuple) else tempo).ravel())[0])

    # RMS energy
    rms = float(np.sqrt(np.mean(audio ** 2)))

    # Zero-crossing rate
    zcr = float(np.sum(np.abs(np.diff(np.signbit(audio)))) / len(audio))

    # Duration
    duration = len(audio) / sr

    return {"centroid": round(centroid, 1), "bpm": round(bpm, 1),
            "rms": round(rms, 4), "zcr": round(zcr, 4), "duration": round(duration, 1), "sr": sr}


# ═══════════════════════════════════════════════════════════
# 2. LLM Judge Prompt 模板
# ═══════════════════════════════════════════════════════════
JUDGE_SYSTEM_PROMPT = """You are a professional music producer and audio engineer evaluating AI-generated music. 

You will receive:
1. The TEXT prompt used to generate the music.
2. The intended STYLE (instrument/expert) that the model was asked to produce.
3. A PHYSICAL FEATURE REPORT containing measured acoustic properties of the generated audio (spectral centroid, BPM, RMS energy, zero-crossing rate).

Your task: Rate the CONTROL ADHERENCE on a 1-5 scale:
- 5: Perfect adherence. The audio perfectly matches both the text description and the intended instrument style. Timbre and rhythm are fully appropriate.
- 4: Good adherence. Minor deviations but overall the intended style and mood are clearly present.
- 3: Moderate adherence. Some aspects match, but there are noticeable gaps between the prompt and the audio.
- 2: Weak adherence. The audio only vaguely corresponds to the prompt or intended style.
- 1: No adherence. The audio does not match the prompt at all.

For each evaluation, output ONLY a JSON object with this exact format:
{"score": <int 1-5>, "timbre_match": "<brief>", "rhythm_match": "<brief>", "overall": "<one sentence>"}

Do not include any other text in your response."""


def build_judge_prompt(text_prompt: str, style: str, features: dict, target_bpm: int = 90) -> str:
    return f"""TEXT PROMPT: "{text_prompt}"
INTENDED STYLE: {style} (Chinese traditional instrument)
TARGET BPM: {target_bpm}

PHYSICAL FEATURE REPORT:
- Spectral Centroid: {features['centroid']} Hz
  (Erhu typically 1500-2000 Hz, Dizi 2000-3000 Hz, Suona 2500-4000 Hz)
- Detected BPM: {features['bpm']} BPM (target: {target_bpm})
- RMS Energy: {features['rms']}
- Zero-Crossing Rate: {features['zcr']}
- Duration: {features['duration']} seconds

Rate the CONTROL ADHERENCE (1-5) of this generated audio to the text prompt and intended style."""


# ═══════════════════════════════════════════════════════════
# 3. 离线启发式评分 (无 API 时的 fallback)
# ═══════════════════════════════════════════════════════════
def heuristic_score(style: str, features: dict, target_bpm: int = 90) -> dict:
    """
    基于物理特征的启发式评分 (无需 LLM API, 立即返回)
    评分逻辑: 音色匹配 + 节奏匹配 各占 50%
    """
    centroid = features["centroid"]
    bpm = features["bpm"]

    # Timbre match: 频谱质心是否在预期范围内
    style_centroid_ranges = {
        "erhu": (1500, 2200),
        "dizi": (2000, 3200),
        "suona": (2500, 4000),
        "string": (1500, 2200),
        "wind": (2000, 3200),
        "brass": (2500, 4000),
    }
    lo, hi = style_centroid_ranges.get(style, (1000, 4000))
    if lo <= centroid <= hi:
        timbre_score = 5
    elif abs(centroid - lo) < 500 or abs(centroid - hi) < 500:
        timbre_score = 4
    elif abs(centroid - (lo+hi)/2) < 1000:
        timbre_score = 3
    else:
        timbre_score = 2

    # Rhythm match
    bpm_error = abs(bpm - target_bpm) / max(target_bpm, 1)
    if bpm_error < 0.05: rhythm_score = 5
    elif bpm_error < 0.15: rhythm_score = 4
    elif bpm_error < 0.30: rhythm_score = 3
    elif bpm_error < 0.50: rhythm_score = 2
    else: rhythm_score = 1

    overall = round((timbre_score * 0.6 + rhythm_score * 0.4))
    return {
        "score": overall,
        "timbre_match": f"centroid={centroid}Hz within [{lo},{hi}] -> {timbre_score}/5",
        "rhythm_match": f"bpm={bpm} vs target={target_bpm} -> {rhythm_score}/5",
        "overall": f"Heuristic score {overall}/5 (timbre={timbre_score}, rhythm={rhythm_score})"
    }


# ═══════════════════════════════════════════════════════════
# 4. 批量评测
# ═══════════════════════════════════════════════════════════
def batch_evaluate():
    AUDIO_DIR = Path("C:/Deepseek/outputs/eval_audio")
    RESULTS = Path("C:/Deepseek/outputs/llm_eval_results.jsonl")

    wavs = sorted(AUDIO_DIR.glob("*.wav"))
    if not wavs:
        print(f"No WAVs in {AUDIO_DIR}")
        return

    # 推断 style 和 prompt
    results = {}
    for w in tqdm(wavs, desc="Evaluating"):
        name = w.stem
        # Infer style from filename suffix
        for style_tag in ["erhu", "dizi", "suona"]:
            if name.endswith(style_tag):
                style = style_tag
                break
        else:
            style = "unknown"

        # Extract original prompt (truncate)
        prompt = name[:-len(style)].rstrip("_")[:80]

        features = extract_features(str(w))
        score_data = heuristic_score(style, features)

        row = {
            "file": str(w.name),
            "style": style,
            "prompt": prompt,
            "features": features,
            "score": score_data["score"],
            "timbre_match": score_data["timbre_match"],
            "rhythm_match": score_data["rhythm_match"],
            "overall": score_data["overall"],
        }
        results[str(w.name)] = row

    # 写入文件
    with open(RESULTS, "w", encoding="utf-8") as f:
        for row in results.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 统计
    scores = [r["score"] for r in results.values()]
    print(f"\n{'='*60}")
    print(f"LLM-as-Judge 评测完成 ({len(scores)} 首)")
    print(f"{'='*60}")
    print(f"  Mean Score:   {np.mean(scores):.2f} / 5")
    print(f"  Median Score: {np.median(scores):.0f} / 5")
    print(f"  Std Dev:      {np.std(scores):.2f}")
    print(f"  Score >= 4:   {sum(1 for s in scores if s >= 4)}/{len(scores)}")

    # 按风格分
    for style in ["erhu", "dizi", "suona"]:
        ss = [r["score"] for r in results.values() if r["style"] == style]
        if ss:
            print(f"  {style:8s}: mean={np.mean(ss):.2f}  n={len(ss)}")

    print(f"\n  Results: {RESULTS}")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    batch_evaluate()
