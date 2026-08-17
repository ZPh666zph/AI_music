#!/usr/bin/env python3
"""
evaluate_metrics.py — 客观指标评估: Chord Accuracy + Rhythm Obedience + 更多
================================================================================
输入: gufeng_train_dataset.jsonl (含 audio_path + tokens + meta)
输出: 每首的评分 JSON + 汇总报告
"""

import sys, os, json, re, argparse
from pathlib import Path
from collections import defaultdict

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════
# 指标 1: 和弦准确率
# ═══════════════════════════════════════════════════════════

def extract_chroma(wav_path: str, sr: int = 16000) -> np.ndarray:
    """
    从 WAV 提取 Chroma 特征 (12 维, 每帧一个向量)
    使用 librosa 的 CQT (Constant-Q) chroma, 精度优于 STFT chroma
    """
    import librosa
    y, _sr = librosa.load(wav_path, sr=sr, mono=True)
    # CQT chroma: 对音乐信号的和声分析更准确
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
    return chroma  # shape: [12, T_frames]


def chroma_to_chord_sequence(chroma: np.ndarray, hop_length: int = 512, sr: int = 16000) -> list:
    """
    将 Chroma 特征序列转换为和弦序列 (简化版 Viterbi 风格)
    每 0.5 秒 (≈ 约 16 帧) 取一次和弦估计
    """
    pitch_names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    window_frames = int(0.5 * sr / hop_length)  # 0.5 秒窗口
    chords = []

    for t in range(0, chroma.shape[1], window_frames):
        chunk = chroma[:, t:t+window_frames]
        if chunk.shape[1] < 4:
            continue
        # 取窗口内 Chroma 平均值
        mean_chroma = chunk.mean(axis=1)  # [12]

        # 找能量最大的 3 个音高 (常见三和弦的根三五音)
        top3_idx = np.argsort(mean_chroma)[-3:]

        # 和弦模板匹配: Major = {0,4,7}, minor = {0,3,7}
        if len(top3_idx) >= 2:
            pitch_set = set(top3_idx)
            root_candidates = []
            for r in top3_idx:
                if (r + 4) % 12 in pitch_set and (r + 7) % 12 in pitch_set:
                    root_candidates.append((r, "maj"))
                if (r + 3) % 12 in pitch_set and (r + 7) % 12 in pitch_set:
                    root_candidates.append((r, "min"))

            if root_candidates:
                root, quality = root_candidates[0]
                # 用 Chroma 能量作为置信度
                confidence = mean_chroma[root] / (mean_chroma.sum() + 1e-8)
                chords.append({
                    "time": t * hop_length / sr,
                    "chord": f"{pitch_names[root]}:{quality}",
                    "confidence": round(float(confidence), 3),
                })

    return chords


def extract_target_chords(token_seq: str) -> list:
    """从 Token 序列提取目标和弦列表"""
    chords = []
    for m in re.finditer(r'\[CHORD:([^\]]+)\]', token_seq):
        chords.append(m.group(1))
    return chords


def chord_accuracy(audio_path: str, token_seq: str) -> dict:
    """
    计算和弦准确率:
    1. 从音频提取 Chroma → 估计和弦序列
    2. 从 Token 提取目标和弦
    3. 匹配率 = 匹配数 / 总数
    """
    try:
        chroma = extract_chroma(audio_path)
        predicted = chroma_to_chord_sequence(chroma)
        targets = extract_target_chords(token_seq)

        if not targets or not predicted:
            return {"accuracy": 0.0, "predicted_count": len(predicted),
                    "target_count": len(targets), "error": "empty"}

        # 时间对齐: 为每个 target chord 找最近的 predicted chord
        matches = 0
        total = len(targets)

        # 简化: 比较和弦名 (忽略 quality 差异, 只看根音)
        pred_roots = [c["chord"].split(":")[0] for c in predicted]
        target_roots = [c.split(":")[0] if ":" in c else c for c in targets]

        for i, tr in enumerate(target_roots):
            # 找时间窗口内的预测
            ratio = i / max(total, 1)
            pred_idx = int(ratio * len(pred_roots))
            # 取窗口内最近的预测
            window = pred_roots[max(0,pred_idx-1):min(len(pred_roots),pred_idx+2)]
            if tr in window:
                matches += 1

        accuracy = matches / max(total, 1)

        return {
            "accuracy": round(accuracy, 3),
            "predicted_count": len(predicted),
            "target_count": total,
            "matches": matches,
            "predicted_roots": pred_roots[:5],
            "target_roots": target_roots[:5],
        }

    except Exception as e:
        return {"accuracy": 0.0, "error": str(e)[:100]}


# ═══════════════════════════════════════════════════════════
# 指标 2: 节奏服从度
# ═══════════════════════════════════════════════════════════

def detect_bpm(wav_path: str, sr: int = 16000) -> dict:
    """
    检测音频的 BPM (使用 librosa 的 beat tracking)
    返回: detected_bpm, confidence, beat_times
    """
    import librosa
    y, _sr = librosa.load(wav_path, sr=sr, mono=True)

    # 1. 起始点检测 (onset) → 用于 beat tracking
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)

    # 2. 全局 BPM 估计
    bpm = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, hop_length=512)

    # 3. Beat tracking → 获取 beat times
    _, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr,
                                        hop_length=512, units='time')

    return {
        "detected_bpm": round(float(bpm[0]), 1),
        "beat_times": [round(float(b), 3) for b in beats[:20]],
        "num_beats": len(beats),
    }


def extract_target_bpm(token_seq: str) -> float:
    """从 Token 序列提取目标 BPM"""
    m = re.search(r'\[GLOBAL:TEMPO:(\d+)\]', token_seq)
    if m:
        return float(m.group(1))
    m = re.search(r'\[TEMPO:(\d+)\]', token_seq)
    if m:
        return float(m.group(1))
    return None


def rhythm_obedience(audio_path: str, token_seq: str) -> dict:
    """
    计算节奏服从度:
    1. 检测音频 BPM
    2. 与目标 BPM 比对
    3. 误差 = |detected - target| / target
       Obedience = max(0, 1 - error)
    """
    target_bpm = extract_target_bpm(token_seq)
    if target_bpm is None:
        return {"obedience": 0.0, "error": "no target BPM in tokens"}

    try:
        detected = detect_bpm(audio_path)
        detected_bpm = detected["detected_bpm"]

        # 相对误差
        error = abs(detected_bpm - target_bpm) / max(target_bpm, 1)
        obedience = max(0.0, 1.0 - error)

        return {
            "obedience": round(obedience, 3),
            "target_bpm": int(target_bpm),
            "detected_bpm": detected_bpm,
            "absolute_error": round(abs(detected_bpm - target_bpm), 1),
            "relative_error": round(error, 3),
            "num_beats": detected["num_beats"],
        }
    except Exception as e:
        return {"obedience": 0.0, "error": str(e)[:100]}


# ═══════════════════════════════════════════════════════════
# 指标 3: 音色频谱质心偏差 (补充)
# ═══════════════════════════════════════════════════════════

def spectral_centroid_deviation(audio_path: str, token_seq: str) -> dict:
    """
    检测频谱质心, 与 [COLOR:xxx] Token 预期值比较
    warm → 低质心 (~400Hz), bright → 高质心 (~5500Hz)
    """
    import librosa
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_centroid = float(np.mean(centroid))

    # 从 token 提取预期 color
    expected = 3000  # default neutral
    if "COLOR:warm" in token_seq or "COLOR:dark" in token_seq:
        expected = 400
    elif "COLOR:bright" in token_seq:
        expected = 5500

    deviation = abs(mean_centroid - expected) / max(expected, 1)

    return {
        "centroid_hz": round(mean_centroid, 1),
        "expected_hz": expected,
        "relative_deviation": round(deviation, 3),
    }


# ═══════════════════════════════════════════════════════════
# 指标 4: 动态 RMS 相关性 (补充)
# ═══════════════════════════════════════════════════════════

def rms_correlation(audio_path: str, token_seq: str) -> dict:
    """
    分段 RMS 与 Token 中 DYN 指令的 Pearson 相关系数
    DYN 映射: pp=0, p=1, mp=2, mf=3, f=4, ff=5
    """
    import librosa
    y, sr = librosa.load(audio_path, sr=16000, mono=True)

    # 提取 DYN 序列
    dyn_map = {"pp":0,"p":1,"mp":2,"mf":3,"f":4,"ff":5}
    dyn_targets = []
    for m in re.finditer(r'\[DYN:([^\]]+)\]', token_seq):
        val = m.group(1).split("|")[0]  # 处理 [DYN:pp|...] 格式
        if val in dyn_map:
            dyn_targets.append(dyn_map[val])

    if len(dyn_targets) < 2:
        return {"correlation": 0.0, "error": "too few DYN tokens"}

    # 分段 RMS
    seg_len = len(y) // len(dyn_targets)
    rms_vals = []
    for i in range(len(dyn_targets)):
        seg = y[i*seg_len:(i+1)*seg_len]
        if len(seg) > 0:
            rms_vals.append(float(np.sqrt(np.mean(seg**2))))

    if len(rms_vals) > 1:
        corr = float(np.corrcoef(rms_vals, dyn_targets)[0,1])
        return {"correlation": round(corr, 3), "segments": len(rms_vals)}
    return {"correlation": 0.0}


# ═══════════════════════════════════════════════════════════
# 批量评估主循环
# ═══════════════════════════════════════════════════════════

def evaluate_dataset(jsonl_path: str, output_path: str = None, max_samples: int = None):
    """遍历训练集 JSONL, 逐首计算全部指标"""
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        print(f"未找到: {jsonl_path}")
        return

    with open(jsonl_path, encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    if max_samples:
        samples = samples[:max_samples]

    print(f"评估 {len(samples)} 首音频...\n")

    all_results = []
    agg = defaultdict(list)

    pbar = tqdm(samples, desc="指标计算", unit="首")
    for s in pbar:
        audio = s["audio_path"]
        if not os.path.exists(audio):
            pbar.write(f"  ✗ 音频缺失: {s['track_name']}")
            continue

        tokens = s["tokens"]

        result = {
            "track_name": s["track_name"],
            "bpm_target": s["meta"].get("bpm"),
            "key": s["meta"].get("key"),
        }

        # 和弦准确率
        ca = chord_accuracy(audio, tokens)
        result["chord_accuracy"] = ca.get("accuracy", 0)

        # 节奏服从度
        ro = rhythm_obedience(audio, tokens)
        result["rhythm_obedience"] = ro.get("obedience", 0)

        # 频谱质心偏差 (可选, 较慢)
        # sc = spectral_centroid_deviation(audio, tokens)
        # result["spectral_dev"] = sc.get("relative_deviation", 1)

        # RMS 相关性 (可选, 较慢)
        # rc = rms_correlation(audio, tokens)
        # result["rms_correlation"] = rc.get("correlation", 0)

        all_results.append(result)

        # 聚合统计
        agg["chord_acc"].append(result["chord_accuracy"])
        agg["rhythm_ob"].append(result["rhythm_obedience"])

        pbar.set_postfix_str(
            f"CA:{result['chord_accuracy']:.2f} RO:{result['rhythm_obedience']:.2f}"
        )

    # ── 汇总 ──
    print(f"\n{'='*60}")
    print(f"评估完成 · {len(all_results)} 首")
    print(f"{'='*60}")
    print(f"  和弦准确率:    均值={np.mean(agg['chord_acc']):.3f}  "
          f"中位={np.median(agg['chord_acc']):.3f}  "
          f"σ={np.std(agg['chord_acc']):.3f}")
    print(f"  节奏服从度:    均值={np.mean(agg['rhythm_ob']):.3f}  "
          f"中位={np.median(agg['rhythm_ob']):.3f}  "
          f"σ={np.std(agg['rhythm_ob']):.3f}")

    # 保存
    if output_path:
        out_dir = os.path.dirname(output_path) or "."
        os.makedirs(out_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "summary": {
                    "total_samples": len(all_results),
                    "chord_accuracy_mean": round(float(np.mean(agg['chord_acc'])), 3),
                    "chord_accuracy_std": round(float(np.std(agg['chord_acc'])), 3),
                    "rhythm_obedience_mean": round(float(np.mean(agg['rhythm_ob'])), 3),
                    "rhythm_obedience_std": round(float(np.std(agg['rhythm_ob'])), 3),
                },
                "details": all_results,
            }, f, ensure_ascii=False, indent=2)
        print(f"\n  结果已保存: {output_path}")


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Music ControlNet 客观指标评估")
    parser.add_argument("--data", default="C:/Deepseek/data/gufeng_train_dataset.jsonl")
    parser.add_argument("--output", default="C:/Deepseek/outputs/eval_metrics.json")
    parser.add_argument("--max", type=int, default=None, help="最多评估 N 首")
    parser.add_argument("--single", type=str, help="评估单首 WAV (调试用)")
    parser.add_argument("--tokens", type=str, default="", help="单首对应的 Token 序列")
    args = parser.parse_args()

    if args.single:
        print(f"单首评估: {args.single}")
        ca = chord_accuracy(args.single, args.tokens)
        ro = rhythm_obedience(args.single, args.tokens)
        print(f"  和弦准确率: {json.dumps(ca, ensure_ascii=False, indent=2)}")
        print(f"  节奏服从度: {json.dumps(ro, ensure_ascii=False, indent=2)}")
    else:
        evaluate_dataset(args.data, args.output, args.max)
