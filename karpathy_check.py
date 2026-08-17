"""Karpathy 军规: 客观指标交叉验证"""
import json, os, numpy as np, soundfile as sf

RESULTS = "C:/Deepseek/outputs/results.json"

with open(RESULTS, encoding="utf-8") as f:
    data = json.load(f)

print("=" * 60)
print("KARPATHY 军规 — 完整性检查")
print("=" * 60)

# 1. 文件完整性
missing = 0
for r in data["results"]:
    path = r["output"].replace("\\", "/")
    if not os.path.exists(path):
        print(f"  MISS: {r['id']}")
        missing += 1
    else:
        info = sf.info(path)
        if info.duration < 4.0:
            print(f"  SHORT: {r['id']} = {info.duration:.1f}s")
            missing += 1
if missing == 0:
    print("1. 全部 14 文件存在且长度正常")
else:
    print(f"1. {missing} 个文件异常！")

# 2. 客观响度 (dynamics)
print("\n--- 动态维度 RMS 能量 ---")
for r in data["results"]:
    if r["dimension"] in ("dynamics", "baseline"):
        path = r["output"].replace("\\", "/")
        audio, sr = sf.read(path)
        rms = float(np.sqrt(np.mean(audio**2)))
        r["rms"] = rms
        print(f"  {r['id']:20s} RMS={rms:.5f}  ({r['description']})")

# 3. 频谱质心 (timbre)
print("\n--- 音色维度频谱质心 ---")
for r in data["results"]:
    if r["dimension"] == "timbre":
        path = r["output"].replace("\\", "/")
        audio, sr = sf.read(path)
        spec = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1 / sr)
        centroid = float(np.sum(freqs * spec) / np.sum(spec)) if np.sum(spec) > 0 else 0
        r["spectral_centroid"] = centroid
        print(f"  {r['id']:20s} centroid={centroid:.0f}Hz  ({r['description']})")

# 4. 音符密度 (texture: 过零率作为密度代理)
print("\n--- 织体维度过零率 (密度代理) ---")
for r in data["results"]:
    if r["dimension"] == "texture":
        path = r["output"].replace("\\", "/")
        audio, sr = sf.read(path)
        zcr = float(np.sum(np.abs(np.diff(np.signbit(audio)))) / len(audio))
        r["zero_crossing_rate"] = zcr
        print(f"  {r['id']:20s} ZCR={zcr:.4f}  ({r['description']})")

# 5. 奏法分析 (articulation: 音符起始检测)
print("\n--- 奏法维度音符起始密度 ---")
for r in data["results"]:
    if r["dimension"] == "articulation":
        path = r["output"].replace("\\", "/")
        audio, sr = sf.read(path)
        hop = 512
        rms_frames = np.array([np.sqrt(np.mean(audio[i:i+hop]**2)) 
                               for i in range(0, len(audio)-hop, hop)])
        onsets = np.sum(np.diff(rms_frames > 0.01) > 0)
        r["onset_count"] = int(onsets)
        print(f"  {r['id']:20s} onsets={onsets:3d}  ({r['description']})")

# 保存
with open(RESULTS, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"\n2. 客观指标已写入 {RESULTS}")
print("\n=== KARPATHY CHECK COMPLETE ===")
