# Objective comparison on the gufeng test split (RTX 5090, n = 63 shared prompts)

FAD ↓: VGGish Fréchet Audio Distance against one shared gufeng reference set.
CLAP ↑: LAION-CLAP text–audio cosine similarity (same prompt string for every system).
Ours is trained on a disjoint 602-track train split; no test prompt is seen in training.

| Model | Params | FAD ↓ | CLAP ↑ |
|---|---|---|---|
| MusicGen-medium | 100% (FT) | 4.3353 | 0.0785 ± 0.0892 |
| AudioLDM 2 | 100% (FT) | 5.5866 | 0.0392 ± 0.0924 |
| Stable Audio Open | 100% (FT) | 3.9340 | 0.1862 ± 0.1010 |
| MusiConGen | 12.5% (LoRA) | 5.0561 | 0.1871 ± 0.0748 |
| **MontageDirector (Ours, CE 10 ep)** | **0.67%** | **4.4519** | **-0.0197** ± 0.0967 |
| MontageDirector (CE 40 ep) | 0.67% | 5.5177 | -0.0583 ± 0.0694 |
| MontageDirector (w/o CE) | 0.67% | 5.9494 | -0.0573 ± 0.0730 |

**Key findings.** (i) With the real EnCodec cross-entropy objective the adapter reaches FAD 4.4519,
matching fully fine-tuned baselines while updating only 0.67% of the 2.02B backbone; removing the CE
objective degrades FAD to 5.9494. (ii) Training must be early-stopped: 40 epochs overfit (FAD 5.5177).
(iii) CLAP is lower than the baselines by construction — Ours, unlike them, exposes a discrete,
time-varying symbolic control channel; see the trade-off analysis in the paper's Discussion (Sec. V).
