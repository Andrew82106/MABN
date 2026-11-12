# Microclaim full-evidence cross-encoder v1

主候选直接微调原始三分类 ModernBERT NLI；输入包含问题、三篇完整资料和单个原子微主张。fit 为 source-connected 五折 OOF，cal 由独立的 full-fit 模型预测。

| 方法 | fit OOF窗口F1 | cal严格窗口F1 | fit OOF整答F1 | cal严格整答F1 |
|---|---:|---:|---:|---:|
| fp-aware full-evidence | 0.576361 | 0.596282 | 0.760902 | 0.776471 |
| frozen full-evidence NLI | 0.445632 | 0.463939 | 0.695548 | 0.765957 |

统一4-BPE窗口和fit阈值未改；cal-F1Opt仅为共同开发诊断。正式baseline未改，official test未打开。
