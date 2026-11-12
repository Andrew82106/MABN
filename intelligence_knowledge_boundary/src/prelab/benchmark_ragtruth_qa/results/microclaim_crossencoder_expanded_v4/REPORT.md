# Expanded-v4 microclaim cross-encoder

唯一训练候选直接微调原始三分类 ModernBERT NLI；fit为source-connected五折OOF，calibration由full-fit模型预测。

| 层级 | fit OOF F1 | calibration严格F1 |
|---|---:|---:|
| 4-BPE窗口 | 0.551070 | 0.579847 |
| 整答 | 0.666667 | 0.796020 |

正式baseline未修改；official test未打开。calibration F1-opt仅作共同开发诊断。

已知限制：fit中49条无关回答共享固定拒答文本且均为负例，但其完整question+evidence+hypothesis输入哈希互异、跨折零重叠；该模板仍可能带来分类捷径。
