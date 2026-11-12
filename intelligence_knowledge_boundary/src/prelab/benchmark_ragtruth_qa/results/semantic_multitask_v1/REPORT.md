仅新增两个辅助类型头：训练区分无依据/与资料冲突；所有推理和评测仍只用原总风险头。额外MiniCheck与原1090输入未变。

| 模型 | 选中轮次 | fit窗F1 | cal窗P | cal窗R | cal窗F1 | cal整答F1 |
|---|---:|---:|---:|---:|---:|---:|
| minicheck_hidden64_risk_tcn_w32 | 3 | 0.667 | 0.619 | 0.658 | 0.638 | 0.860 |
| semantic_tcn_aux_types_w32 | 3 | 0.663 | 0.662 | 0.621 | 0.641 | 0.857 |

| 模型 | 类型 | 阳性窗命中 | 原span至少命中 | 原span完整覆盖 |
|---|---|---:|---:|---:|
| minicheck_hidden64_risk_tcn_w32 | Evident Conflict | 273/997 | 18/43 | 9/43 |
| minicheck_hidden64_risk_tcn_w32 | Subtle Conflict | 78/109 | 5/5 | 2/5 |
| minicheck_hidden64_risk_tcn_w32 | Evident Baseless Info | 3003/4086 | 90/109 | 55/109 |
| minicheck_hidden64_risk_tcn_w32 | Subtle Baseless Info | 604/814 | 36/43 | 25/43 |
| semantic_tcn_aux_types_w32 | Evident Baseless Info | 2850/4086 | 87/109 | 50/109 |
| semantic_tcn_aux_types_w32 | Subtle Baseless Info | 560/814 | 36/43 | 25/43 |
| semantic_tcn_aux_types_w32 | Evident Conflict | 248/997 | 17/43 | 9/43 |
| semantic_tcn_aux_types_w32 | Subtle Conflict | 76/109 | 5/5 | 2/5 |

| 模型 | 所选轮risk BCE | 最后轮risk BCE | 最后轮cal窗F1 |
|---|---:|---:|---:|
| minicheck_hidden64_risk_tcn_w32 | 0.3148 | 0.0346 | 0.571 |
| semantic_tcn_aux_types_w32 | 0.3180 | 0.0349 | 0.604 |

辅助头所选轮fit BCE（无依据、冲突）=[0.24775601695815583, 0.28099270198999443]；最后轮=[0.04549439997297639, 0.03312179465900986]。

主干/risk初始化与原模型逐参数及训练模式dropout输出一致，辅助RNG隔离；同30次shuffle、同优化器、同风险权重，辅助系数0.25固定，未按类型选择checkpoint。所有旧基线保留。
原人工risk标签与4原始BPE窗口完全不变。两个辅助标签可重叠，OR与risk逐词元一致；类权重只用fit统计，非lexical loss为0。
这是单seed、cal选轮/选阈值后的开发结果，test未打开。辅助监督也改变优化和正则效应，不能单凭这次结果证明唯一错误原因。
