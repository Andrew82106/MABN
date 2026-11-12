固定1090维、宽32序列融合结果。MiniCheck额外核查模型读取整句和资料，此方法是离线融合，全部为cal选轮/选阈值后的开发成绩。

| 方法 | fit窗F1 | cal窗P | cal窗R | cal窗F1 | cal整答F1 |
|---|---:|---:|---:|---:|---:|
| minicheck_hidden64_risk_tcn_w32 | 0.667 | 0.619 | 0.658 | 0.638 | 0.860 |
| minicheck_hidden64_C0.001 | 0.608 | 0.558 | 0.684 | 0.615 | 0.850 |
| minicheck_hidden64_lookback_nll_C0.001 | 0.685 | 0.603 | 0.616 | 0.609 | 0.849 |
| minicheck_hidden64_lookback_nll_risk_C0.001 | 0.687 | 0.599 | 0.625 | 0.612 | 0.840 |
| minicheck_calibrated | 0.520 | 0.484 | 0.671 | 0.562 | 0.810 |
| lookback_minicheck_fusion | 0.674 | 0.580 | 0.631 | 0.604 | 0.852 |
| full_lb_pca64_tcn | 0.633 | 0.613 | 0.600 | 0.607 | 0.854 |
| full_lb_harp64_tcn | 0.634 | 0.589 | 0.633 | 0.610 | 0.854 |

选择第3轮；共固定30轮。该轮fit加权BCE=0.3148，第30轮=0.0346；第30轮cal窗口F1=0.571。

| 类型 | 阳性窗命中 | 原span至少命中 | 原span完整覆盖 |
|---|---:|---:|---:|
| Evident Conflict | 273/997 | 18/43 | 9/43 |
| Subtle Conflict | 78/109 | 5/5 | 2/5 |
| Evident Baseless Info | 3003/4086 | 90/109 | 55/109 |
| Subtle Baseless Info | 604/814 | 36/43 | 25/43 |

原始标签、4原始BPE窗口及全部评测分母未变。PCA未重拟合，模型与scaler仅用fit，单seed，无test访问；所有旧基线和9LR保存在BASELINE_COMPARISON.json。
窗口大量重叠，不能当作独立数据；全量统计提升仍须在未打开的测试集确认。Subtle Conflict仅5个cal span。
