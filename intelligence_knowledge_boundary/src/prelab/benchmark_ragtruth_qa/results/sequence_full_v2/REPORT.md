两项新TCN均为1089维输入、94529参数、RF7、同seed/权重/30轮；下列是校准选轮与选阈值后的开发成绩。

| 方法 | 轮次 | fit窗F1 | cal窗F1 | fit整答F1 | cal整答F1 |
|---|---:|---:|---:|---:|---:|
| token_mlp | 6 | 0.567 | 0.563 | 0.738 | 0.838 |
| token_tcn | 6 | 0.602 | 0.593 | 0.741 | 0.827 |
| full_lb_pca64_tcn | 3 | 0.633 | 0.607 | 0.762 | 0.854 |
| full_lb_harp64_tcn | 3 | 0.634 | 0.610 | 0.776 | 0.854 |

冲突类使用同一个全局窗口阈值；原span被已报警4BPE窗覆盖任一风险token算至少命中，覆盖全部风险token算完整命中。

| 方法 | 冲突类 | 阳性窗召回 | 原span至少命中 | 原span完整命中 |
|---|---|---:|---:|---:|
| token_mlp | Evident Conflict | 178/997 | 20/43 | 6/43 |
| token_mlp | Subtle Conflict | 51/109 | 5/5 | 1/5 |
| token_tcn | Evident Conflict | 202/997 | 18/43 | 6/43 |
| token_tcn | Subtle Conflict | 43/109 | 4/5 | 0/5 |
| full_lb_pca64_tcn | Evident Conflict | 162/997 | 14/43 | 2/43 |
| full_lb_pca64_tcn | Subtle Conflict | 49/109 | 4/5 | 1/5 |
| full_lb_harp64_tcn | Evident Conflict | 187/997 | 16/43 | 5/43 |
| full_lb_harp64_tcn | Subtle Conflict | 67/109 | 5/5 | 1/5 |

校准集Subtle Conflict只有5个原span，细分类结果样本很少。原标签与4BPE几何未改，未按错误类型另调阈值。
A/B输入维数与参数数相同；与旧193维TCN比较时投影参数也增加，不能把差异全部归因为保留信息。PCA64/HARP64都不是完整隐藏状态。
旧MLP和193维TCN原结果均保留，未重训；测试未打开。全部60轮checkpoint与预测已保存。
现有LR候选中校准窗口最高：layerband_slots_C0.001，F1=0.563。全部候选及常数基线见BASELINE_COMPARISON.json。
现有LR候选中校准整答最高：harp64_lookback_nll_C0.001，F1=0.869。全部候选及常数基线见BASELINE_COMPARISON.json。
