# 同预算三轮检测器对照

| 新信号 | 既有组合 | 窗口F1 | 整答F1 |
|---|---|---:|---:|
| large | lookback__old_lr | 0.679293 | 0.873096 |
| large | lookback__old_tree | 0.677619 | 0.879227 |
| large | harp_claim__old_lr | 0.689809 | 0.883495 |
| large | harp_claim__old_tree | 0.685120 | 0.890995 |
| large | semantic_claim__old_lr | 0.690159 | 0.878788 |
| large | semantic_claim__old_tree | 0.690281 | 0.891089 |
| nli | lookback__old_lr | 0.665229 | 0.872549 |
| nli | lookback__old_tree | 0.670381 | 0.874419 |
| nli | harp_claim__old_lr | 0.685063 | 0.873096 |
| nli | harp_claim__old_tree | 0.683599 | 0.876289 |
| nli | semantic_claim__old_lr | 0.680312 | 0.864078 |
| nli | semantic_claim__old_tree | 0.682716 | 0.870466 |
| fava | lookback__old_lr | 0.672805 | 0.871795 |
| fava | lookback__old_tree | 0.671722 | 0.871795 |
| fava | harp_claim__old_lr | 0.687696 | 0.870466 |
| fava | harp_claim__old_tree | 0.687364 | 0.865979 |
| fava | semantic_claim__old_lr | 0.683675 | 0.858586 |
| fava | semantic_claim__old_tree | 0.680976 | 0.859903 |

每行两项来自同一候选，保留原4BPE窗口和整答max。只比较既有选型，不重新选阈值或拟合。

在本表全部已选Lookback/HARP对照前，两项点估计均未落后的语义候选：semantic_claim__old_tree__large_weight0.4，0.690281/0.891089。

这是反复使用的159答校准集，不能据此宣称稳定优于基线、统计不劣或SOTA。独立测试仍封存。
