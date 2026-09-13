本轮仅使用R16实际train的固定事件五折，属于反复开发的数据；不是封存的新测试。

| 方法 | 4原始BPE窗口F1 | 整答F1 |
|---|---:|---:|
| base | 0.6404 | 0.6897 |
| slots_base | 0.6554 | 0.6912 |
| r19_all | 0.6431 | 0.7312 |
| base_harp_delta | 0.6651 | 0.6821 |
| lookback_tuned | 0.6167 | 0.6993 |
| redeep_tuned | 0.3558 | 0.4857 |
| verifier_learned_broadcast | 0.4364 | 0.8350 |
| verifier_learned_slots | 0.5751 | 0.8232 |
| slots_base_smooth | 0.6791 | 0.6931 |
| local_direct | 0.2598 | 0.4422 |
| local_probe | 0.5451 | 0.6585 |
| local_mean_fusion | 0.6851 | 0.7292 |
| local_slots_fusion | 0.6729 | 0.7222 |
| local_slots_fusion_smooth | 0.6941 | 0.7226 |
| slots_base_smooth_global | 0.6989 | 0.7625 |
| local_mean_fusion_global | 0.6802 | 0.7589 |
| local_slots_fusion_smooth_global | 0.7090 | 0.8000 |

分母保持原样：598个可评整答、481个定位回答、9526个可评窗口。安全拒答为整答负类，其候选窗口仍参加整答max，但不进入定位分母。标签为助手复核裁决，不是独立人工金标。

固定三个局部模型，各用相同六组alpha，将局部风险logit与整答核查风险logit相加；只在各折cal选择alpha及阈值，全部折冻结后才评分。没有新增模型拟合或标签改动。较强组合窗口0.7090/整答0.8000；同样优化过的旧slots基线为0.6989/0.7625，因此增量仍较小。
