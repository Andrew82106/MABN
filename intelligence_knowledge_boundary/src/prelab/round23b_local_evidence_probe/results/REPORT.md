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

分母保持原样：598个可评整答、481个定位回答、9526个可评窗口。安全拒答为整答负类，其候选窗口仍参加整答max，但不进入定位分母。标签为助手复核裁决，不是独立人工金标。

12222个局部窗口各做一次完整、无cache、无padding的同Qwen离线核查；45次LR/5次fit-only PCA。较强结果是局部核查+slots+平滑，0.6941/0.7226。它相对旧平滑slots的窗口差值95%区间为[-0.00988,0.03910]，尚无稳健优势证据。约48分钟特征提取的额外开销须单独报告。原batch/cache版的数值失败保留，没有放宽门槛。
