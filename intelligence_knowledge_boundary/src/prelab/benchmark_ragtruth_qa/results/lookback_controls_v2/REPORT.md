五组 Lookback 定义对照均使用同一人工 QA 开发集。以下是用于选 C 和阈值的校准成绩，不能当最终测试成绩。

| 定义 | C | cal 窗口 F1 | cal 整答 F1 |
|---|---:|---:|---:|
| lb_source_pre_header | 0.001 | 0.549 | 0.860 |
| lb_prefix_pre_header | 0.001 | 0.562 | 0.845 |
| lb_source_post_header | 0.001 | 0.556 | 0.853 |
| lb_prefix_post_header | 0.001 | 0.566 | 0.852 |
| lb_source_post_legacy | 0.001 | 0.558 | 0.861 |

四格各3次新拟合，共12次；旧锚点3个模型直接重放，全部窗口矩阵、分数、阈值与原 lookback_mean 精确一致。
prefix_pre_header 的范围和时刻较接近官方代码，但 chat 头、量化和原始 trace 条件仍不同；不称完全复现原版实验。
fit634/cal159，168123/42241 可评窗口；未使用 NLL/PCA/hidden，未打开 test，未重拟合 fit+cal。
