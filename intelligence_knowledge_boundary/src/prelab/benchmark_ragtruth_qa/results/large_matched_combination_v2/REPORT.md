# 各基线获得相同large信号

| 对象 | 组合 | 定位F1 | 整答F1 |
|---|---|---:|---:|
| lookback | large_lr | 0.672237 | 0.860000 |
| lookback | large_tree | 0.670858 | 0.858537 |
| harp_claim | large_lr | 0.673836 | 0.857143 |
| harp_claim | large_tree | 0.670858 | 0.858537 |
| semantic_claim | large_lr | 0.674838 | 0.859813 |
| semantic_claim | large_tree | 0.670858 | 0.858537 |

同一已固定large轮次，三对象各三档LR和一次固定单调树；原控制未重训。
这里仍是反复开发的校准集、额外离线语义核查融合；原4BPE窗口/标签/完整回答max未改，官方测试未打开。
须逐候选对照原强基线和large单独成绩，不能拼接两个最优指标。
