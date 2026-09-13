# Direct-window TCN 独立结果审计

输入、旧fit-only scaler/PCA、原LR权重及30轮覆盖均已核对；选中epoch由保存状态做CPU纯前向回放。未训练、调用GPU或读取test。

| 方法 | C / epoch | cal窗口F1 | cal整答F1 |
|---|---:|---:|---:|
| same_input_old_LR | 0.001 | 0.554486 | 0.867580 |
| old_token_TCN | 6 | 0.593235 | 0.827004 |
| direct_window_TCN | 4 | 0.601182 | 0.859813 |

这些校准成绩用于选择参数/epoch和阈值，存在选择乐观；不能当独立最终测试成绩。
与旧tokenTCN相比同时改变输入宽度、网络宽度、窗口覆盖和权重，不能把差异单独归因于直接窗口监督。
