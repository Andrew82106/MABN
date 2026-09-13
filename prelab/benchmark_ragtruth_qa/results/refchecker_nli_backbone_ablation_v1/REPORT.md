# Retrieved-evidence NLI v1 开发结果

每条陈述先在三个来源内固定检索最多两句，再由冻结 NLI 评分。逻辑回归只在 fit 的 source-group 五折预测上定阈值；calibration 只报告。

| 方法 | fit窗口F1 | cal窗口F1 | fit整答F1 | cal整答F1 |
|---|---:|---:|---:|---:|
| raw_or_risk | 0.3311 | 0.3316 | 0.6919 | 0.7826 |
| claim_lr | 0.4230 | 0.4366 | 0.6964 | 0.7871 |

这属于我们的方法候选；正式 baseline 的结构和参数没有改。句级广播会牺牲边界精度，测试仍封存。
