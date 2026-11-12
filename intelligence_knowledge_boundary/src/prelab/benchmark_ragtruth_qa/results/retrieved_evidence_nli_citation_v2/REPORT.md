# Citation-conditioned retrieved-evidence NLI v2

v2 只改读出：区分被引用与未引用来源；单一 C、source-group OOF、fit 阈值。calibration 只报告。

| 方法 | fit窗口F1 | cal窗口F1 | fit整答F1 | cal整答F1 |
|---|---:|---:|---:|---:|
| citation_lr | 0.3996 | 0.4468 | 0.7143 | 0.7983 |
| fixed_whitebox_fusion | 0.7748 | 0.6641 | 0.8675 | 0.8254 |

固定融合权重在结果前冻结；其白盒上游有旧 calibration 选型和 fit 非完整 OOF 历史，只能作为开发诊断。正式 baseline 未修改，测试仍封存。
