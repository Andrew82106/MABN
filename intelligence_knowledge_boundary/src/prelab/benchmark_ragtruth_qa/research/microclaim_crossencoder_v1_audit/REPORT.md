# Microclaim full-evidence cross-encoder v1 audit

严格 calibration：窗口 F1 0.596282，整答 F1 0.776471。
相对 incumbent 的独有误报：旧 atomic NLI 1524，新模型 974，减少 550。

| 标签类型 | 新模型召回 | 旧 atomic NLI | incumbent |
|---|---:|---:|---:|
| Evident Baseless Info | 0.6148 | 0.7643 | 0.7425 |
| Evident Conflict | 0.2347 | 0.2578 | 0.1956 |
| Subtle Baseless Info | 0.6585 | 0.7371 | 0.6732 |
| Subtle Conflict | 0.3303 | 0.6789 | 0.7523 |

这些都是反复使用 calibration 后的开发诊断；official test 未打开，正式 baseline 未修改。
