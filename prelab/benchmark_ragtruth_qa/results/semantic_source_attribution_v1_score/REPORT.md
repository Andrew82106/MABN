# Semantic source attribution v1 score

模型保留完整 32×32 source-attribution 坐标，并用注意力选中的精确来源句做 NLI；clean Aligned Evidence 基座不含历史 calibration 优化分数。两个部件都按 source-connected group 五折交叉预测。

| 口径 | 窗口F1 | 整答F1 |
|---|---:|---:|
| strict fit阈值 | 0.577538 | 0.810000 |
| 统一cal F1-opt诊断 | 0.582855 | 0.834951 |

语义门在 calibration 新增 TP 459、FP 590。

未改正式 baseline，未打开 official test。
