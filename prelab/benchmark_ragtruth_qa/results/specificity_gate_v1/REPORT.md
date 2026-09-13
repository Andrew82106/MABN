# Specificity gate v1

候选和全部门槛只由 fit 决定；下表 strict cal 是冻结后一次报告。cal-F1Opt 只作非独立诊断。

| 方法 | fit窗口F1 | fit整答F1 | strict cal窗口F1 | strict cal整答F1 | cal-F1Opt窗口 | cal-F1Opt整答 |
|---|---:|---:|---:|---:|---:|---:|
| gate_full_hgb | 0.819761 | 0.902208 | 0.687696 | 0.806630 | 0.690454 | 0.882353 |

strict cal 增量：TP 71，FP 105，精度 0.403409。
NLI独有候选中保留 TP 4/366，同时保留 FP 7/1289。

与历史 incumbent 0.690281/0.891089 比较：不接受，停止该候选。

限制：incumbent 上游本身曾使用 calibration 选择；本实验没有改它。新门控及本轮阈值只使用 fit，正式 baseline 与 official test 均未改动或打开。
