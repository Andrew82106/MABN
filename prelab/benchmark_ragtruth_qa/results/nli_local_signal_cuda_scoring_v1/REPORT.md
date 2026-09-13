# 冻结NLI局部信号：原生QA开发结果

所有阈值和两个固定LR均只由fit决定；calibration只报告一次。官方test未读取。

| 方法 | cal窗口 AUROC | AP | F1 | cal整答 AUROC | AP | F1 |
|---|---:|---:|---:|---:|---:|---:|
| raw_max_contradiction | 0.652 | 0.270 | 0.303 | 0.570 | 0.709 | 0.775 |
| nli_only_fixed_lr | 0.839 | 0.468 | 0.522 | 0.763 | 0.811 | 0.811 |
| current_plus_nli_fixed_lr | 0.891 | 0.640 | 0.685 | 0.896 | 0.930 | 0.822 |

三项均为我们的方法候选，不属于基线：raw是固定的跨片段/四资料视图最大矛盾概率；NLI LR只读12个冻结概率；融合LR再加入既有current窗口分数。
融合继承current候选曾用本calibration选定的历史，只能作为开发诊断，不能视为独立最终成绩。
