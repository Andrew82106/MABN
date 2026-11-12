# Round8：逐词元定位

本轮复用Round7原回答和冻结检测器，检验能否定位具体无依据/矛盾片段。已完成9个原生词元预测器及14个整项广播对照；原阈值、验证集词元阈值两套结果均保存。没有改复查策略或重训分类器。

- [结论与下一步](results/REPORT.md)
- [全部指标表](results/TABLES.md)
- [七个预先选定案例](results/CASE_REVIEW.md)
- [独立计数核验](results/COUNT_AUDIT.json)
- [计划](PLAN.md)、[协议](protocol.json)、[片段标注规则](ANNOTATION_GUIDE.md)

`data/spans_*.jsonl`是冻结片段标注；`data/decisions/`保留初标和裁决，`data/reviews/`保留独立助手复核。标注不是研究者人工金标准。`data/token_features/`保留原生成IDs对应的词元特征，`results/token_scores_*.jsonl`保留逐词元分数。

`src/evaluate8.py`分验证集阈值冻结和测试两个阶段；已有冻结结果时拒绝覆盖。`src/report8.py`仅从冻结JSON重建指标表，不计算新分数。独立核验记录保存在COUNT_AUDIT.json。

本轮主测试含40题组/80回答，外部50题组/100回答；另有40题组/80回答供阈值选择。结果属于已使用题目的定位诊断，不能当作新的独立排行榜实验。
