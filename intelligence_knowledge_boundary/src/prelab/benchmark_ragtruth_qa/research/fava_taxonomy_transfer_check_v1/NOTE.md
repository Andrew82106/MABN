# FAVA辅助数据与当前冲突漏检的关系

核查时间：2026-09-12。本记录不修改标签、候选名单或训练协议。

[FAVA作者项目页](https://fine-grained-hallucination.github.io/)将实体和关系错误列为矛盾错误的细类，将整句矛盾定义为被给定参考资料反驳。因此这里的contradictory不是仅指回答自相矛盾。发表版本为[COLM 2024论文](https://openreview.net/pdf?id=dJMTn3QOWO)。

当前已经冻结的FAVA v2包含7482答、19729个合成范围：entity5743、relation5317、contradictory4560、invented4109。前三类合计15620个范围。这些是标注范围数，不是独立事件数或人工核实后的事实数。

原公共QA的3680个训练回答中，人工冲突类原范围为301个；对应原BPE中含冲突的lexical词元5320，占全部47398个风险词元约11.2%。两类同时标记的36个词元保留，不能将“无依据/冲突”硬转成互斥类别。

当前选定候选在原校准集检出显性冲突195/997个风险窗口，而显性无依据为3034/4086。这说明值得检查对两类错误的不同学习效果；数量差异本身不证明失败原因。

FAVA辅助训练具有较多实体、关系及整句矛盾示例，是对这一缺口的合理检验。但标签体系、合成方式、上下文和问题格式不同，不能仅据类别名把所有FAVA标签直接改成RAGTruth类型。现有已冻结训练仍按原方案只用二值风险，非标注位置仅作为有噪声的辅助负标签。只有原人工QA评测才能检验迁移是否有效。

本地计数来源：`auxiliary_fava_v2/manifest.json`、`results/fit_type_partition_inventory_v1/summary.json`、`results/selected_convex_error_diagnosis_v1/summary.json`，相对于benchmark根目录。
