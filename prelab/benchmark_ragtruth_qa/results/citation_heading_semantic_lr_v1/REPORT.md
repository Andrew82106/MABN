# 标题继承正文的语义来源：匹配LR对照

两模式共享完全相同的1161对新推理、原两分数/8词面/3scope及任意来源支持度；仅15列候选新增继承来源支持差。
原634fit/159cal、全部793答/210364窗口、权重与两级阈值规则保持。全18候选保留；原13列scope控制只保存为历史参考，不重训。

| peer | 模式 | C | 窗口F1 | 整答F1 |
|---|---|---:|---:|---:|
| lookback | any_source_control | 0.001 | 0.647749 | 0.867580 |
| lookback | inherited_source_gap | 0.001 | 0.647557 | 0.867580 |
| harp_claim | any_source_control | 0.01 | 0.679609 | 0.868687 |
| harp_claim | inherited_source_gap | 0.01 | 0.679715 | 0.868687 |
| semantic_claim | any_source_control | 0.001 | 0.660697 | 0.860000 |
| semantic_claim | inherited_source_gap | 0.001 | 0.660697 | 0.860000 |

该输入只补冻结标题继承正文的来源支持，不解决步骤编号、时间顺序或句尾括号引用。
额外语义核查模型与生成白盒融合，上游fit分数非交叉拟合。校准反复用于开发，不是独立测试或SOTA。
