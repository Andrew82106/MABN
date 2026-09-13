# 冻结结果中的三个解释案例

以下是完成正式评测后，对原资料、原回答及冻结预测的只读复核。例子用于说明错误类型，不代表其发生频率，不用于修改标签、阈值或选择方法。得分未校准为真实幻觉概率，不跨方法比较数值大小。

本轮固定报警阈值：线性探针 `0.620733086`，Lookback Lens `0.746901226`，自评风险 `0.050000001`，直接核查 `0.119206630`；均为得分大于等于阈值时报警。

## 1. 保留主体也会被线性探针误报

`hotpot_5ae52beb5542990ba0bbb1e1__partial__2`

问题问 Trollius 属于哪个植物科。可见资料 `Trollius` 原始第 0 句明确写出其属于 Ranunculaceae。此次被替换的是另一个主体 Kunzea 的科名资料；Trollius 的这条证据仍完整保留。

实际回答：

> Trollius belongs to the plant family Ranunculaceae.

冻结标签是 `asserted / supported / correct`，属于正常回答。线性探针得分 `0.837479`，错误报警；Lookback `0.137435`、自评 `0`、直接核查 `0.000006151` 均未报警。这一项没有进入四种方法的前 20% 复查名单。

同题完整资料条件下，这一项回答文字完全相同，线性探针得分为 `0.002440`，未报警。可见其他资料或回答前文的变化会伴随正常项得分变化；这个配对不能单独证明具体内部机制，也不能把整份回答有问题等同于每一项都有问题。

## 2. 从资料抄到数字，仍可能把主体张冠李戴

`ragognize_test_2337__partial__1`

问题问 Jetour Freedom 在中国开始销售的年月。当前唯一可见片段标题为 `Leapmotor B10`，其中写的是该车型 2025 年 3 月预售、同年 4 月开始交付，没有 Jetour Freedom 的销售时间。

实际回答：

> Sales of the Jetour Freedom commenced in China in April 2025.

冻结标签是 `asserted / unsupported / incorrect`。回答把另一车型的交付月份移接给了被问车型。完整参考资料 `Jetour Freedom` 第 4 句支持的是 **2025 年 2 月**；此参考仅用于复核真值，不是 partial 条件下模型可见的证据。

线性探针 `0.001451`、Lookback `0.095820` 均漏报；自评风险 `1.000000`、直接核查 `0.999344` 均报警。直接核查将它选入前 20% 复查名单，自评虽报警却未选入；自评出现相同分数时，原评测按预定 `item_id` 顺序打破并列。

这说明“文字或数字来自输入资料”不足以保证事实归属正确。此例与注意力类信号漏检主体错配相容，但不是对注意力机制的因果证明。

## 3. 正确拒答占据复查名额，但不计入事实断言 F1

`ragognize_test_0042__partial__1`

问题问 First Bus London 初始车队规模。可见片段是 `Arrow-class oil tanker` 的油轮尺寸和数量，以及 `Manston arrivals and processing centre` 的人员容量，均不提供目标公交车队规模。

实际回答：

> The size of First Bus London's initial fleet is not provided in the given search results.

冻结标签是 `abstained / not_applicable`，风险标签为空；这是合理拒答，并不是“正常事实项 risk=0”。线性探针得分 `0.9999999976`、Lookback `0.992608`、自评风险 `1.000000`，均报警并进入各自前 20% 名单；直接核查 `0.018042`，未报警，也未进入名单。

此项不进入二分类 F1 的 TP/FP/FN/TN，但会消耗实际复查预算。不能把它在 F1 中改算成一个 FP，也不能计算前 20% 时用事后拒答标签把它排除。

## F1 与前 20% 结果为何不同

原指标中，F1 只针对有明确二分类标签的事实断言：主集 219 项、外部集 79 项。前 20% 则从全部预定回答项排序：主集 240 项选 48 项，外部集 100 项选 20 项，包含拒答和未决项。

- 主集线性探针 F1 为 `0.846`，前 48 项为 **30 项风险、17 项拒答、1 项未决**；Lookback F1 为 `0.934`，前 48 项为 **28 项风险、20 项拒答**。
- 外部线性探针 F1 为 `0.702`，前 20 项为 **5 项风险、14 项拒答、1 项未决**。
- 外部自评 F1 为 `0.951`，前 20 项找到 **8 项风险**，另 12 项为拒答；直接核查 F1 为 `0.830`，前 20 项找到 **16 项风险**，另 4 项为拒答。

所以，事实断言上的高 F1，并不自动意味着有限复查名额能优先用于已写出的无依据事实。这里的 `30/48`、`5/20` 等是原评测的“已确认风险占选中项比例”，不是新的 F1，也不是对未决项真值已知的精确率。

## 保留主体误报数核对

主集 partial 条件有 40 个保留主体回答项，实际标签均为有依据的事实断言，因此此表正常项分母为 40。已从冻结预测逐项重数，全部方法与 `supplement.json` 一致。

| 方法 | 保留主体 FP / 正常项数 |
|---|---:|
| 线性探针 | 2 / 40 |
| HalluRAG MLP 三个种子（各自统计） | 0 / 40；0 / 40；0 / 40 |
| Lookback Lens | 0 / 40 |
| ReDeEP | 0 / 40 |
| LUMINA | 6 / 40 |
| NLL | 10 / 40 |
| Entropy | 9 / 40 |
| Surface | 3 / 40 |
| 自评风险 | 3 / 40 |
| 直接核查 | 16 / 40 |
| 全部报警 | 40 / 40 |
| 全不报警 | 0 / 40 |

外部集是单主体问答，没有“保留另一主体”这一对照，相关误报率不适用。上述分组只用冻结的删除位置确定；正常或风险仍取实际回答标签。

复核依据：[输入资料](/D:/Projects/Multi_Agent_Graph_Analysis/prelab/round7_evidence_grounding/data/inputs.jsonl)、[完整参考](/D:/Projects/Multi_Agent_Graph_Analysis/prelab/round7_evidence_grounding/data/references.jsonl)、对应逐份生成记录、[主集预测](/D:/Projects/Multi_Agent_Graph_Analysis/prelab/round7_evidence_grounding/results/predictions_main.jsonl)、[外部预测](/D:/Projects/Multi_Agent_Graph_Analysis/prelab/round7_evidence_grounding/results/predictions_external.jsonl)、[主集指标](/D:/Projects/Multi_Agent_Graph_Analysis/prelab/round7_evidence_grounding/results/metrics_main.json)、[外部指标](/D:/Projects/Multi_Agent_Graph_Analysis/prelab/round7_evidence_grounding/results/metrics_external.json)和[补充计数](/D:/Projects/Multi_Agent_Graph_Analysis/prelab/round7_evidence_grounding/results/supplement.json)。没有追加指标、重跑模型或修改任何结果。
