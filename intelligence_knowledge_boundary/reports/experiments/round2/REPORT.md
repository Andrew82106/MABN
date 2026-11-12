# 第二轮：更大模型、自己生成的回答和基线审计

状态：五组主实验已完成；完整性核验见报告末尾。

本轮将问题拆开验证：模型和数据是否不匹配，token 时刻是否正确，以及整段标签能否提供足够的定位监督。基线实现范围见 [BASELINES.md](../BASELINES.md)。没有把本地适配称为顶会原始实验的完整复现。

## 最重要的结果

1. 短问答的整段正确性可以检测：7B 的读取答案后状态探针 AUROC 为 0.868，简单词概率对照为 0.806。这是参考答案匹配任务，不是逐 token 幻觉定位；不能与论文不同数据和模型的数值直接比较。
2. 人工新闻上的弱监督定位仍弱：7B HaMI-before 的回答内 token AUROC 为 0.578；直接提供错误位置标签后的状态线性探针为 0.688。有局部可读信号，但当前整段监督未稳定提取它。
3. 定位排序有效不代表能可靠报警：同一个监督状态探针在原测试集正常回答误报率为 3.1%，错误回答检出率仅 4.0%。该集只有 25 条错误回答，不能把小样本点估计当成部署表现。
4. 三事实自动标签不够可靠：逐条对照材料审阅测试集全部 14 条候选错误，只有 4 条明确事实冲突，3 条实际受材料支持，7 条含糊、答非所问或问题不当。因此这组仅作自动匹配诊断；即使表中定位指标高，也不能证明事实性幻觉检测有效。审阅由执行助手完成，未取得独立双人标注。

AUROC 是排序指标，0.5 约为随机；0.69 不代表 69% 准确率。

## 可以继续推进的切口

最有依据的方向是“有限位置标注下的证据冲突定位”，先辅助人工复核，再验证自动报警。不要把现阶段工作命名为已经实现了可靠的联网幻觉监测。

| 训练回答中使用位置标注的数量 | 0.5B 定位 AUROC | 7B 定位 AUROC |
|---:|---:|---:|
| 24 | 0.582 | 0.580 |
| 60 | 0.592 | 0.590 |
| 120 | 0.626 | 0.632 |
| 240 | 0.668 | 0.660 |

上述对照仅使用训练集所选回答的位置标签，验证集只用整段标签，三种子平均。当前更一致的变化来自增加位置标注，扩大模型没有同样稳定的收益。这是探索线索，尚非新算法或已证实的显著提升。

下一阶段应先建“可信材料—自然生成回答—事实冲突位置”的小型人工核验集，并按事件隔离。先限定人物、地点、时间、数量等明确字段；把材料支持、材料冲突、材料不足分开，后两者不能混标。之后固定标注预算，对比零位置标注、少量位置标注及完整位置标注；以固定误报下的检出率和人工复核命中率为主要目标。此为后续研究建议，本轮没有再启动这些实验。

## 未使用过的新闻复核

额外冻结 110 条人工标注新闻（80 正常、30 错误），来源与原 394 条训练／验证／测试均无重叠。它们来自官方训练部分，但在本研究中全程留出；不是额外的官方测试集。

在查看原测试结果后，仅增加两个受监督对照：同一个注意力 MLP 学错误位置，以及额外惩罚正常回答的最高风险。系数固定为 0／0.5，三种子；用原验证集的低误报检出率选训练步。新留出集不用于拟合、选参数、选择方法或校准阈值。这个扩展是探索性的，不声称算法原创。

| 方法 | 回答内定位 AUROC | 全体 token AUROC | 正常回答误报率 | 错误回答检出率 |
|---|---:|---:|---:|---:|
| HaMI ori / before | 0.587 | 0.570 | 9.6% | 6.7% |
| Attention / weak | 0.570 | 0.570 | 2.9% | 6.7% |
| Attention MLP / local labels | 0.709 | 0.728 | 5.4% | 7.8% |
| Attention MLP / local labels + alarm loss | 0.730 | 0.719 | 9.2% | 17.8% |
| States after / local labels | 0.655 | 0.646 | 0.0% | 0.0% |
| Attention / local labels | 0.688 | 0.708 | 2.5% | 0.0% |
| Surprisal | 0.585 | 0.569 | 6.2% | 3.3% |

复核结果：位置监督 MLP 的定位 AUROC 为 0.709，同特征整段监督为 0.570；按来源重采样的差值 95% 区间约为 [0.060, 0.216]。这支持位置监督有用，但两者的监督信息不同，不能作为新算法公平胜出。与已经使用位置标签的线性探针相比，差值区间约为 [-0.009, 0.050]，没有证明 MLP 稳定更好。

额外误报惩罚未形成可靠突破：错误回答检出率由 7.8% 升至 17.8%，但正常回答误报也从 5.4% 升至 9.2%；局部排序改善的差值区间包含 0。不能仅挑检出率提高来宣称抗幻觉监测成功。

[每个种子的完整数值](confirmation/metrics.json)，[来源重采样区间](confirmation/confidence_intervals.json)，[冻结的模型哈希](confirmation/frozen_models.json)，[数据来源及隔离](../data/news_confirmation/manifest.json)。训练种子均值不等于独立重复抽取的数据集；30 条错误的检出率波动较大。验证集仅 40 条正常回答，约 5% 的经验阈值不保证现实误报率也不超过 5%。


![各实验指标](comparison.png)

- 新闻：394 条 RAGTruth 人工标注回答，同一批输入分别由 0.5B 和 7B 重放；属于受控表征比较。
- TriviaQA：每个模型独立生成 2,700 个短答，按问题隔离。此项评估参考目标答案的正确性，没有可靠的逐 token 错误金标。
- 三事实 RAG：7B 按 1,050 段文章、每段三个问题生成。训练只用整段候选错误标签；局部标签来自单题答案匹配，和人工事实核验分开解释。0.5B 只做了流程试跑，由于格式与标签可用性差，没有完成此项对比，见 [停跑依据](rag_small_pilot_decision.json)。
- 7B 使用第三方 Unsloth NF4 量化版，模型规模与精度同时变化，不能把差异完全归因于参数量。
- 弱监督方法报告三种子的均值；直接局部监督是单独的诊断对照。局部金标用于监督训练的结果不能冒称只用了整段标签。
- 高分是相对风险，未作概率校准。定位是在完整回答上离线排名；生成前和生成后状态分开报告。
- NLL 与证据概率对照依赖已经选出的 token；这些组合方法属于输出后监测，不是提前预测具体错误词。

## 标签与评估限制

新闻的 Evident Conflict 是相对提供证据的明显冲突；不等于重新核验了全部现实事实。固定材料模拟联网检索后的阅读阶段，没有运行实时搜索代理。

自动匹配已排除问题复述导致的别名误匹配、部分词形歧义、拒答及多句附带断言，但仍可能误判语义等价、含糊问题和错误参考。抽查说明见 [标签审计](label_rule_audit.json)、[三事实逐条审阅](rag_large_label_review.json)、[大模型短答审阅](trivia_large_label_review.json)。不能将 QA 候选错误的高检出率直接等同于开源情报事实性幻觉检出率。

TriviaQA 是历史问答基准，其参考答案不等于 2026 年的现时事实；未逐题重做时间有效性核验。

整段分类和回答内部定位分别统计。后者在同时含正常与错误 token 的回答中计算，随机排序约为 0.5；RAG 只评估可判定答案字符串内部，排除 JSON 标点。三事实排序在同一回答既有正确事实又有候选错误时计算。置信区间按来源／文章／问题聚类重采样，不能把 token 当作独立样本扩大显著性。

另报告全体回答的 token 排序与固定阈值报警。条件定位高，不代表能够区分哪一段回答真的含错；误报控制与错误回答检出率必须一起看。报警阈值只使用验证集的正常整段标签，不读取验证集错误位置。事实片段方法需要等该片段完整输出后才能计算分数。

RAG 的局部金标把错误答案字符串整体标为错误，是事实片段粒度；新闻则使用原始错误字符区间。两者不能作为同一定位难度直接横向比较，也不能把事实片段内的统一分数解释为已经找出了最小错误词。

更换特征、局部监督的比较属于探索性分析；不预设正结果，不将多个对照里偶然最高的一项包装为稳定突破。


## news_large

划分与可用标签：`{"train": {"total": 240, "usable": 240, "errors": 100}, "val": {"total": 65, "usable": 65, "errors": 25}, "test": {"total": 89, "usable": 89, "errors": 25}}`。

| 方法 | 标签 | 整段 AUROC | 回答内 token AUROC | 三事实排序 AUROC | 最高 10% 精确率 |
|---|---|---:|---:|---:|---:|
| HaMI ori / before | 整段／无需训练 | 0.585 | 0.578 | — | 0.180 |
| HaMI ori / after | 整段／无需训练 | 0.576 | 0.580 | — | 0.215 |
| Last state / linear | 整段／无需训练 | 0.539 | 0.530 | — | 0.168 |
| Mean state / linear | 整段／无需训练 | 0.634 | 0.559 | — | 0.152 |
| Attention / weak | 整段／无需训练 | 0.615 | 0.550 | — | 0.159 |
| Evidence + attention / weak | 整段／无需训练 | 0.603 | 0.533 | — | 0.135 |
| Surprisal | 整段／无需训练 | 0.579 | 0.572 | — | 0.202 |
| Source string match | 整段／无需训练 | 0.517 | 0.510 | — | 0.122 |
| Text TF-IDF | 整段／无需训练 | 0.649 | — | — | — |
| Attention / 10% local training labels | 局部位置 | 0.519 | 0.580 | — | 0.182 |
| Attention / 25% local training labels | 局部位置 | 0.537 | 0.590 | — | 0.192 |
| Attention / 50% local training labels | 局部位置 | 0.570 | 0.632 | — | 0.233 |
| Attention / all local labels, bag validation | 局部位置 | 0.589 | 0.660 | — | 0.247 |
| States before / local labels | 局部位置 | 0.635 | 0.632 | — | 0.252 |
| States after / local labels | 局部位置 | 0.560 | 0.688 | — | 0.294 |
| Attention / local labels | 局部位置 | 0.589 | 0.660 | — | 0.247 |
| Attention MLP / local labels | 局部位置 | 0.604 | 0.672 | — | 0.257 |
| Attention MLP / local labels + alarm loss | 局部位置 | 0.585 | 0.676 | — | 0.246 |

完整结果：[metrics.csv](news_large/metrics.csv)，[置信区间](news_large/confidence_intervals.json)。

所有正常和错误回答一起评估：仅使用验证集正常回答的最高 token 分数，将验证误报控制在约 5% 以内，再固定阈值测试。以下检出率是“能否对错误回答发出至少一次警报”，与前表的条件定位指标不同。

| 方法 | 全部 token AUROC | 正常回答误报率 | 错误回答检出率 | 全局最高 10% token 精确率 |
|---|---:|---:|---:|---:|
| HaMI ori / before | 0.579 | 0.083 | 0.120 | 0.045 |
| Evidence + attention / weak | 0.526 | 0.062 | 0.080 | 0.037 |
| States after / local labels | 0.615 | 0.031 | 0.040 | 0.064 |
| Attention / local labels | 0.658 | 0.062 | 0.040 | 0.089 |
| Attention MLP / local labels | 0.667 | 0.083 | 0.027 | 0.089 |
| Attention MLP / local labels + alarm loss | 0.669 | 0.062 | 0.173 | 0.084 |

[全部测试回答及错误位置](news_large/token_risk_report.html)。

## news_small

划分与可用标签：`{"train": {"total": 240, "usable": 240, "errors": 100}, "val": {"total": 65, "usable": 65, "errors": 25}, "test": {"total": 89, "usable": 89, "errors": 25}}`。

| 方法 | 标签 | 整段 AUROC | 回答内 token AUROC | 三事实排序 AUROC | 最高 10% 精确率 |
|---|---|---:|---:|---:|---:|
| HaMI ori / before | 整段／无需训练 | 0.567 | 0.556 | — | 0.155 |
| HaMI ori / after | 整段／无需训练 | 0.559 | 0.566 | — | 0.172 |
| Last state / linear | 整段／无需训练 | 0.521 | 0.592 | — | 0.205 |
| Mean state / linear | 整段／无需训练 | 0.585 | 0.517 | — | 0.117 |
| Attention / weak | 整段／无需训练 | 0.583 | 0.558 | — | 0.171 |
| Evidence + attention / weak | 整段／无需训练 | 0.590 | 0.570 | — | 0.170 |
| Surprisal | 整段／无需训练 | 0.536 | 0.561 | — | 0.188 |
| Source string match | 整段／无需训练 | 0.517 | 0.510 | — | 0.122 |
| Text TF-IDF | 整段／无需训练 | 0.649 | — | — | — |
| Attention / 10% local training labels | 局部位置 | 0.499 | 0.582 | — | 0.177 |
| Attention / 25% local training labels | 局部位置 | 0.546 | 0.592 | — | 0.164 |
| Attention / 50% local training labels | 局部位置 | 0.588 | 0.626 | — | 0.214 |
| Attention / all local labels, bag validation | 局部位置 | 0.592 | 0.668 | — | 0.236 |
| States before / local labels | 局部位置 | 0.583 | 0.649 | — | 0.242 |
| States after / local labels | 局部位置 | 0.581 | 0.661 | — | 0.259 |
| Attention / local labels | 局部位置 | 0.606 | 0.671 | — | 0.240 |

完整结果：[metrics.csv](news_small/metrics.csv)，[置信区间](news_small/confidence_intervals.json)。

所有正常和错误回答一起评估：仅使用验证集正常回答的最高 token 分数，将验证误报控制在约 5% 以内，再固定阈值测试。以下检出率是“能否对错误回答发出至少一次警报”，与前表的条件定位指标不同。

| 方法 | 全部 token AUROC | 正常回答误报率 | 错误回答检出率 | 全局最高 10% token 精确率 |
|---|---:|---:|---:|---:|
| HaMI ori / before | 0.549 | 0.021 | 0.040 | 0.040 |
| Evidence + attention / weak | 0.562 | 0.073 | 0.173 | 0.048 |
| States after / local labels | 0.643 | 0.062 | 0.040 | 0.070 |
| Attention / local labels | 0.645 | 0.047 | 0.160 | 0.078 |

[全部测试回答及错误位置](news_small/token_risk_report.html)。

## rag_large

划分与可用标签：`{"train": {"total": 700, "usable": 331, "errors": 93}, "val": {"total": 150, "usable": 66, "errors": 20}, "test": {"total": 200, "usable": 105, "errors": 14}}`。

| 方法 | 标签 | 整段 AUROC | 回答内 token AUROC | 三事实排序 AUROC | 最高 10% 精确率 |
|---|---|---:|---:|---:|---:|
| HaMI ori / before | 整段／无需训练 | 0.682 | 0.612 | 0.667 | 0.552 |
| HaMI ori / after | 整段／无需训练 | 0.694 | 0.664 | 0.726 | 0.611 |
| Last state / linear | 整段／无需训练 | 0.524 | 0.674 | 0.714 | 0.488 |
| Mean state / linear | 整段／无需训练 | 0.802 | 0.681 | 0.643 | 0.762 |
| Attention / weak | 整段／无需训练 | 0.637 | 0.627 | 0.667 | 0.524 |
| Evidence + attention / weak | 整段／无需训练 | 0.666 | 0.701 | 0.750 | 0.667 |
| Three-fact MIL / states | 整段／无需训练 | 0.735 | 0.747 | 0.762 | 0.690 |
| Three-fact MIL / evidence + attention | 整段／无需训练 | 0.642 | 0.732 | 0.714 | 0.667 |
| Surprisal | 整段／无需训练 | 0.554 | 0.620 | 0.786 | 0.833 |
| Source string match | 整段／无需训练 | 0.522 | 0.524 | 0.518 | 0.429 |
| Text TF-IDF | 整段／无需训练 | 0.411 | — | — | — |
| States before / local labels | 局部位置 | 0.717 | 0.720 | 0.821 | 0.643 |
| States after / local labels | 局部位置 | 0.726 | 0.747 | 0.821 | 0.786 |
| Attention / local labels | 局部位置 | 0.608 | 0.711 | 0.786 | 0.619 |

完整结果：[metrics.csv](rag_large/metrics.csv)，[置信区间](rag_large/confidence_intervals.json)。

所有正常和错误回答一起评估：仅使用验证集正常回答的最高 token 分数，将验证误报控制在约 5% 以内，再固定阈值测试。以下检出率是“能否对错误回答发出至少一次警报”，与前表的条件定位指标不同。

| 方法 | 全部 token AUROC | 正常回答误报率 | 错误回答检出率 | 全局最高 10% token 精确率 |
|---|---:|---:|---:|---:|
| HaMI ori / before | 0.733 | 0.106 | 0.310 | 0.183 |
| Evidence + attention / weak | 0.697 | 0.073 | 0.262 | 0.215 |
| Three-fact MIL / states | 0.823 | 0.066 | 0.310 | 0.242 |
| Three-fact MIL / evidence + attention | 0.755 | 0.110 | 0.214 | 0.231 |
| States after / local labels | 0.827 | 0.110 | 0.429 | 0.274 |
| Attention / local labels | 0.719 | 0.110 | 0.286 | 0.218 |

[全部测试回答及错误位置](rag_large/token_risk_report.html)。

## trivia_large

划分与可用标签：`{"train": {"total": 2000, "usable": 1963, "errors": 916}, "val": {"total": 300, "usable": 298, "errors": 129}, "test": {"total": 400, "usable": 390, "errors": 175}}`。

| 方法 | 标签 | 整段 AUROC | 回答内 token AUROC | 三事实排序 AUROC | 最高 10% 精确率 |
|---|---|---:|---:|---:|---:|
| HaMI ori / before | 整段／无需训练 | 0.823 | — | — | — |
| HaMI ori / after | 整段／无需训练 | 0.868 | — | — | — |
| Last state / linear | 整段／无需训练 | 0.794 | — | — | — |
| Mean state / linear | 整段／无需训练 | 0.824 | — | — | — |
| Surprisal | 整段／无需训练 | 0.806 | — | — | — |
| Text TF-IDF | 整段／无需训练 | 0.556 | — | — | — |

完整结果：[metrics.csv](trivia_large/metrics.csv)，[置信区间](trivia_large/confidence_intervals.json)。


## trivia_small

划分与可用标签：`{"train": {"total": 2000, "usable": 1761, "errors": 1424}, "val": {"total": 300, "usable": 264, "errors": 207}, "test": {"total": 400, "usable": 345, "errors": 270}}`。

| 方法 | 标签 | 整段 AUROC | 回答内 token AUROC | 三事实排序 AUROC | 最高 10% 精确率 |
|---|---|---:|---:|---:|---:|
| HaMI ori / before | 整段／无需训练 | 0.767 | — | — | — |
| HaMI ori / after | 整段／无需训练 | 0.789 | — | — | — |
| Last state / linear | 整段／无需训练 | 0.726 | — | — | — |
| Mean state / linear | 整段／无需训练 | 0.777 | — | — | — |
| Surprisal | 整段／无需训练 | 0.752 | — | — | — |
| Text TF-IDF | 整段／无需训练 | 0.676 | — | — | — |

完整结果：[metrics.csv](trivia_small/metrics.csv)，[置信区间](trivia_small/confidence_intervals.json)。

## 完整性与追溯

- [final_audit.json](final_audit.json)：通过。
- [alert_metric_audit.json](alert_metric_audit.json)：通过。
- [generation_audit_large.json](generation_audit_large.json)：通过。
- [generation_audit_small.json](generation_audit_small.json)：通过。
- [token_repair_rag_large.json](token_repair_rag_large.json)：通过。

RAG 全部 1,050 条保存生成时实际 token ID，并按原 ID 重放；11 条因解码后重新切词不同而修复。此前 Trivia 没有完整保存生成轨迹，只进行了固定抽样的同种子再生成检查；不能把抽样通过写成所有回答实时状态已逐一验证。

模型权重与数据哈希见 [模型清单](model_manifest.json) 和各数据清单；代码哈希包含在最终完整性核验中。指标核验不等于标签真实性、因果机制或现实部署有效性证明。
