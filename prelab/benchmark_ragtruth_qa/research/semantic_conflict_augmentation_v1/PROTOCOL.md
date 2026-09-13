# Semantic Conflict Augmentation v1：冻结方案

## 目的与边界

本数据只补强“回答细节与给定资料冲突”的训练信号。它不替代 RAGTruth 人工标签，不作为测试集，也不修改 Lookback Lens、LUMINA、GHOST 等正式 baseline。

- 输入固定为 `data/fit.jsonl` 的 634 个回答、615 个 source group。
- 生成规则只读取问题、三段资料、response/source/group 身份；不按 RAGTruth 标签或现有模型分数筛选。
- 本轮不读 calibration/test，不加载模型，不用 GPU，不训练。
- 正例的依据是原资料中的精确句子；负例是规则生成的 **silver controlled corruption**，不是人工核验的事实金标。

## 最小对构造

每条样本包含同一问题、同一资料、一个支持句，以及只差一个连续槽位的两条 claim。

| 类型 | 支持端 | 扰动端 | 主要约束 |
|---|---|---|---|
| entity | 资料原句 | 换一个同类人名或机构名 | 人名只取 `born`/称谓模式；机构按 academic/public/company 等子类配对；记录 donor group |
| number | 资料原句 | 改一个金额、比例、温度、尺寸等数值 | 排除序号、IP/章节、范围歧义和近邻模糊词 |
| negation | 资料原句 | 插入或删除一个否定 | 排除条件从句、能力情态和明显评价词；保留原句其余字符 |
| temporal | 资料原句 | 改一个年份、月份、星期、时刻或时长 | 排除电话号、近似时长；一次只改一个时间槽 |
| attribution | `Passage p states: 原句` | 仅把 `p` 改成另一个 passage | 原句只在 p 精确出现；目标 passage 的最相似句 Jaccard < 0.35；NLI premise 同时保留 p 与目标 passage |

所有 pair 均保存原/新字符范围、原句哈希、passage/sentence ID、问题和来源身份。`checks` 必须全部为真；严格子集再排除不完整句、标题、外部指代、评价性否定和宽泛能力情态。

实体 donor 可能来自另一个 fit group。为防止折间泄漏，`split_component_groups` 把 owner group、donor group，以及共享精确支持句的 group 合并成连通分量。后续每个 OOF 折必须排除与 held-out group 有交集的全部 pair，不能只按 `group_id` 随机拆分。

## 训练接法

先做两个阶段，任何阶段失败都停止，不直接把 silver 数据并入完整模型。

1. **冻结 NLI 压力测试**：用未改动的 ModernBERT-base-nli 分别评分支持端和扰动端，逐类型报告 `P(C_corrupt > C_supported)`、成对 margin 和 E/N/C 概率。该阶段不训练。某类若不能稳定排序，就不进入下一阶段。
2. **冲突小头**：保持 NLI 编码器冻结，使用 NLI 概率、source-sentence attribution、来源编号关系、生成 log probability 与内部状态摘要训练一个小冲突头。真实 RAGTruth fit 标签仍是主监督；silver pair 只加入 `softplus(r_supported-r_corrupt)` 的相对排序损失，不能把修后整答标成“全对”。

如果要让 silver pair 监督白盒信号，必须把支持/扰动两端分别放回原问题和三段资料下 teacher-force 重放，分别提取目标模型内部状态。不能把原回答 trace 复用到改写端。插入/删除造成长度变化时，以编辑边界附近同一 4-BPE 规则汇总，不伪造空 token 标签。

样本权重先按 split component 等权，再按组件内 pair 等分；类型权重使用逆平方根频率并封顶，避免只有 12 条的 entity 对被极端放大。silver loss 权重只允许在 fit group OOF 中选择，之后冻结。

## 共同评测与通过条件

- 沿用我们的 4-BPE window、answer max、人工标签和共同 evaluator。
- 固定现有 QA source-group OOF 折；每折训练时移除与 held group 相连的 synthetic component。
- 报告总体 window F1/AP、answer F1，以及 EC、SC 的 miss recall 和新增预测 precision。
- 与当前模型融合必须使用 fit OOF 学到的校准头，不能再用无条件 `max/OR`。旧实验表明 OR 会把噪声冲突分数直接变成 FP。
- 只有在 held-fit 上总体 window F1 不降、冲突漏报召回提高且新增预测 precision 达到预注册门槛时，才冻结结构并做一次开发集报告。
- 正式 baseline 只迁移其作者模型到同一数据和评测器；不向 baseline 加本数据、特征、头或调参规则。

## 质检解释

规则在开发样本上收紧后，使用新种子 `qc-audit-v2` 抽取冻结后的最多 20 条/类型。92 条 post-freeze 样本中，研究代理读检为 84 条 clear silver、5 条 ambiguous、3 条 reject。91.3% 只是分层样本的描述性结果，不是独立人标精度或总体置信区间；因此严格集仍须低权重使用，不能称为 gold。

