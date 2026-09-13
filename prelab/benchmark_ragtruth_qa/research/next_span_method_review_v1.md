# 下一步跨度检测：方法与落地核查

本轮仅检索一手论文、作者代码并核对本地接口；没有训练、提取新特征或读取 official_test150。以下是待冻结的设计建议，不是已完成实验，也不将我们的结构改造称为原论文基线。

**建议优先做 MVA 启发的多视角注意力序列探针。** 它直接面向词元/跨度检测，能增加目前缓存里没有的信息。逐来源归因保留为第二选择。没有找到可以据现有证据保证解决全部漏报的算法。

## 1. 先明确要解决的缺口

[现有位置诊断](../results/current_span_position_diagnosis_v1/REPORT.md)在原固定阈值下有170个真实风险连续段，其中70个完全没有报警；2145个漏报窗中1397个只属于这些完全漏报段。因此，仅把已有报警向两边延长，无法直接解决主要缺口。这里的170段按风险词元连续性定义，**不是170个独立事实或人工标注span**；这是反复使用的cal159上的事后描述，不能推出整体因果。

## 2. 两个可落地的结构

### A. 多视角注意力＋序列检测器：优先

**一手依据。** [Hallucinated Span Detection with Multi-View Attention Features](https://aclanthology.org/2025.starsem-1.31/)发表于 **\*SEM 2025正式会议**，不是ACL/EMNLP主会。它做词元二分类序列标注，使用每层每头的incoming attention均值、incoming entropy和outgoing entropy，后接Transformer编码器与CRF；不是仅报告整答AUROC的工作。[论文](https://aclanthology.org/2025.starsem-1.31.pdf)、[官方代码](https://github.com/Ogamon958/mva_hal_det)。原文优势主要见于较长上下文任务，不能据此断言在我们QA上超过当前模型。

**这次真正新增什么。** 我们的Lookback把注意力压成“上下文/已输出文字”的比例；它没有保留一个词被后续哪些词关注、这种关注集中还是分散的信息。拟在冻结Llama-2-7B NF4回放中新增每个原始答案BPE的3×32×32=3072维统计，保留所有层头。每个答案作为完整序列；不是先把词元均值压成整句。incoming特征会读取后续答案，序列编码器也会看未来，必须称为**离线回放检测**。

**实现前必须处理的明确差异。** 已核作者commit `f8b871a06b6c18dabe5881bd02a68854b2940b81` 的 [gen_features.py](https://github.com/Ogamon958/mva_hal_det/blob/f8b871a06b6c18dabe5881bd02a68854b2940b81/features/gen_features.py)：`get_features`对完整输入矩阵计算，行号从1起；`key_avg`用逐行乘行号后的整列均值，分母是完整长度T，而论文式(2)写T−j+1。代码incoming entropy沿列归一，且按非零数归一熵，含FP16转换；论文式(5)的归一方向还需和式(4)一起核对。不能混拼公式后声称精确复现。推荐首先锁定**作者raw代码口径**，用小因果矩阵核验流式统计；数值有限性单独检查。我们只取三项MVA统计，不调用作者依赖Llama3特殊标记的Lookback辅助函数。完整答案位置仍用我们已冻结的IDs/坐标。

**建议的单卡版本是本地适配。** 冻结7B，只训练3072→256投影、2层/4头Transformer（FFN512、dropout0.1）和两状态CRF。这是为了8GB预算固定的小结构，不是作者原超参搜索或论文最优结构。作者 [optimize.py](https://github.com/Ogamon958/mva_hal_det/blob/f8b871a06b6c18dabe5881bd02a68854b2940b81/features/optimize.py)确实包含Transformer与CRF，但不能把这个缩小配置登记为原版基线。

**监督与未知位置。** 原人工risk span映射到原lexical BPE即可，无需新大模型标注。非lexical token保留为上下文，不能为了普通CRF接口给它们补“正确”标签。可用部分标注CRF：损失为全部路径log-partition减去与已知lexical标签一致的路径log-partition；范围外未知边缘化。输出连续风险取CRF边缘概率，另外保留Viterbi跨度。该部分标注和概率输出属于我们的评测适配，需小序列枚举核验。

**为何可能补漏报。** 这是增加不同的词元关系信号，再从完整序列学习边界；它有机会发现没有任何旧报警的段，和已有分数HMM平滑不同。但attention形态仍不是资料语义真值，关系反转也可能正常流畅，不能承诺一定改善。

### B. 逐来源依赖估计＋有标签风险探针：备选

**一手依据。** [Learning to Attribute with Attention / AT2](https://arxiv.org/abs/2504.13752)公开了[官方代码](https://github.com/MadryLab/AT2)。本次查到的作者仓库仍用2025 arXiv引用，未确认正式录用，故按**预印本**列出。它用每层头“目标文字→某来源”的注意力作特征，学习来源遮蔽造成的概率变化；原任务是归因，不是幻觉span分类。已核commit `5fd42ba1e1c5978bc6926cbec9dd216dbacda409` 的 [trainer.py](https://github.com/MadryLab/AT2/blob/5fd42ba1e1c5978bc6926cbec9dd216dbacda409/at2/attribution/trainer.py)及 [score_estimators.py](https://github.com/MadryLab/AT2/blob/5fd42ba1e1c5978bc6926cbec9dd216dbacda409/at2/attribution/score_estimators.py)：前者支持逐token目标、默认32个随机mask和负Pearson损失，后者是线性头。不是简单平均所有头。

**我们的可执行改造。** 对每个答案token保留三份原passage各自的1024维attention mass。只在fit材料上，以原三来源的8种保留/遮蔽组合学习小的归因头；遮蔽保持token位置，原答案teacher-force不变。不把“没有影响”直接标幻觉，而是再用原人工risk标签学习局部风险。来源维可用共享投影与无序聚合，避免把passage编号当类别；有显式引用时另比较被引来源与高依赖来源，未引用仍保留。局部边界来自原token监督，不能把整句归因分数机械当作细粒度定位成功。

**成本及限制。** fit最多3680×8=29440次冻结模型前向，cal159只需原输入前向；7B无需反向，CPU小头训练。可流式保留按来源聚合后的特征，不存全部attention。若归因头输出再训练风险头，fit必须按材料组生成OOF归因输出，否则又出现上游训练内分数问题。可共用一份遮蔽缓存完成3组折归因拟合，不需额外LLM前向。8组合、风险头和OOF均为本地改造；不能叫完整AT2幻觉基线。

**它与现有方法的区别。** Lookback只看总关注比例，LUMINA主要看随机上下文下的预测变化，GHOST看内部几何；这里保留“到底是哪份资料影响了哪个词”。但“引用错资料”与“依赖对资料却读错关系”不同，归因只能直接帮助前者。它也可能对来源互相替代、跨来源联合推理失效。因此优先级低于直接有span监督的A。

## 3. 查到的主会方法，为什么不直接启动原训练

**RLSeek，ACL 2026主会。** [论文](https://aclanthology.org/2026.acl-long.1492/)真实输出幻觉字符跨度，在核查推理中显式引用证据；不是仅整答分类。[作者run.sh](https://github.com/WaldenRUC/RLSeek/blob/b4981a2a5b3cbff3d3a64f6ed7507616de704442/run.sh)使用Qwen2.5-7B、每输入16个rollout、8张GPU等配置。故不能把原RL训练列成单8GB可直接执行的方案。压缩成小编码器证据对齐训练是另一种自研结构，还需要可靠的证据定位目标；当前risk标签不自动提供这种目标。发布的RAGTruth训练检查点也不能直接当作未见本cal159的公平基线。

**MIRAGE，EMNLP 2024主会。** [论文](https://aclanthology.org/2024.emnlp-main.347/)与[作者实现](https://github.com/Betswish/MIRAGE/blob/3471fdf95ddf550687e3cb86769b66136370876e/mirage.py)先识别上下文敏感token，再用对比梯度归因到来源。它提供了更细的证据对齐思路，但目标仍是归因；原FP16梯度路径不等于我们已验证可在8GB运行的NF4前向。暂不新增第三个未经资源验证的主方案。

## 4. 本地是否够用，最低执行预算

| 项目 | 核对结论 |
|---|---|
| 输入、答案坐标 | [原793 plans](../data/feature_preparation/plans.jsonl)与[新增3046 plans](../fit_expansion/data/new_token_plans.jsonl)已齐。LUMINA纯输入[完整清点](../results/lumina_qa_preparation_v1/preparation_complete.json)绑定全部3839答：原输入2408466 tokens、答案708506 raw BPE、最长1232，无答案尾部遗落。可以只读其原输入，不用随机输入。 |
| 标签与划分 | 原3680fit/615材料组、159cal不变；原人工lexical/span与4-BPE窗可复用。无需FAVA修复文本、标准答案或新裁判模型。 |
| MVA特征 | 尚未缓存。现[Q/K分块hook](../src/feature_lookback_controls_v2.py)只保存比例，无法还原incoming/outgoing熵；最后层hidden同样不能还原全部层头。需新的一次原输入重放，统计后仅保存原答案轴3072维。 |
| 单卡执行 | 可沿现NF4加载、逐层逐query块重建attention；不在GPU同时存32层完整矩阵。答案key的incoming只需累加当前与后续query，outgoing按完整可见前缀算，不能截短后续答案。最大长度已知，但仍需一次真实最长样例资源自检，不能把估计当已通过。 |
| 存储 | 全708506×3072：float32约8.71GB，float16约4.35GB（十进制、未含坐标）。使用磁盘数组与CPU统计；若保留作者FP16中间口径须记录精度，不静默改为另一公式。 |

建议首轮只做A，**两项固定拟合**：3072维逐token线性探针作为新信号控制；同输入的小Transformer＋部分标注CRF作为序列结构。线性C固定0.01、不搜网格；序列模型固定30轮、单seed、AdamW学习率0.001/衰减0.01、有效batch8答、梯度裁剪1。fit-only标准化；原材料→回答权重保持，序列损失按lexical数量归一。神经与线性差别同时含结构、损失和正则几何，不能只归因CRF。

预算是**一次3839答/约241万输入token的冻结提取＋1个CPU线性模型＋30轮小网络**，卸载7B后再训练小网络，不并占显存。实际墙钟时间要用固定无标签资源样例测吞吐后报告，目前没有该新路径实测，不借旧LR秒数推算。另直接复用已完成Lookback等原baseline，不改它们的模型、阈值或报告粒度。

评测保持原4-BPE窗、cal159全部42241窗/159答：lexical token连续概率→窗max→答max，同一个checkpoint同时报告两级F1；校准只沿原规则选轮次与阈值，不按70个失败段单独选参。同时报告词元AUROC/AP、完全漏报run数、run任一命中与误报数；若与正式Lookback主指标比，另用完整8-BPE几何评分，不能拿4窗F1替代原生8窗AUROC。当前cal已反复用于开发，任何提升都只是开发证据，official_test150保持未读。

**下一步可直接开始的工作是A的CPU公式/坐标接口和单卡提取设计；当前文件不启动这些工作。** 先验证能否补进新信息，再判断是否值得扩为模型。不能因新方法名称或原论文成绩预设能到0.75。
