# 矛盾与无依据：三篇直接相关工作的核查

核查日期：2026-09-12。仅阅读论文与作者公开资源；未下载数据集/模型、未训练、未读取本项目测试回答或标签。

**最有用的是 HAD 的按类型构造与核验规则，以及 ANAH 的显式类别监督。没有查到这三篇能证明“增加矛盾头就一定提高原始词元定位”。** 当前显性冲突召回 195/997、显性无依据 3034/4086 来自项目已有诊断，不是本文重新评测的结果。

| 工作及正式层级 | 实际做什么 | 定位与指标边界 |
|---|---|---|
| [ANAH: Analytical Annotation of Hallucinations in Large Language Models](https://aclanthology.org/2024.acl-long.442/)，ACL 2024 主会长文 | 判别版把 LLM 输出层换成四类线性头：正常、矛盾、无法核实、非事实；生成版另输出参考片段和修正。 | **句子分类**，不是原回答 token/span 标签头。类型 F1/准确率不能当作我们的窗口 F1。 |
| [HAD: HAllucination Detection Language Models Based on a Comprehensive Hallucination Taxonomy](https://aclanthology.org/2026.acl-industry.11/)，ACL 2026 **Industry Track** | Qwen2.5-7B/14B 端到端生成错误类型、错误片段、修正；不是多个独立 token 分类头。类型条件改写制造单片段错误，再按规则过滤。 | 真输出错误片段；论文 76.01 是**仅有幻觉样本**上的 word-F1，不是全部回答的原始 4-BPE 窗口 F1。 |
| [Learning to Reason for Hallucination Span Detection（RL4HS）](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5b0c1a9ce8c71654833210865a256161-Abstract-Conference.html)，ICLR 2026 主会 | 用推理轨迹和片段 F1 奖励训练检测器；修改 GRPO 的优势权重以减少“全部判正常”的偏向。 | 真做 span 定位；按预测/真值覆盖的**字符位置集合**算 F1。其 class 是有/无幻觉，**不是矛盾/无依据**。 |

## 最相关：HAD 可以借什么，不能推出什么

论文表 12 的定义很贴合本项目：**CwIC** 要能仅凭输入反驳，且输出内部自洽；**BI** 引入输入未给出的内容，但不与输入冲突。表 10–12 提供按类型局部改错、确认只改一个错误片段、检查错误类型的流程。这是定向合成与过滤，未使用配对对比损失或困难负例挖掘。

同数据/超参的 HAD-14B 与 Binary 对照，HADTest 二值准确率为 89.10% / 87.77%，但外部数据结果有升有降；**没有两者定位 F1 的匹配消融**。不能据此把类型监督当成已证实的定位增益。HADTest 是合成样例经人工复核、编辑后的集合，不能称自然回答的人工逐词元标注。详细依据：[正文 §3–4、附录表 5、10–14](https://aclanthology.org/2026.acl-industry.11.pdf)。

资源实际状态：[官方仓库](https://github.com/pku0xff/HAD) 发布数据压缩包链接及 `task_input/task_output/hallucination_type/hallucination_span/correction` 格式；已检查的 [utils.py](https://github.com/pku0xff/HAD/blob/main/utils.py) 只格式化、解析生成文本，未提供字符对齐与评分实现；本次未核到权重或完整训练入口。因此不能说“下载即复现”。其省略信息类别甚至可以返回原回答中不存在的片段，我们不能直接统一转成定位金标。

论文明确训练源为 ELI5、Super-NaturalInstructions、GSM8K、FaithDial、Alpaca；评测 HADTest、HaluEval、FactCHD、FaithBench。**未把 RAGTruth 列为训练或评测源**；对 RAGTruth 的引用支持分类定义，不等于使用了该训练集。但本次未检查所有源材料重叠，不能宣称完全独立。未来若取得成品权重，仍须确认实际训练记录；凡用过包含本项目 cal159 的官方训练数据，就不能把本 cal 成绩当公平泛化对比。

## ANAH 与 RL4HS 的可复用部分

**ANAH** 的原论文 §3.2 确实实现四类线性输出层，而不只是文字分类表。其参考片段、类型、修正由 GPT-4 初标再人工复核；约 12K 句子标注。[官方仓库](https://github.com/open-compass/ANAH) 有数据、7B/20B 模型与评估入口。可以借类别定义、完整证据与句子判断方式，不能将句子类别广播成整句所有 token 的精确正标签；修正文字也不能未经对齐核查就当人工风险范围。生成版还混入问答等辅助任务，这与“在现有 token 网络加两个辅助头”不是同一个实验。[论文 §2.4、§3.1–3.2](https://aclanthology.org/2024.acl-long.442.pdf)

**RL4HS** 的 CAPO 是先按 GRPO 标准化奖励，再将无幻觉样本优势乘 α（文中验证集选择 0.5）；不是把矛盾类加权。原实验在 RAGTruth 三任务、7B/14B、8 H100、每组 16 次 rollout 下训练，成本明显超出当前小探针。该文能提醒我们检查训练目标是否偏向易判正常样本，却不能直接解释两种错误召回差距。论文与[作者研究页](https://machinelearning.apple.com/research/hallucination-span-detection)可用，本次未找到作者公开训练代码/检查点；原始数据是公开 RAGTruth。[方法与训练细节](https://arxiv.org/html/2510.02173v2)

## 对当前实验的有限结论

1. 两类标签分开是有正式先例的；是否有收益，仍要由同输入、同预算的二值/双类型对照回答。我们已经做过辅助头，不能仅换叫法当新算法。
2. 如后续现成辅助监督仍不能补足冲突，更针对性的候选是：围绕同一来源保留正常事实，另构造一个只改主体、数字、关系或否定的矛盾版本，再构造无依据版本。此处是**根据文献提出的适配假设**，不是已实施方案，也不能保证改写后其余文字天然为真。
3. 无论新增类型还是数据，继续报告同一候选的全部窗口 F1、整答 F1、各类召回与正常内容误报；本记录不给出新的阈值、采样比例或训练授权。
