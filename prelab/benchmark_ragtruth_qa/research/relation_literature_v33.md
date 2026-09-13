# 关系感知与白盒细粒度幻觉检测：2024–2026 文献核查 v33

更新：2026-09-12。范围只保留能直接帮助当前“检索资料 + 生成回答 + 定位无依据片段”任务的顶会工作；Lookback Lens、LUMINA、ReDeEP、RefChecker、HalluRAG 不再重复。

## 结论

下一版不宜继续把激活值和 log probability 简单拼接。最有依据的组合是：

1. 用**关系型最小事实单元**组织标签：保留主体、客体、关系、否定、条件、时间及其原文位置。
2. 用 **RAGLens 的 SAE 稀疏特征**和 **HARP 的低维投影**替代高维原始激活。
3. 用 **FLaG 的少量潜在组路由 + 组内分类器 + log-marginal 汇总**融合内部激活、注意力和概率轨迹。
4. 用“只改一个关系槽位”的真假配对样本加入排序损失，并显式训练回答片段与证据的对应关系。

这四步分别针对当前最明显的困难：高维激活过拟合、不同错误机制混在一起、关系/角色/范围错误难学、只有真假标签而缺少证据约束。

## 最值得迁移的工作

| 工作 | 原方法实际检测粒度 | 对当前项目最有用的部分 | 使用边界 |
|---|---|---|---|
| [RAGLens，ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/b6f6bfbd260fbf2f5acb0a1d6439ca0e-Abstract-Conference.html)；[代码](https://github.com/Teddy-XiongGZ/RAGLens) | 用回答级真假标签训练：逐 token 隐状态经 SAE 稀疏编码，跨 token 最大池化，再用互信息选特征和 GAM 分类 | 保留池化前的 token SAE 激活，按本项目 4-BPE 窗口汇总；在训练折内做互信息筛选；用可解释 GAM 先做单信号实验 | 论文中的 token 热区是模型归因，不是用 gold span 训练或评测的定位器。其 RAGTruth Llama2-7B macro-F1 0.7636 是**回答级**，不能与窗口 F1 比 |
| [HARP，ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/1e58b1bf9f218fcd19e4539e982752a5-Abstract-Conference.html)；[预印本](https://arxiv.org/abs/2509.11536) | 将每个 token 隐状态投影到 unembedding 矩阵 SVD 的尾部约 256 维，再用回答级标签和 max pooling 训练 | 这是成本最低的白盒降维试验；可直接在现有 4-BPE 窗口标签上训练，检验低维投影能否减少过拟合 | 原论文仍是回答级弱监督；“reasoning subspace”是论文提出的解释，不等于已证明的真实推理子空间 |
| [FLaG，KDD 2026](https://arxiv.org/abs/2606.00301)；[ACM](https://doi.org/10.1145/3770855.3818137) | 回答级检测；将末 token、回答均值、问答漂移与 7 个概率轨迹特征投影后，由潜在组路由到多个线性检测器，并以 log-marginal 汇总 | 回答了“两个信号怎样复杂结合”：让不同专家分别处理低概率、证据脱离、语义漂移等机制；真假配对用排序损失 | “fine-grained”指潜在错误组，不是逐 token 定位。论文使用 K=64；当前小数据必须只试 K=2/4/8，并用组外验证 |
| [TOHA，ACL 2026](https://aclanthology.org/2026.acl-long.704/)；[代码](https://github.com/sb-ai-lab/TOHA) | 把注意力头变成图，以连接回答 token 到提示/证据 token 的最小生成森林代价检测回答级幻觉 | 可把森林边代价分摊到 token 或窗口，形成比简单 lookback 比率更有结构的“脱离资料”信号 | 官方结果是回答级 AUROC；短窗口图可能退化。正式基线必须保留原算法，局部化版本另列为本项目适配方法 |
| [RLSeek，ACL 2026](https://aclanthology.org/2026.acl-long.1492/)；[代码](https://github.com/WaldenRUC/RLSeek) | 真正评测 hallucinated span；要求推理过程逐步引用原资料，并用 span-F1 与引用合法性奖励训练 | 最关键的启示是给每个关系事实保存**精确证据片段**，再增加“目标窗口应更接近其证据而非困难负证据”的对比损失 | 它是生成式强化学习检测器，不是轻量白盒探针；RAGTruth 7B span-F1 57.5、sample-F1 82.6，说明回答级高分不代表定位已经解决 |
| [Teaching Language Models to Check Grounded Claim Factuality with Human Test-Taking Strategies，ACL 2026](https://aclanthology.org/2026.acl-long.1468/)；[代码/数据](https://github.com/Haruhi07/Test-Taking) | 先拆原子事实，再依次检查主体/客体、描述、关系和可推断信息；用 SFT + DPO 训练小检查器 | 当前最适合的数据构造模板：只替换主体、客体、谓词，或改变否定、细节和条件，生成最小真假对；变化处可自动得到精确 span 标签 | 这是语义检查器，可作为独立基线或教师；不能把它偷偷并入其他正式基线 |
| [VeriFact，EMNLP 2025](https://aclanthology.org/2025.emnlp-main.905/)；[代码](https://github.com/launchnlp/VeriFact) | 修复事实拆分时漏掉的时间、条件、比较对象等信息 | 建立关系单元质检：每条必须保留主体、客体、谓词、否定、比较对象、条件和时间，并保留源位置 | 其多模型精炼成本高；当前先做规则审计和抽样人工复核。论文发现 24.9% 的不完整事实在修复后改变了事实性标签 |
| [DnDScore，EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1205/) | 区分“要核查的目标子事实”和为消歧补入的上下文，避免验证器误判附加内容 | 每个样本保存三元组：`目标输出 span / 补全后的上下文载体 / 来源证据 span`；标签只落在目标 span | 可解决代词补全后把无关词也标成幻觉的问题 |
| [Decomposition Dilemmas，NAACL 2025](https://aclanthology.org/2025.naacl-long.320/)；[代码](https://github.com/qishenghu/Decomp_Dilemmas) | 系统分析事实拆分的遗漏、逻辑关系丢失、指代歧义、过拆和改义 | 数据质检不应只看标点；关系、因果、比较和条件必须留在同一个可核查单元中 | 是拆分质量研究，本身不是白盒检测器 |
| [Claimify，ACL 2025](https://aclanthology.org/2025.acl-long.348/)；[数据](https://huggingface.co/datasets/microsoft/claimify-dataset) | 先选事实句，再消歧，最后拆成原子事实 | 可补充事实拆分训练数据与标注格式；其 6,490 句、396 个回答可用于测试拆分器迁移性 | 没有直接提供 token 白盒检测方法 |

## 可补充但优先级较低

- [SiGHT，AISTATS 2026](https://proceedings.mlr.press/v300/chen26d.html) 用 Word2Vec 近邻替换制造伪事实并以图网络分类；RAGTruth-QA F1 仅 45.90±6.90，适合借鉴“成对反事实数据”，不适合作为主模型。
- [Real-time Factuality Assessment from Adversarial Feedback，ACL 2025](https://aclanthology.org/2025.acl-long.81/)；[数据/代码](https://github.com/sanxing-chen/adv-fake) 提供 2024 年实时新闻及迭代伪造文本，可作为开源情报领域外测试。它只有文章级真假标签；若用于定位，必须恢复实际改写 span 并抽样人工核验。
- [The Missing Parts / TRACER，EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1724/)；[代码](https://github.com/tangyixuan/TRACER) 研究“说了真的但故意漏掉关键事实”。它更适合后续研究“应查 5 个主体却漏了 2 个”的覆盖性错误，不属于当前无依据输出 span 的主任务。

## 建议的新模型与数据形式

每个训练实例不再只是“一个窗口 + 真假标签”，而是：

```text
question
retrieved_documents
answer
target_window_span
atomic_relation = (subject, predicate, object, qualifier, time, condition, polarity)
support_or_contradict_evidence_span
window_label
answer_label
counterfactual_pair_id
```

对每个 4-BPE 窗口提取四类输入：

- SAE 稀疏激活：来自 RAGLens 思路；
- HARP 低维投影激活；
- 资料注意力/TOHA 图脱离度；
- token log probability、熵、top-1/top-2 margin、低概率词比例等概率轨迹。

融合器先把每类信号单独投影，再用 K=2/4/8 的软路由选择小型组内分类器，最终用 log-marginal 得到窗口风险。训练损失建议为：

```text
L = L_window_BCE + λ1·L_counterfactual_rank + λ2·L_evidence_align
```

- `L_window_BCE` 学当前窗口真假；
- `L_counterfactual_rank` 要求最小改错窗口风险高于对应真窗口；
- `L_evidence_align` 要求目标事实更接近正确证据，并远离同实体但关系错误的困难负证据。

回答级风险由事实单元风险汇总；窗口级指标仍按项目既定 4-BPE 标注计算。特征筛选、路由和阈值都只能在训练折/验证折拟合，来源相同、问题相同、反事实配对相同的样本必须落在同一数据划分，防止近重复泄漏。

## 公平基线与执行顺序

正式基线保持作者结构和训练目标，不为提高或压低分数而改动：RAGLens、HARP、FLaG、TOHA 按其原始回答级输出评测回答级指标；RLSeek 按官方 span 输出评测本项目 span/window 指标。若要把回答分数广播给窗口以满足统一报表，只能作为固定的评测映射并明确标注，不能称为基线的 token 能力。

建议依次做：

1. **HARP 窗口投影**：最便宜，先确认降维是否缓解过拟合。
2. **RAGLens SAE 窗口特征**：单独评测，避免一开始就混入旧信号。
3. **关系最小反事实数据**：优先补主体、客体、谓词、否定、数字/日期、条件/比较六类。
4. **低 K 的 FLaG 式融合**：与简单拼接 MLP 做严格消融。
5. **证据对齐损失**：只在前四步已有稳定组外收益后加入。
6. **TOHA 与 RLSeek 正式基线**：前者计算较轻；后者训练较重，但提供真正的 span 级参照。

评判时必须把回答级 F1/AUROC 与窗口/span F1 分开报告。现有文献中，回答级达到 0.8 以上很常见，真正的 span F1 仍约在 0.6 左右；因此“回答很准”不能推出“具体位置也很准”。
