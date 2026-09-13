# 4-BPE RAG 幻觉定位：可落地的新方法筛选（截至 2026-09）

## 先给结论

当前最值得做的不是再堆一组静态 hidden-state 数值，而是按以下顺序补信息：

1. **先查模型实际受哪份资料影响，再查那份资料是否支持它。** 这是最贴合“模型用了检索资料，还是自己猜”的改动，也最可能补当前几乎漏掉的显性冲突。
2. **把 Lookback 的汇总注意力换成 UHead 的原始局部注意力结构。** 它是最直接、有公开实现的结构升级。
3. **把词元看成有先后依赖的风险序列，单独学习错误开始点。** 当前最早风险窗明显比后续风险窗难检出，独立逐窗分类没有利用这个结构。

当前开发集最好值是 4-BPE F1 **0.690281**；它来自反复使用的 cal159，不是独立测试成绩。下面的论文数字也不是同数据、同指标，因而只能说明“值得试”，不能保证超过 0.69。

## 1. 因果归因 × 三类证据核查（首选）

### 原论文分别做了什么

- [ContextCite，NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/adbea136219b64db96a9941e4249a857-Abstract-Conference.html)（[官方代码](https://github.com/MadryLab/context-cite)）对资料做删除实验，用“删除某资料后原陈述概率下降多少”判断模型**实际依赖**了哪部分上下文。论文允许被解释对象是回答中任意连续词元片段。它做的是 contributive attribution，**不负责判断资料是否真的支持陈述**。
- [SelfCite，ICML 2025](https://proceedings.mlr.press/v267/chuang25a.html)（[官方代码](https://github.com/facebookresearch/SelfCite)）把反事实拆成两面：删掉被引资料后原陈述应变难，叫**必要性**；只保留被引资料时原陈述仍应容易，叫**充分性**。原论文用它训练/筛选引用，不是幻觉检测器；这里仅迁移这两个信号的定义。
- [RefChecker / Knowledge-Centric Hallucination Detection，EMNLP 2024](https://aclanthology.org/2024.emnlp-main.395/)（[官方代码](https://github.com/amazon-science/RefChecker)）把回答拆成 `(主体, 关系, 客体)` 命题，再对资料判 `Entailment / Neutral / Contradiction`。它还给出基于 RoBERTa span embedding 的证据定位基线。它判断**资料是否支持命题**，但不判断生成模型当时真正用了哪份资料。
- [RAGChecker，NeurIPS 2024 Datasets & Benchmarks](https://proceedings.neurips.cc/paper_files/paper/2024/hash/27245589131d17368cccdfa990cbf16e-Abstract-Datasets_and_Benchmarks_Track.html)同样用 claim entailment 把“无资料支撑的自生内容”和“受错误/噪声资料影响”分开，适合作为类型定义与评测依据。

### 我们的适配是什么

RAGTruth QA 每条恰好有 3 份 passage，所以不必学习 ContextCite 的 LASSO 近似；直接穷举 `2^3=8` 个资料子集，对**原回答同一词元序列**做 teacher forcing：

- 对每个 claim、每份 passage 得到删除必要性、单独保留充分性和精确三来源 Shapley/交互量；
- 对同一 claim 与每份 passage 保留 `E/N/C` 三类概率；
- 形成四种很直白的状态：`高依赖+支持`、`高依赖+冲突`、`低依赖+均无证据`、`高依赖+无证据`；
- 将 claim 向量广播给其覆盖的 4-BPE 窗口，再与现有生成模型白盒分数做一个小门控/树模型。

这一步与现有 MiniCheck 不重复：当前 MiniCheck 主要是二元“支持/不支持”且多来源常取最大值；它没有“冲突”类别，也没有“生成模型到底受哪一来源影响”的因果量。已有显式引用选源没有明显收益，因而这里应使用模型反事实归因，不能只相信回答自己写的引用编号。

### 资源与超过 0.69 的依据

- Llama 缓存：8 个 float32 词元 log-prob × 708,506 BPE，约 **22.7 MB**；另存 claim-source 的 E/N/C 很小。
- 推理：最多 `8 × 3,839 = 30,712` 次回答级前向；按原输入长度作宽松上界约 **1,927 万输入词元**，实际因删资料更少。RTX 3070 上应先跑固定 20 条无标签资源样本测墙钟，再决定全量。
- 最有希望的原因是当前 `Evident Conflict` 窗口只检出 `195/997=19.56%`，而无依据类已到 74.25%。`高依赖+冲突` 正对这一缺口；若它仍无增量，就说明主要瓶颈在 claim/金标边界，而不是缺少资料依赖信号。

### baseline 与 ours 的边界

- **原生 baseline**：官方 RefChecker extractor + checker 原样运行；仅把其命题结果映射到共同 4-BPE 评测坐标。ContextCite本身是归因器，不能把它包装成原生幻觉分类器。
- **ours**：8 子集精确归因、归因与 E/N/C 的交叉特征、窗口映射及与生成模型白盒信号的融合。论文没有给出这一组合。

## 2. UHead 原始局部注意力 + 概率序列（第二优先）

[UHead，EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1809/)（[官方代码](https://github.com/IINemo/llm-uncertainty-head)）不再把注意力压成 Lookback ratio。对每个生成词元，它保留每层每头指向前 `k` 个词元的原始注意力，再拼接 top-m 词元概率的对数；经过线性降维、1–2 层 Transformer、claim marker、claim 内平均池化和两层分类器输出风险。

论文在 Mistral biography 开发集的同结构消融中，`UHead+Lookback特征` PR-AUC 0.609，`原始注意力` 0.617，`原始注意力+概率` 0.642；这是其数据上的 PR-AUC，不可与本项目 F1 比。论文还明确报告 hidden state 容易很快过拟合，而原始注意力较稳。

**迁移 baseline**：保持官方 UHead 结构，把每个 4-BPE 窗口当作 claim mask；Llama2 没有作者预训练 head，需在本项目 fit 资料组上训练。先固定 `k=2`，因为论文在 Mistral/Gemma 的最优值就是 2；不要复制作者耗时约 150 GPU-hours 的全网格搜索。

**ours 的后续升级**：在 UHead 输出前加入方案1的 claim-source 交叉向量，或在最终门控层融合；这才是新模型，不能把融合成绩记成 UHead baseline。

- 回答词元最低缓存：`708,506 × 32层 × 32头 × k × fp16`，`k=2` 约 **2.90 GB**，`k=5` 约 **7.26 GB**。若严格保存整段 prompt+answer 的同类特征，`k=2` 约 9.9 GB，必须磁盘流式写入。
- Llama 冻结提取一次；卸载 7B 后，小 UHead 可在 8GB 显卡上训练。作者报告其 head 本体约 40MB。
- 它可能超过 0.69 的依据是：当前 Lookback 正在做强汇总，UHead 的受控消融表明“保留头别和局部位置 + 概率”确有独立增量；但本项目数据更小，必须用资料组隔离、早停和固定少量配置压制过拟合。

## 3. RAUQ 递归风险 + 首次错误状态（第三优先，成本最低）

[RAUQ，ICML 2026](https://arxiv.org/abs/2505.20045)（论文页标明 PMLR 306；[官方代码](https://github.com/mbzuai-nlp/rauq-hallucination-detection)）每层选择“生成词元最常关注其前一个词元”的 head，并递归合并当前词元概率、前一词元置信度与该 attention。原方法最后聚合成**整答不确定度**，无监督、单次前向，论文报告额外延迟低于 1%。

[On Early Detection of Hallucinations in Factual QA，KDD 2024](https://www.amazon.science/publications/on-early-detection-of-hallucinations-in-factual-question-answering)证明错误出现前的词元信号已能预测后续错误；[First Hallucination Tokens Are Different from Conditional Ones](https://arxiv.org/abs/2507.20836)（[官方代码](https://github.com/jakobsnl/RAGTruth_Xtended)，预印本、不是顶会结论）直接在 RAGTruth 发现每段第一个幻觉词元比后续条件词元更可检测。

**我们的适配**：不把 RAUQ 立即压成整答分数，保留每层每词元递归置信度；标签由现有 gold span 自动派生为 `安全→错误开始→错误延续` 三状态，用两状态/半马尔可夫头或 CRF 学转移。4-BPE 窗口取“本窗开始概率或上游错误延续概率”。原生 RAUQ 不含这个窗口头，二者必须分开报告。

- 若流式选 head，只存 `708,506 × 32层` float32 约 **90.7 MB**；若先保留全部层头的前一词元 attention，fp16 约 **1.45 GB**。只需一次 Llama replay。
- 本项目最早重叠风险窗召回约 **45.96%**，后续风险窗约 **64.78%**；170 条 gold risk run 中 70 条完全漏掉。这个结构直接优化“先找到开始点，再延续”，比让每个窗口独立学习更贴数据。
- 风险是持续状态会扩大误报范围，所以放行条件应同时看 span any-hit、覆盖率和安全词元误报面积，不能只看窗口 F1。

## 4. Layer-wise Information：资料贡献的跨层轨迹（第四优先）

[Layer-wise Information Deficiency，EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1644/)在“有上下文/空上下文”两次前向中，对每一层 hidden state 过语言模型输出头，计算真实词元 log-prob 的差，再跨层求和。原论文是训练免费、问题/回答级的可回答性检测；截至检索日未在论文页找到作者代码。

**我们的适配**：问题与回答前缀保持不变，只删 3 份资料；不先跨词元平均，给每个回答词元保存 32 层的有资料/无资料 log-prob 差。4-BPE 窗口使用早/中/晚层和、斜率、符号翻转数、负贡献质量。它和已完成 LUMINA 不同：LUMINA 是正常资料对随机资料并按其公式压成少数值；这里是正常资料对空资料并保留有符号的层轨迹。

- 最终 32 维 float32 缓存约 **90.7 MB**，但要两次 Llama 前向，还要对所有 32 层做 LM-head/词表归一化；在 708,506 回答词元上可能很慢。先做 20 答资源与区分度 pilot，不通过就停止。
- 它可能有用，因为现有 NLL 只有最终层、LUMINA 已表明答案级仍有信息，而论文中全层 LI 明显好于单层基线；但原论文目标是“问题是否可回答”，没有证明能定位 4-BPE 错误，优先级低于前三项。

## 5. ICR：注意力路由与 residual 更新是否一致（资源预检项）

[ICR Probe，ACL 2025](https://aclanthology.org/2025.acl-long.880/)（[官方代码](https://github.com/XavierZhang2002/ICR_Probe)）把每层 residual update 投影到上下文词元方向，得到一个“更新信息来自哪些词元”的分布，再与 attention 分布做 Jensen–Shannon divergence；最后将各层 ICR 送入很小的 MLP。它避免直接拿 4096 维静态 hidden state，论文消融中 hidden-update 与 attention 合用优于只用更新方向。

**原生方法**先跨词元池化，输出整答分数。**我们的适配**才是保留每词元 32 层 ICR、在 4-BPE 内池化，并与方案1的来源归因对齐；不能把窗口结果称为原论文复现。

- 只存最终 ICR 是 `708,506 × 32` float32，约 **90.7 MB**；真正成本在每层把 update 与上下文 token 做投影，长上下文近似二次增长。论文全部实验累计 600–800 RTX3090 GPU-hours，不能套用为本项目时长，也说明 3070 上必须先跑 20 答资源样本。
- 它能补充当前最后层 PCA，因为它压缩的是“注意力搬运上下文”与“FFN/参数记忆注入”之间的动态差异。可行性和原生词元级证据都弱于 UHead/RAUQ，若资源 pilot 慢或 OOF 无增量，应直接放弃。

## 统一评测时的无参数 4-BPE 映射

这一步只是把不同粒度的**原生预测**放到同一评测坐标，不训练映射器、不改变模型结构、不重新定义其预测。坐标固定为本项目原回答的 raw Llama BPE：窗口长 4、步长 1；窗口至少含一个 lexical token 才参评。映射只使用回答文字、原生输出及固定 tokenizer offset，不能看 gold span。

- **答级 baseline**：将原生整答风险分数原样赋给该回答的每个有效窗口，即 `s(w)=s(answer)`。若原生方法只给硬标签，就原样广播 0/1。它在窗口表中没有定位能力；窗口 F1 低正是其粒度限制，不能另加选句器补救。
- **claim/句级 baseline**：每个 claim 的原生风险分数赋给其原生归属的回答字符范围；一个窗口取与其 lexical token 相交的所有 claim 分数最大值，无相交则为 0。RefChecker 直接使用官方 `RCClaim.attributed_sent_ids` 和 `RCText` 句子起止坐标，不另做语义对齐；多个 claim 共用一句时取最大值。若某方法不返回、也无法由其官方元数据确定回答范围，则只进答级/claim级原生表，不能臆造跨度后参加窗口主表。
- **跨度级 baseline**：按其原生预测字符跨度与固定 BPE offset 求相交；每个 lexical token 取覆盖它的预测跨度最高分，4-BPE 窗再取其中 lexical token 最大值，无覆盖为 0。二值跨度使用 1/0。边界不扩句、不膨胀、不平滑。

共同窗口表中的整答分数统一取该回答全部有效窗口最大值；答级 baseline 因为全窗同分，所以仍等于它的原生整答分数。连续分数的阈值按现有冻结流程选择并在 test 前锁定，阈值选择不属于映射。原论文指标、原生粒度指标和上述统一窗口指标要分栏报告，防止把确定性投影误写成 baseline 的词元级能力。

## 固定实验顺序与停手条件

1. 先做方案1的 20 答资源 pilot，再在 **fit 内资料组 OOF** 比较：现有模型、仅 E/N/C、仅反事实归因、两者交叉。只有交叉项对 OOF AP/AUROC 或固定规则 F1 有稳定增量才跑全量。
2. 并行做方案3，因为只需一次 replay；先报告原生 RAUQ 整答 baseline，再报告我们的 onset/window 适配。
3. 再做 UHead `k=2` 单一配置。UHead baseline 保持结构不变；加入方案1后另命名 ours。
4. LI、ICR 都先过 20 答资源与信号预检；不因论文成绩高就直接做全量。
5. cal159 已被反复使用，只能继续称开发结果；最终结论必须一次性打开封存 test150。论文指标、原生 baseline、共同 4-BPE 映射和 ours 融合成绩分表呈现。
