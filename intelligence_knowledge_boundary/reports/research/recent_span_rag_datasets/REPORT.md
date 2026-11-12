# 细粒度 RAG 幻觉数据集候选（截至 2026-09）

> 调研范围：只查论文、官方仓库和官方数据卡的元数据；未下载数据，也未查看任何候选测试样本或测试标签。

## 阶段结论

当前最实用的组合是：

1. **训练**：优先用 **RAGognize train**；再用 **RAGBench train** 补充大量句级弱标签，用 **LettuceDetect v2 train** 补充精确 span 标签。
2. **独立外测**：冻结模型和阈值后，依次测 **FaithBench、DelucionQA test、HDM-Bench**。
3. **标签审计**：用 **RAGTruth-Enhance / RAGTruth++** 复测当前 RAGTruth 结果。它们不是新样本，不能与原 RAGTruth 混用训练。

最关键的限制是：**白盒激活与生成模型绑定**。只有数据中的回答确实由同一开源模型生成，才能恢复有意义的内部激活和生成时 log probability。把闭源模型的现成回答强行送入当前模型，只能得到“教师强制”状态，不能当作原始生成状态。因此，RAGBench、DelucionQA、FaithBench 等更适合训练证据/文本分支或做外部测试；白盒主训练仍应以 RAGognize 中可复现的开源生成器，或我们自己用目标模型重新生成并标注的数据为主。

## 最相关的 8 个候选

| 数据集与出处 | 规模与标签粒度 | 是否真实 RAG；回答生成器 | 许可与下载 | 对当前 4-BPE 检测的用途 |
|---|---|---|---|---|
| [RAGognize](https://arxiv.org/abs/2604.15945)（arXiv 2026）/ [官方仓库](https://github.com/F4biian/RAGognizer) / [数据](https://huggingface.co/datasets/F4biian/RAGognize) | 4,623 个问题，1,842 train / 2,781 test；每题 4 个回答，约 18,492 个回答；精确字符/词元 span | **是**；Llama-2-7B-chat、Llama-3.1-8B-Instruct、Mistral-7B-Instruct v0.1/v0.3 | 数据 CC-BY-SA-4.0，代码 MIT；公开 | **最适合直接训练白盒探针**，前提是目标模型为这四个检查点之一；测试集只在最终冻结后使用 |
| [RAGBench](https://arxiv.org/abs/2407.11005)（arXiv 2024）/ [数据](https://huggingface.co/datasets/galileo-ai/ragbench) | 约 10 万条、12 个子集、5 个领域；约 78k train / 12k dev / 11k test；逐句“有/无资料支持” | **是，属于给定检索资料后生成**；GPT-3.5-0125、Claude 3 Haiku，部分沿用 HAGRID/ExpertQA 输出；GPT-4 自动标注 | CC-BY-4.0；公开 | **可直接作辅助训练**。句标签可铺到句内 4-BPE 窗口，但属于弱标签；闭源生成器使其不适合直接训练原生成状态探针 |
| [LettuceDetect v2 / Beyond Document Grounding](https://arxiv.org/abs/2607.00895)（arXiv 2026）/ [官方仓库](https://github.com/KRLabsOrg/LettuceDetect) / [数据](https://huggingface.co/datasets/KRLabsOrg/lettucedetect-code-hallucination) | 新构造 74,285 条，66,368 / 2,816 / 5,101；精确字符 span；含 ACL 论文、README、Wikipedia、工具输出和代码 | **RAG 格式，但错误由局部替换合成**；非代码由 Qwen 3.6 35B A3B 生成/注入，代码由 Gemma 4 31B 注入 | 数据 CC-BY-4.0，代码/模型 MIT；公开 | **可作精确 span 辅助训练**，尤其训练证据分支；须单独报告“合成→自然幻觉”迁移，防止学到注入痕迹 |
| [DelucionQA](https://aclanthology.org/2023.findings-emnlp.59/)（Findings of EMNLP 2023）/ [官方仓库](https://github.com/boschresearch/DelucionQA) | 913 个问题、2,038 个问答-资料组合；1,151 train / 216 dev / 671 test；人工逐句标为支持、冲突或无法验证 | **是，领域文档检索问答**；ChatGPT 生成回答 | 数据重建文件 CC-BY-4.0，代码 AGPL-3.0；公开，但正文需按偏移从 Jeep 手册站点重建 | **可训练句级辅助头，也适合人工标签外测**；原生成器闭源，不能还原生成时内部状态 |
| [ANAH v1](https://aclanthology.org/2024.acl-long.442/)（ACL 2024 主会）/ [官方仓库](https://github.com/open-compass/ANAH) | 约 4.3k 回答、12k 句级标注、700+ 主题，中英双语；每句附证据片段、错误类型和修正 | **部分符合**；高质量样本由 GPT-3.5 看资料生成，低质量样本由 InternLM-7B 不看资料生成 | Apache-2.0；公开 | **可作中英句级辅助训练/压力测试**；生成器与是否看资料高度相关，容易产生捷径，不能作为主结果 |
| [FaithBench](https://aclanthology.org/2025.naacl-short.38/)（NAACL 2025 Short）/ [官方仓库](https://github.com/vectara/FaithBench) | 750 个 source-summary 对；专家人工精确字符 span，并区分一致、可疑、内在/外在错误等 | **不是在线检索**，是给定资料的摘要；10 种模型，包括 GPT-4o、Claude-3.5、Gemini、Llama-3.1、Qwen、Mistral 等 | CC-BY-NC-SA-4.0；公开 | **只建议作独立外部 span 测试**。它刻意收集难例，规模小且分布偏难，不应用来调阈值或训练 |
| [HDM-Bench](https://arxiv.org/abs/2504.07069)（arXiv 2025）/ [官方仓库](https://github.com/aimonlabs/hallucination-detection-model) / [数据](https://huggingface.co/datasets/AimonLabs/HDM-Bench) | 1,000 条；句级和短语级三分类，并给字符 span；部分人工复核 | **混合 RAG 资料，但回答/错误受控合成**；Mistral-7B、Qwen2.5-7B、Mixtral、Nous-Hermes 等 | CC-BY-NC-SA-4.0；HF 需申请访问 | 官方明确定位为 **test-only**，只做冻结后的外测；不能训练、选特征或调阈值 |
| [RAGTruth-Enhance](https://arxiv.org/abs/2603.27752)（arXiv 2026）/ [公开工件](https://zenodo.org/records/19249142)；[RAGTruth++](https://huggingface.co/datasets/blue-guardrails/ragtruth-plus-plus)（重标资源） | Enhance 重审 2,675 个原 RAGTruth 测试回答，发现约 1.68 倍幻觉回答、3.1 倍 span；++ 重标 408 条，span 从 86 增至 865 | **继承 RAGTruth 的真实 RAG 样本与生成器** | 可公开获取；工件数据许可未在当前官方元数据中明确，使用前需核对 | **只做 RAGTruth 标签审计**。它们与原测试集重叠，绝不能加入训练；若分数明显下降，说明当前高分部分来自漏标 |

## 训练与测试边界

| 类别 | 可用数据 | 具体用法 |
|---|---|---|
| 白盒主训练 | RAGognize train；或自行用目标模型生成的数据 | 提取真实生成时的激活、log probability、注意力等信号，训练 4-BPE 窗口头 |
| 辅助训练 | RAGBench train、LettuceDetect v2 train、DelucionQA train/dev、ANAH v1 | 训练文本/证据一致性分支；句级标签只作为弱监督，合成 span 需降权并做来源消融 |
| 独立外测 | FaithBench、DelucionQA test、HDM-Bench | 所有结构、阈值和后处理冻结后一次性评测；不查看测试标签、不据其调参 |
| 标签审计 | RAGTruth-Enhance、RAGTruth++ | 只复算现有模型，量化原 RAGTruth 漏标对回答级和窗口级 F1 的影响 |

## 标签统一到 4-BPE 窗口

- **字符/span 标签**：先按目标模型 tokenizer 映射到 BPE；窗口与金标 span 有交集即标为可疑，同时另报字符重叠 span-F1，避免窗口规则虚高。
- **句级标签**：把该句覆盖的所有窗口赋同一标签，仅作为弱监督；不能把这种结果称为精确词元定位。
- 数据切分按**问题、来源文档和主题**去重，不能只按回答随机切分，否则同一证据会同时出现在训练和测试中。

## 未列入主候选的原因

- **HalluRAG、Mu-SHROOM**：已在当前方案中；前者贴近句级 RAG，后者没有“模型生成时使用了检索资料”的条件，不再重复列为新候选。
- **PsiloQA、EnokiQA、FAVAbench**：都有细粒度标签，但回答生成时未看到后置核验资料，更适合一般事实核验，不适合研究“是否使用了已检索资料”。
- **ANAH-v2**：论文规模很大，但官方仓库目前没有发布完整带标签回答。
- **NewsSum（ACL 2026）**：场景最接近多篇新闻汇总，论文含人工幻觉 span；官方仓库未公开该数据集，当前不能用。
- **HALoGEN**：公开的是原子事实单元，未稳定对齐到原回答字符 span，不能直接转成 4-BPE 金标。

## 对当前实验的直接建议

先不要简单把所有公开数据混在一起。第一轮做三组消融：`RAGognize` 主训练；`+ RAGBench` 句级弱监督；`+ LettuceDetect v2` 精确合成 span。模型和阈值确定后，再盲测 FaithBench、DelucionQA test 和 HDM-Bench，并在 RAGTruth-Enhance/++ 上复核原有高分。这样能分别回答三件事：是否学到目标模型的内部风险信号、是否受益于更多证据监督、以及能否迁移到人工标注的自然错误。
