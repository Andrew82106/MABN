# R20 外部数据候选核查

核查日期：2026-09-11。范围仅为官方论文、仓库代码、数据卡 schema 与汇总统计；未下载语料或模型、未运行生成、未打开未用 QA 的具体回答或 test 标签文件，未按标签筛题。以下数字表示官方资料描述的规模，不表示本项目尚未接触且独立的可用规模。

**最贴合“question + retrieved contexts + model answer + 人工错误 span”的候选是 RAGTruth 的 QA 子集。** 原生 Mistral 白盒回放仍有版本与输入还原缺口，见 [回放核查](MISTRAL_REPLAY_FEASIBILITY.md)。

| 数据集/子集 | 规模与任务 | 标签粒度与来源 | 许可 | 适配与局限 |
|---|---|---|---|---|
| RAGTruth QA | 989 个 source；6 个模型共 5,934 个回答、2,927 个错误 span。问题来自 MS MARCO train，每题保留 3 个检索段落。[论文 §3.2、表2](https://aclanthology.org/2024.acl-long.585.pdf) | 人工词/span 标注；字符起止位置、类型。每回答两人独立标注，重大分歧交第三人复核。[论文 §3.3](https://aclanthology.org/2024.acl-long.585.pdf)、[schema](https://github.com/ParticleMedia/RAGTruth#dataset) | 官方仓库 [MIT](https://github.com/ParticleMedia/RAGTruth/blob/main/LICENSE)；底层 MS MARCO 材料另有来源条款。 | 最合适的人工定位候选；989 个 source 不能算成 5,934 个独立问题。原回答来自 GPT、Llama、Mistral，未包含 Qwen。 |
| RAGTruth Summary | 943 个 source、5,658 个回答；CNN/DM 628 个 source，近期新闻 315 个。[论文表2](https://aclanthology.org/2024.acl-long.585.pdf) | 同上人工 span。 | 同仓库 MIT；新闻原文权利不能仅由仓库许可推定。 | 是文章摘要，不是用户问题与检索上下文的 QA。扩 QA 时应明确分开。 |
| RAGBench | 论文称约 10 万，来自 12 个子集；当前官方数据卡 `num_examples` 相加为 **95,381 行**，非独立问题去重数。[论文 §3](https://arxiv.org/html/2407.11005v1)、[数据卡](https://huggingface.co/datasets/galileo-ai/ragbench/blob/main/README.md) | GPT-4-0125-preview 自动标注回答句子是否有支持，并标上下文相关/被使用片段；有 question/documents/response。不是全量人工错误 span。[论文 §3.3、§5.2](https://arxiv.org/html/2407.11005v1) | 数据卡 [CC BY 4.0](https://huggingface.co/datasets/galileo-ai/ragbench/blob/main/README.md)。 | 可扩输入与句级弱标签研究；不能把整句广播为词标签后称人工定位金标。MS MARCO 等来源仍需交叉去重。 |
| HalluRAG | **19,731 个有效标注句子**，不能写成 19,731 个独立问答。GPT-4o 基于 Wikipedia 造题和可答/不可答上下文，Llama2/Mistral 回答。[论文](https://arxiv.org/html/2412.17056v1) | GPT-4o 句级自动标注，仅 274 个人工标注句子用于核验标注器；保存回答及内部状态。[论文](https://arxiv.org/html/2412.17056v1)、[仓库](https://github.com/F4biian/HalluRAG) | 数据许可未确认：仓库未见 LICENSE；[官方数据 DOI](https://doi.org/10.17879/84958668505) 的机构页本次未能访问。论文许可不等于数据许可。 | 适合句级白盒方法参考；不满足“真实用户问题 + 全量人工 span”要求。 |
| RAGtelligence | 本次按精确名称及 arXiv/GitHub/Hugging Face/ACL 限定检索，未找到可核实的官方论文、仓库或数据卡。 | 未确认。 | 未确认。 | 暂不能列为已验证数据集；不据此断言全球不存在，也不与 RAGognize、RAISE 等名称混同。 |

RAGTruth 全部三类任务合计 2,965 个 source、17,790 个回答；Data2txt 为 1,033 个 source、6,198 个回答。论文划分每任务 150 个 source 为 test，因此 QA 纸面规模为 train 839 / test 150；实际发布索引及本地暴露情况由独立元数据审计确认。[论文表2、§5.2](https://aclanthology.org/2024.acl-long.585.pdf)

RAGBench 的 95,381 是仅求和官方 README YAML 中 12 个配置的 train/test/validation `num_examples` 得到：1,765 + 2,550 + 1,826 + 1,318 + 2,027 + 16,562 + 4,532 + 2,697 + 2,690 + 24,500 + 33,104 + 1,810；未读取任何数据行。[官方数据卡](https://huggingface.co/datasets/galileo-ai/ragbench/blob/main/README.md)

规划边界：先按 source_id 将同题的多模型回答放在同一分区，再审计旧来源、同文和同事件重叠。RAGTruth 的 `implicit_true` 是“可能世界真实但上下文没说”，应先固定它是否计作错误，不能看到新 test 表现后改口径。[字段定义](https://github.com/ParticleMedia/RAGTruth#dataset)

**别的 LLM 的原回答与人工 span 可以评价该回答的上下文支持情况；把这个回答强制输入 Qwen，只能研究 Qwen 对外部文本的评分/表征，不能称为 Qwen 自己生成回答的白盒错误标签。** 若要测 Qwen 自己生成的错误，需要独立生成并标注其实际回答；源模型的 span 不能直接搬过去。本轮只做候选与新 final test 规划。
