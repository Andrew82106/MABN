# SIRG 正式基线可复现性审计

审计日期：2026-09-13  
论文：*Detecting Hallucinations in Retrieval-Augmented Generation via Semantic-level Internal Reasoning Graph*，Findings of ACL 2026  
审计对象：作者公开代码提交 `0e5e310891f16d81b8f91c874125bdcc8000c259`

## 结论

| 问题 | 结论 |
|---|---|
| 能否作为正式基线 | **可以，但目前只能列为“待运行的 SIRG 官方代码结构迁移”** |
| 能否复现论文原表数字 | **不能**。作者未公开最终微调权重、原始数据、LRP 中间结果和预测输出；论文与代码还有多处不一致 |
| 能否在本项目统一协议下迁移 | **有条件可行**。需要完整 BF16 Llama-2-7B、LXT 2.0 和约 35–50 A100 GPU 小时的执行预算 |
| 当前正式成绩 | **N/A**。本次只做 CPU 审计，没有启动 GPU，也没有产生分数 |
| 现有 `token_source_attribution_v4` 是否是 SIRG | **不是**。它是自研的来源归因窗口分类器，不能命名为 SIRG、SIRG 实现或 SIRG 分数 |

SIRG 与本项目场景高度相关：它检测的正是回答是否依赖给定资料的“忠实性幻觉”，并原生输出语义片段级风险。因此，它比只有整答分数的基线更适合检验 4-BPE 局部定位。不过，只有完整走通作者的 **AttenLRP → 语义图 → Top-15 线性化 → 微调 AlignScore** 链条，才能进入正式基线表。

## 已核实的官方来源

- [ACL Anthology 论文页](https://aclanthology.org/2026.findings-acl.1385/)与 DOI `10.18653/v1/2026.findings-acl.1385`确认其为 Findings of ACL 2026，页码 27826–27841。
- ACL 页面仍指向已经失效的匿名仓库 `anonymous.4open.science/r/SIRG-1022`；最终 PDF 本身给出了现已公开的[作者仓库](https://github.com/hk-hp/SIRG/tree/main)。
- 作者仓库当前只有 `main`，没有 tag 或 release。本审计冻结提交为 [`0e5e310`](https://github.com/hk-hp/SIRG/commit/0e5e310891f16d81b8f91c874125bdcc8000c259)。关键文件的哈希见 [`official_source_hashes.json`](./official_source_hashes.json)。
- [arXiv 版本](https://arxiv.org/abs/2601.03052)为 2026-01-06 提交的 v1。
- LRP 依赖可定位到作者指定的 [LXT 2.0](https://github.com/rachtibat/LRP-eXplains-Transformers/releases/tag/v2.0)。下游初始模型可定位到官方 [AlignScore](https://github.com/yuh-zha/AlignScore)，其 `AlignScore-base.ckpt` 下载仍有效；它只是预训练起点，不是 SIRG 微调后的权重。
- README 中的 NJUBox 下载地址目前返回“外链不存在”，所以论文实验使用的数据、LRP 文件、baseline 输出和最终模型均无法取得。

没有找到新的作者仓库或作者镜像。现存可验证实现就是上述 GitHub 仓库。

## 必须冻结的原方法

### 1. 检测对象与粒度

SIRG 检测的是 **RAG 忠实性幻觉**：回答中的语义片段与给定上下文不一致或缺乏上下文支持。它不是开放世界真实性检测器，也不能仅凭自身世界知识断言事件真假。

论文在 RAGTruth 上使用 Llama-2-7B-chat 和 Llama-2-13B-chat；在 Dolly closed-QA 上使用 Qwen2.5-3B 和 Qwen2.5-7B。原生最细输出是回答的**语义片段**，不是 token。

### 2. AttenLRP 归因

作者代码通过 LXT/AttenLRP 修改 Llama，使用完整 BF16 权重、`model.train()` 和梯度检查点。对已经给定的回答做 teacher forcing：每一步以实际回答 token 的温度缩放 logit 为目标反向传播，最多处理 500 个回答 token。该过程产生目标 token 对此前 prompt 与回答 token 的 LRP 相关度。

正式迁移必须保留：

- LXT 2.0 的 AttenLRP 规则；
- 完整 BF16 Llama-2-7B-chat，不用 NF4、普通注意力或隐藏状态替代；
- 每步目标、温度字段和 `max_steps=500`；
- 作者代码中的 BOS/推理头处理及相关度绝对值聚合。

### 3. 语义片段和实体

代码先按换行，再按 spaCy 句界切分 prompt 与回答；长度少于 5 个字符的片段不进入图。它从回答中抽取名词、实体、动词、名词短语等“实义内容”，将这些词映射到回答 token。

这里必须按冻结代码运行，因为代码与论文文字并不完全一致：

- `is_passage=False`，所以 prompt 也被拆成句子，问题和指令也会形成图节点；
- 实际并集只使用 `noun_spacy + entity_spacy + verb_stan + noun_stan + negations`；
- `get_entity.py` 把 `noun_spacy` 字段错误写成了 `noun_stan`，导致 Stanza 名词重复而 spaCy noun chunks 丢失；
- 目标片段只在回答的实义 token 上求均值；源片段实际上在该句的**全部 token**上取最大值，与论文声称只用源片段实义 token 不一致。

正式 SIRG 基线应保留这些可运行的代码语义。修正后的“论文意图版”只能另列敏感性分析或本文适配，不能替代正式基线。

### 4. 语义图与 Top-15 线性化

对每个回答片段，作者代码对选中的目标 token 的绝对 LRP 向量求均值，再对每个先前语义片段覆盖的 token 取最大值。代码随后执行 `node / sum(node[1:])`，去掉推理头位置，取贡献最大的 15 个先前片段。

源节点按如下文字形式输入分类器：

```text
Context 1: "<source fragment>"; Contribution Score 1: <weight rounded to 2 decimals>
...
```

先前生成的回答片段加前缀 `Previous Generate:`，目标回答片段作为 claim。Top-15、两位小数、文字模板、排序与截断都属于方法特征，不能调。

### 5. AlignScore 判别器与训练

作者以 `AlignScore-base.ckpt` 为起点，使用 RoBERTa-base 的 pooled output、dropout 0.1 和二分类线性头。类别 0 为无幻觉，类别 1 为幻觉；损失是带权交叉熵，权重固定 `[0.1, 0.9]`。

冻结代码中的训练参数是：

| 项 | 冻结值 |
|---|---|
| batch size | 16；验证 batch 16 |
| 最大长度 | 512 |
| 截断 | `only_first`，即先截 source/context 侧 |
| seed | 2022 |
| optimizer | AdamW |
| learning rate | `1e-5` |
| epsilon | `1e-6` |
| weight decay | `0.1` |
| warmup | 6% 线性 warmup/decay |
| epoch | 100 |
| precision | FP32 |
| checkpoint | 每 epoch 以验证片段 F1 选最好模型；判定阈值 `p(risk)>0.5` |
| MLM | 关闭 |

论文写“Adam、100 iterations”，而代码明确是 AdamW、100 epochs。正式迁移以冻结代码为准，并在论文中披露该差异。

### 6. 原生输出

二分类 softmax 的 `p(class 1)`就是连续片段风险；`>0.5`得到原生硬片段标签。整答原生规则是“预测为幻觉的片段数 / 片段总数”，超过 α 则整答为幻觉。

作者 `graph_test.py`枚举 α 为 `[-1, 0, 0.1, …, 1.1]`，却没有冻结一个最终 α。更关键的是，代码把整答**正确**记为正类 1，因此论文表中的整答 F1 是“正确回答为正类”的 F1，不能直接与本项目“幻觉风险为正类”的 F1 比较。

## 论文数字及其限制

论文报告：

| 数据/生成模型 | Precision | Recall | F1 |
|---|---:|---:|---:|
| RAGTruth / Llama-2-7B-chat | 73.64 | 79.83 | 76.61 |
| RAGTruth / Llama-2-13B-chat | 78.48 | 85.51 | 81.84 |
| Dolly / Qwen2.5-3B | 72.10 | 95.49 | 82.17 |
| Dolly / Qwen2.5-7B | 84.21 | 96.00 | 89.71 |

这些数字不能作为本项目基线成绩，因为数据、正类方向、整答聚合和阈值协议均不同。论文正文还把 Qwen2.5-7B 的 F1 写成 89.17，与表格 89.71 冲突。RAGTruth Llama-13B 主表 F1 为 81.84，而 Top-k 消融表的 Top-15 F1 为 87.21；仓库默认又是 Top-15。缺少作者 checkpoint 和输出后，无法确定这些结果具体对应哪组设置。

## 本项目允许的统一接口

本项目继续决定输入任务、共享样本、材料组划分、4-BPE 金标、整答聚合、阈值与指标。SIRG 只负责从冻结原方法输出连续片段风险。

1. **训练标签接口**：回答片段只要与任一正类 4-BPE 金标窗口有字符重叠，片段标签即为风险 1，否则为 0。该规则预先固定，只在 fit 内产生 SIRG 所需的片段监督，不调整模型结构或选取“更有利”的位置。
2. **片段到窗口**：把片段风险原值赋给其精确字符跨度；每个项目 4-BPE 窗口取与其重叠片段风险的最大值。没有被 SIRG 片段覆盖的窗口固定为 0。
3. **窗口到整答**：整答分数为全部合格 4-BPE 窗口分数的最大值。
4. **阈值与指标**：窗口和整答各自在 159 条 cal 上按 `F1 → precision → 较高阈值`独立选阈值，报告风险为正类的 F1、AUROC、AP；正式 test 继续封存。

这一层不能包含校准器、平滑、NLI 替换、特征融合、窗口逻辑回归或其他可学习映射。作者原生 `p>0.5` 与比例/α 只作诊断，不取代统一主比较。

## 数据迁移边界

共同数据为 3,680 个 fit 回答和 159 个 cal 回答。SIRG 的模型和训练方法保持不变，但训练、验证和评分必须服从项目的数据隔离：

- cal 不参与梯度、特征选择或 checkpoint 选择；
- fit 内按材料组做确定性的 80/20 内部 train/validation；同一材料组不得跨组；
- 100 epoch 期间仍按作者原生验证片段 F1、阈值 0.5 选 checkpoint；
- cal 只用于上述统一阈值与开发评测。

共享回答来自多个生成器，而闭源生成器无法提供原始白盒轨迹。主比较只能把全部 3,839 个固定回答 teacher-force 到同一个完整 BF16 Llama-2-7B-chat 上取得 SIRG 信号。因此结果应准确命名为：

> **SIRG 官方代码结构迁移（Llama-2-7B teacher-forced common replay）**

它检验同一检测管线在共同回答上的效果，但不能声称恢复了 GPT-3.5、GPT-4、Mistral 或 Llama-70B 生成当时的内部推理。原生 Llama-2-7B 生成器子集可另列诊断，不能取代主比较。

## 当前无法逐值复现的原因

1. 作者 NJUBox 链接失效，SIRG 微调 checkpoint、数据、LRP、图文件、预测均未公开。
2. 代码没有顶层环境锁；LXT、PyTorch、Transformers、spaCy `en_core_web_lg`、Stanza、Lightning 等版本未完整固定。
3. Llama 和 RoBERTa 的 Hugging Face revision 未给出。
4. 路径和 CUDA 设备大量硬编码；README 命令 `train.pyrun` 是笔误。
5. README 所述 LRP 输出目录与代码实际输出位置不一致。
6. README 写 Llama 流程，但 `get_entity.py` 主入口调用 Dolly；`get_score.py` 默认 Qwen，Llama 分支又可能继承全局 Qwen `model_id`。
7. 训练数据文件名包含 `top15`，测试代码读取的文件名却不含 `top15`；测试还硬编码 Dolly Qwen-7B checkpoint。
8. README 指向的 `baseline/`目录不存在。
9. 仓库根目录没有 SIRG 自身许可证；只有内嵌 AlignScore 的 MIT 许可证。
10. 35 个 Python 文件中，六个主流程入口能通过 AST 解析；未使用的 `core/AlignScore-main/benchmark.py:225`存在缩进错误，另有一处无效转义警告。

这些问题允许通过路径、设备和调用封装来修复，但不能改变张量、文本模板、训练损失或读出语义。每一处执行修复都要记录补丁和前后哈希。

## 资源估算

CPU 统计显示：

- 3,839 个回答；平均总序列约 627 BPE，最长 1,232；
- 回答部分平均约 185 BPE；18 个回答超过作者 `max_steps=500`，合计约 203 个尾部 token 无 LRP；
- 预计相关度矩阵约 4.86 亿个单元，原始 float32 约 1.81 GiB，JSON 很可能超过 4.5 GiB；
- 按论文 A100 单样本耗时，已有回答无需重新生成时，LRP 约 27.2 小时，完整预处理与推理约 29.5 小时；若连生成阶段计算则约 42.4 小时；
- AlignScore 微调时间论文未报告，因此建议总预算按 **35–50 A100 GPU 小时**预留，这只是工程预算，不是实测承诺；
- 完整 BF16 Llama-2-7B 权重约 12.55 GiB，当前 8 GiB RTX 3070 无法忠实运行。建议 40 GiB A100；24 GiB 设备只能先做小批试运行；
- 建议额外预留 25–40 GiB 磁盘。

本地已有 `NousResearch/Llama-2-7b-chat-hf` revision `351844e75ed0bcbbe3f10671b3c808d2b83894ee`的完整 safetensors，可作为正式候选资产；不得量化。18 个超长回答仍严格保留 `max_steps=500`，未覆盖尾部窗口按统一适配器记 0，不能为了提高分数而延长。

## `token_source_attribution_v4` 的身份边界

| 组成 | 当前 v4 | SIRG |
|---|---|---|
| 白盒骨干 | NF4 Llama 重放 | 完整 BF16 Llama + LXT/AttenLRP |
| 内部信号 | attention(q,k) × `||V_head(k)||₂` | 对实际目标 token 反传的 LRP 相关度 |
| 基本单位 | 每个词法 BPE、4-BPE 窗口 | 语义片段与实义 token |
| 特征 | source/previous/other 区域份额、ModernBERT NLI、NLL | Top-15 语义图来源片段及贡献权重文本 |
| 学习器 | 4-BPE 窗口逻辑回归 | 微调 AlignScore/RoBERTa 二分类器 |
| 原生输出 | 窗口风险 | 语义片段风险，再按片段比例判整答 |

v4 源码哈希：

- `run_token_source_attribution_v4.py`: `2dcbd70831ef34f5dea316653b610175f842a5053048ebc832f54808fcc73db6`
- `score_token_source_attribution_v4.py`: `c3d39d1a3c8e712efde81cda2a4ac36431121671a9bb74ace767a2c795fe4514`

所以 v4 最多可描述为“受来源归因思想启发的自研方法”或“SIRG-adjacent”。它不能占用 SIRG 的正式基线行，也不能把自身分数写成 SIRG 结果。

## 最终建议

SIRG 应进入正式基线候选清单，状态保持 **N/A / 待运行**。执行时严格采用 [`FROZEN_MIGRATION_PLAN.md`](./FROZEN_MIGRATION_PLAN.md)。只有源资产、CPU 接口、短样本 LRP 和 AlignScore 训练四道门全部通过后，才投入全量 GPU；结果无论高低都原样报告。若无法取得足够显存或依赖无法稳定重建，则论文中应把 SIRG 列为“高度相关但不可执行的公开实现”，不能用 v4 补分。
