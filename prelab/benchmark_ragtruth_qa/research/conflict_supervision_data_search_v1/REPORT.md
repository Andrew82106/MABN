# RAG 事实冲突监督数据：正式调研结论

## 结论

**第一优先使用本地已经准备并隔离的 `auxiliary_human_v1`。** 它与正式 QA 基准同属 RAGTruth，标签定义、资料条件和人工标注方式完全一致；现有 9,678 个辅助回答中有 4,381 个精确字符级冲突 span，远比另找一个只有整条 claim 标签的数据集更适合补当前冲突专头。

外部数据中，AVeriTeC 最贴近开放网络情报场景，VitaminC 最适合学习“文字几乎相同、关系却反转”的证据敏感性，FEVER 可补大规模人工核验关系对。但三者都**没有错误 claim 内的字符或 token 金标**，只能加入 claim 级关系辅助损失；不得把整条错误 claim 的所有 token 标成风险 token。

## 第一优先：本地 RAGTruth 非 QA 人工 span

正式出处是 ACL 2024 主会长文 [RAGTruth](https://aclanthology.org/2024.acl-long.585/)，[官方仓库](https://github.com/ParticleMedia/RAGTruth) 和本地原始许可均为 MIT。它不是重新生成的银标：回答由多种模型自然生成，错误位置与类型由人工标注。

### 已有文件

- 候选记录：`prelab/benchmark_ragtruth_qa/auxiliary_human_v1/candidate_fit.jsonl`
- 固定统计及源文件哈希：`prelab/benchmark_ragtruth_qa/auxiliary_human_v1/manifest.json`
- 被隔离来源：`prelab/benchmark_ragtruth_qa/auxiliary_human_v1/quarantined_source_index.jsonl`
- 材料重叠边：`prelab/benchmark_ragtruth_qa/auxiliary_human_v1/overlap_edge_index.jsonl`
- 质量排除记录：`prelab/benchmark_ragtruth_qa/auxiliary_human_v1/quality_excluded_index.jsonl`
- 固定准备脚本：`prelab/benchmark_ragtruth_qa/src/prepare_auxiliary_human.py`
- 原始许可：`prelab/data/raw/LICENSE`

### 实物统计

| 项目 | 数量 |
|---|---:|
| 回答 / 来源 / 来源连通组 | 9,678 / 1,614 / 1,558 |
| 无任何错误标签的回答 | 4,636 |
| 含冲突的回答 / 来源组 | 2,980 / 1,222 |
| Evident Conflict | 4,252 span |
| Subtle Conflict | 129 span |
| 冲突总计 | **4,381 span** |
| Summary / Data2txt 冲突 span | 643 / 3,738 |
| 只含冲突 / 同时含其他风险的冲突回答 | 1,514 / 1,466 |
| 带非空人工说明的冲突 span | 4,376 / 4,381 |
| 人工说明含 `Original:` 证据或修正的 span | 4,161 / 4,381 |
| 原 span 坐标错误 | 0 |

人工说明往往直接写出证据与错误关系，例如资料是 `WiFi: no`，回答却写 `free WiFi`；或者资料说明强制限水导致用水下降，回答却把因果主体写成干旱本身。这正是当前高文本相似度冲突漏检所需的监督。

### 已完成的重叠与划分保护

- 准备时先封锁正式 QA 非 fit 与官方 test 的来源身份，再处理辅助回答；官方 test 回答与标签没有被解析。
- 以资料的非空精确哈希、连续 20 个归一化 Unicode 词重叠，以及 Data2txt 的商家名称、地址、城市、州信息检查来源关系。
- 62 个触及封锁来源的训练来源已经整体隔离；保留记录的 `group_id` 已把共享材料连接成不可拆分组。
- 该检查能防止已知正文复用和明确商家复用，但不能证明排除了所有改写或仅实体相同的情况。报告中应保留这一限制。

## 机械转换为当前训练接口

转换只产生我们模型的辅助训练数据；正式 QA calibration/test 和论文 baseline 均不改。

### 1. 输入

每条训练微主张保持统一结构：

```text
question + [Passage 1..3] + atomic claim
```

- `question` 逐字复用发布记录中的原问题或任务指令。
- Summary 的单篇文章先确定性分句，以当前 claim 做无标签 BM25 排序，取前三个证据块作为 Passage 1–3；每块保留原文坐标与哈希。
- Data2txt 把结构化资料确定性拆成基础字段、属性/营业时间、顾客评论等事实块，再按同一无标签排序取前三块。
- 不把人工 `meta`、修正说明、错误类型、另一版本回答或标签放进模型输入。
- 这批回答没有测试对象模型的原生生成轨迹。若需要当前 tokenizer 的坐标，可重新分词并明确标为 `retokenized auxiliary`；不得把它声称为原生成时的 logprob/hidden state。

### 2. claim 标签

复用现有确定性原子切分。每个原子 claim 保留其在完整回答中的 `[char_start, char_end)`：

- `conflict_claim=1`：该 claim 至少覆盖一个由 Evident/Subtle Conflict span 映射出的 lexical risk BPE。
- `baseless_claim` 单独保留。只与 Baseless span 相交的 claim 不能称“安全”，也不能并入冲突正类。
- 没有任何人工 span 的 claim 才是最干净的总风险负例；可优先从 4,636 个完全无标签回答中采样。

### 3. token 与 4-BPE 窗口标签

严格复用当前 `ANNOTATION_PROTOCOL.md`：

1. 用固定 tokenizer 给原回答建立字符 offset。
2. lexical BPE 覆盖的字母数字字符只要与任一冲突 span 相交，`conflict_risk_mask=1`。
3. raw BPE 长度 4、步长 1；窗口含任一冲突 lexical risk BPE 时 `conflict_window=1`。
4. 只有标点/空白的 span 交集不产生风险 token；所有边界异常必须单列，不能移动标签。

这可以产生真实的 token/window 冲突监督，因为原数据有人工字符 span。它与下面只有 claim 标签的外部资源不同。

### 4. answer 标签与固定分组

- 辅助冲突标签：回答含任一 Evident/Subtle Conflict span 时 `conflict_answer=1`。
- 正式总风险评测仍按当前规则：任一原人工风险 span 即 `answer_risk=1`。辅助冲突标签不能重写正式总风险标签。
- `group_id` 是最小不可拆分单位；所有同组回答、微主张和窗口进入同一个 GroupKFold fold。
- 训练权重建议沿用“等组质量 → 等回答质量 → 等微主张质量”，然后只在训练损失中平衡冲突二类，避免 Data2txt 或多回答来源支配梯度。

### 5. 建议的训练接口

最稳妥的第一版是只给现有语义关系编码器增加一个 `conflict` 辅助头：

```text
L = L_main_QA_risk + lambda_conflict * L_aux_conflict
```

- 先用辅助集学习证据—claim 的冲突关系，再只用正式 QA fit 做最终微调和阈值选择。
- Baseless 保留为第二个标签或 ignore mask，避免把“资料没说”和“资料明确说反”混成一类。
- 同一来源组的全部样本在同一 fold；`lambda_conflict`、采样比和早停只由 fit 的组外预测决定。
- 最终仍在当前共同协议下报告 4-BPE 窗口 F1 与整答 F1。

## 外部候选

### AVeriTeC：最贴开放网络情报，但只有 claim 级标签

- 级别与出处：[NeurIPS 2023 Datasets and Benchmarks](https://proceedings.neurips.cc/paper_files/paper/2023/file/cd86a30526cd1aff61d6f89f107634e4-Paper-Datasets_and_Benchmarks.pdf)
- 官方数据与代码：[MichSchli/AVeriTeC](https://github.com/MichSchli/AVeriTeC)
- 许可：官方 README 明示 [CC BY-NC 4.0](https://github.com/MichSchli/AVeriTeC#license)
- 质量：4,568 个真实网络声明，由多轮人工标注得到 verdict、核查子问题、答案、来源 URL/存档与 justification。
- 公开 train 3,068：Refuted 1,742，Supported 849，Not Enough Evidence 282，Conflicting Evidence/Cherrypicking 195。dev 500：305 / 122 / 35 / 38。
- 官方仓库包含完整数据、检索、重排、verdict、justification 和评测代码。

机械适配：以 claim 为 hypothesis；把同一 source URL 的人工问答合成一个证据块，按 claim 相关性取最多三个 URL 块；没有原用户问题时使用一个全数据相同、无标签信息的固定核查指令。`Refuted=conflict`、`Supported=non-conflict`；`Not Enough Evidence` 属资料不足，应送 Baseless/unknown 分支；混合的 `Conflicting Evidence/Cherrypicking` 不自动并入冲突正类。按共享 `fact_checking_article`、原声明 URL 和证据 URL 的连通分量分组。

**标签边界：只能监督整条 atomic claim 的证据关系。没有 claim 内错误字符范围，不能产生 token 或 4-BPE 窗口金标。**

### VitaminC：最适合高相似度关系反转，但主要是自动/远程监督

- 级别与出处：[NAACL 2021](https://aclanthology.org/2021.naacl-main.52/)
- 官方数据、代码和预训练模型：[TalSchuster/VitaminC](https://github.com/TalSchuster/VitaminC)
- 数据许可：官方 `DATA_LICENSE` 明示 Wikipedia 条款，缺省为 [CC BY-SA 3.0](https://github.com/TalSchuster/VitaminC/blob/main/DATA_LICENSE)；代码仓库为 MIT。
- 实物总集：train 370,653、dev 63,054、test 55,197 个 evidence-claim 对。train 中 Supports 185,714、Refutes 131,958、NEI 52,981。
- 来源：超过 100,000 次 Wikipedia 事实修订和额外程序合成修订；同一 claim 对应几乎相同但结论相反的证据，特别适合当前“高 entailment 表面下仍有冲突”的错误。
- 官方另有 evidence-word rationale，但它标的是证据中决定 verdict 的词，不是错误 claim 中的字符 span。

机械适配：`claim` 直接作为 atomic claim，`evidence` 作为一个证据块；同一 `case_id` 及同一 Wikipedia page 必须同组。`Refutes=conflict`、`Supports=non-conflict`，NEI 分给 unknown/Baseless。训练时使用成对排序或 claim 级 BCE，使同一 claim 在相反修订证据下风险翻转。

**标签边界：仅 claim 级辅助；不得把 Refutes claim 的全部 BPE 标成错误 BPE。**

### FEVER：人工核验规模大，但 claim 伪迹更强

- 级别与出处：[NAACL 2018](https://aclanthology.org/N18-1074/)
- 官方数据与格式：[fever.ai 数据页](https://fever.ai/dataset/fever.html)
- 官方基线代码：[sheffieldnlp/naacl2018-fever](https://github.com/sheffieldnlp/naacl2018-fever)
- 数据许可：官方许可页要求遵循对应 Wikipedia 条款，缺省为 [CC BY-SA 3.0](https://fever.ai/download/fever/license.html)；基线代码 Apache-2.0。
- 总计 185,445 个声明；train 145,449，其中 Supports 80,035、Refutes 29,775、NEI 35,639。声明由人工改写 Wikipedia 句子，再由看不到原种子句的独立人工标注者判定，并为 Supports/Refutes 选择必要证据句。

机械适配：查回官方 Wikipedia 句子后，`claim` 作为 atomic claim，人工证据集按 Wikipedia 页面组成最多三个 passage；`Refutes=conflict`、`Supports=non-conflict`，NEI 进入 unknown/Baseless。共享证据页面的声明组成连通组，避免同一页面跨 fold。

**标签边界：有人工 evidence 与 claim verdict，但无 claim 内错误 span；只能做 claim 级辅助。**

### 两个可定位但次优的补充

1. **本地 DialSummFactCorr**：`prelab/benchmark_ragtruth_qa/auxiliary_dialsumm_review_v1/` 已有 2,400 个回答、600 个对话组、1,379 个人工错误 span和 981 个实际不同的人工修正文。它能派生局部修复对，但资料许可未明确、尚未做 QA 来源隔离；现阶段不能进入正式训练。
2. **FAVA**：本地 `auxiliary_fava_v2` 有 7,482 个合成回答、19,729 个精确合成错误 span；`auxiliary_fava_local_pairs_v1` 有 10,040 个单处银标修复对。官方数据 [CC BY 4.0](https://huggingface.co/datasets/fava-uw/fava-data)，[官方代码](https://github.com/abhika-m/FAVA)。其错误由模型/程序受控插入，且此前局部修复训练已经表现出表面编辑词捷径，故只适合作为合成消融，不替代人工冲突监督。

FactCC 也可生成约 100 万条规则合成的 source-claim 与变换 span，[EMNLP 2020 论文](https://aclanthology.org/2020.emnlp-main.750/)和[官方代码](https://github.com/salesforce/factCC)均公开；但训练数据依赖 CNN/DailyMail 原文，仓库只明确代码 BSD-3，未给独立数据许可说明，且规则交换容易形成表面捷径，因此不排在当前前列。

## 数据来源等级

| 数据 | 标签性质 | 可直接定位 token/window | 主要用途 |
|---|---|---:|---|
| RAGTruth `auxiliary_human_v1` | 人工 span 金标 | **是** | 第一优先，正式冲突辅助训练 |
| DialSummFactCorr | 人工 span + 人工修正 | 是，但许可/隔离未完成 | 待许可后做局部对比 |
| AVeriTeC | 真实声明、多轮人工 claim/evidence 金标 | 否 | OSINT 场景 claim 级辅助 |
| FEVER | 人工构造并独立人工核验 | 否 | 大规模 claim 级关系辅助 |
| VitaminC | Wikipedia 修订远程监督 + 程序合成 | 否 | 高相似度证据反转预训练 |
| FAVA | LLM/程序合成银标 | 是（合成 span） | 合成消融 |
| FactCC | 规则合成银标 | 是（变换 span） | 合成消融 |

## 建议执行顺序

1. 从 `auxiliary_human_v1` 只导出冲突/Baseless 双标签的原子 claim、retokenized BPE 与 4-BPE 窗口；先做独立字符映射审计。
2. 用同一 `group_id` 做五折，只给我们模型的冲突关系头加辅助损失；正式 QA fit 负责最终微调与选型。
3. 若人工辅助仍不足，再用 VitaminC 做证据反转初始化；AVeriTeC 做开放网络 claim 级场景适配。二者都不参与 token/window 金标统计。
4. FEVER、FAVA、FactCC 只做数据来源消融，防止性能提升被误解释为模型结构贡献。
5. 所有论文 baseline 保持作者模型结构、特征和训练规则；它们接受我们的统一输入映射与共同 QA answer/4-BPE 评测标签。不得改 baseline 使其变弱，也不得改成按其原论文指标决定我们的结果。

本报告只进行了本地文件只读统计、官方页面/仓库核对和小型公开元数据读取；没有训练、模型前向、GPU 使用、official test 读取或 baseline 修改。
