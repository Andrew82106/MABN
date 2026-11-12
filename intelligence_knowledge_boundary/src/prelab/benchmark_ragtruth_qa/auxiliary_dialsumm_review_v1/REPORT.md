# DialSummFactCorr：训练数据实物核查

**它确实保留了人工“原摘要—修正摘要—错误跨度”，比FAVA的编辑投影更容易追溯。可以用于局部关系修复训练，但不是现成的2400条最小事实反转对；对错引资料编号的问题也没有直接监督。** 数据专门许可未查到明确说明，不能把工具的MIT许可当成数据许可。

本轮已完成作者文件下载、仅train解析、全量统计和不改标签的规范导出；没有运行模型/GPU，没有读取本项目QA校准/测试或该资源val/test/total内容，没有混入现有训练。

## 来源、许可、划分

- 正式出处：ACL2023主会长文 *Reference Matters: Benchmarking Factual Error Correction for Dialogue Summarization with Fine-grained Evaluation Framework*。[论文](https://aclanthology.org/2023.acl-long.779/)
- 作者仓库锁定 `72e3589654a7366a1b3ec6541a4842ecc385fd86`。README指向Google Drive的`378_Data.zip`，实际下载1,015,480字节，SHA256 `83c5de82fdb09d54da082de719d0c27c78c0c2cbecf0b9a34b2b803954a9b1fd`。[作者仓库](https://github.com/kite99520/DialSummFactCorr)、[作者数据链接](https://drive.google.com/file/d/1bIBYc81RsnYDSU_5jI7ALVYe5XqKOKQ-/view)
- 压缩包内有`DialogSum/{train,val,test,total}.json`、`SAMSum/{train,val,test,total}.json`和数据README。**只解压并解析两个train及README**；其余成员仅记录文件名、大小、CRC，未打开内容。见 [ARCHIVE_INVENTORY.json](ARCHIVE_INVENTORY.json)、[TRAIN_ONLY_EXTRACTION.json](TRAIN_ONLY_EXTRACTION.json)。
- 论文按每个原对话分300/100/100组，各对话有4个模型版本，故每语料1200/400/400条。本轮实测train各300对话、合计600个不同材料哈希；未读val/test身份，因此跨分区不重叠只能引用作者设计，不能称本轮独立验证过。DialogSum训练记录的`test_13`等ID来自上游摘要数据原test；它们位于作者为本FEC任务另分的**train.json**，不是本项目测试数据。[论文§6.3](https://aclanthology.org/2023.acl-long.779.pdf)
- 数据README和根README无许可条款，压缩包无数据LICENSE，GitHub根license为null。`errant/LICENSE.md`是Christopher Bryant、Mariano Felice于2017年为工具代码提供的MIT许可，未声明覆盖后来对话/人工标注。**本轮只能报告数据许可未明确，不能宣称MIT或推断禁止使用。后续实际训练/发布的使用范围需澄清。** [工具许可证](https://github.com/kite99520/DialSummFactCorr/blob/72e3589654a7366a1b3ec6541a4842ecc385fd86/errant/LICENSE.md)

## 发布字段与全train规模

每个对话保留`id`、`dialogue`、原摘要参考文本`references`和四个`model_summaries`。每个模型版本直接保留：

- `original_summary`：模型原输出；
- `modified`：人工修正全文；
- `consistency`：人工对原输出的整答判断；
- `error_categories[{start,text,type}]`：原输出中的人工错误字符跨度及类型。

因此无需从删除/插入标签猜“原始答案”。`references`是摘要数据原参考摘要，**不是本次修正目标，也不作为检测证据**；证据只用`dialogue`。作者工具FERRANTI依据原版与修正版自动生成编辑对应和类型；这部分自动对齐不能冒称原始人工跨度。[工具说明](https://github.com/kite99520/DialSummFactCorr/blob/72e3589654a7366a1b3ec6541a4842ecc385fd86/errant/README.md)

| 实测项目 | 数量 |
|---|---:|
| 原回答 | 2400（每模型600） |
| 材料组 | 600（每语料300，无相同正文哈希重复） |
| 原答人标一致 | 1415 |
| 原答人标不一致 | 985 |
| 人工修正文实际不同 | 981 |
| 人标不一致但无错误跨度、修正也未变化 | 4（均DialogSum） |
| 原人工错误跨度 | 1379，**全部1379字符slice exact** |
| 错误跨度字符长度 | 中位9，95分位42，最大128 |
| 981个改动答案的token编辑距离 | 中位3，95分位15，最大41 |

人标类型：实体EntE **471**，谓词PredE **261**，指代CorefE **155**，条件/环境CircE **138**，资料外信息OutE **130**，语法GramE **91**，连接关系LinkE **88**，其他OthE **45**。PredE涉及 **244条回答**；不能把所有1379跨度都算成事实反转。

## 能恢复多少最小对应

以下是**审计用正则词元**的单位代价编辑路径统计，不是Qwen/Llama的BPE，也不是FERRANTI原作者算法。算法对多个最优路径保守标歧义；唯一不代表语义上唯一。所有导出的对齐字段均标为派生诊断，原人工标签不变。

| 逐步限制 | 回答数 |
|---|---:|
| 981对中有唯一最优token编辑路径 | 502 |
| 唯一路径且仅一个连续改动块 | 361 |
| 用该块替换后，其余全文逐字不变且精确恢复人工修正文 | 283 |
| 上述单块，两侧各1–3审计词元的替换 | 206 |
| 只有一个人标PredE、单块两侧最多3词元（允许加/删否定词） | **37，来自32个对话** |
| 其中原人标整段范围可直接对应到修正版、两侧范围外全文exact | **33，来自28个对话** |

37中纯替换为28条，另9条涉及插入/删除。最后4个范围不能直接沿原标注投影，保留待核查，不扩大金标范围、不自动补标签。479条编辑路径不唯一不等于不可用，只是本轮不替它们强行决定逐字对应；可保留整答人工对，后续按明确方法处理。

## 亲读样例：哪些真对应目标，哪些仍有问题

完整资料/原答/修正/人标和派生对应见 [TRAIN_EXAMPLES.json](TRAIN_EXAMPLES.json)。

1. **SAMSum13717092 / UniLM：明确的否定反转。** 对话明确决定不去Linda老师的课，原答写`are going`，人工改为`aren't going`。其他文字不变，可对应原人标范围。这和我们“大小/因果/关系说反”漏检的训练目标直接相关。
2. **DialogSum test_256 / UniLM：新增not。** 对话中的说话者说`don't take ... very seriously`，原摘要写`to take ... very seriously`，人工改为`to not take ... very seriously`。材料、主体、上下文相同，只有否定区别。这里只评价是否忠实复述对话，不背书对话中的社会概括。
3. **DialogSum test_13 / MV-BART：局部修复不等于整篇已真。** 资料说还有很多时间；人工把`to hurry up`修成`doesn’t need to hurry up`，局部方向修复成立。但原文和修正文都留着`Tom reminds #Person2#`，而对话中Tom就是Person2，其他角色表述仍需核查。
4. **SAMSum13729857 / UniLM：人工修正也可能引入错关系。** 对话中David建议Jane买iPad；人工修正文却写`Jane recommends David`。不能仅凭`modified`字段把全文所有位置强制标成绝对正确。

这些是train事后例读，不能估计人工标签总体错误率，也未据此改动发布标注。人标比“未标记即真”的推测更有依据，但不是逻辑保证。

## 可训练方案与适用边界

**技术上可以做小型、定向的局部对比训练。** 最直接的第一步是核查上述33条可完整对应的人标关系片段，用同一对话和同一回答上下文分别输入原版、人工修正版；每次输入只能出现其中一个版本，修正版不能作为提示拼给原版。只在原人工范围及其精确对应范围上比较风险，其他未核内容不补“正确”标签。两版本及同对话的四个生成器输出必须同组。

这33条仅覆盖28个材料组，适合做可学性或小规模辅助实验，**不足以单独支撑一个稳健的通用EC检测器**。扩大时可以依次使用206个精确短替换、283个精确单块修改，再处理244条带PredE人标的回答；每层都保留原人标类型、对齐歧义和全文语义核查状态。并非所有PredE都是真反义词，实物中还包含`decide→plan`、`reminds→teaches`等动作/确定性细化。

若做常规单答案监督，可保留1415条原答人标一致样本和带位置的原答错误样本；**4条整答不一致但没有位置的条目不能把全部词元补0**。981对可提供人工编辑监督，但不宜未经核查把每条修正全文都当严格无风险正例。

对当前EC故障的适配：

- **关系反转/主体绑定：有直接价值。** PredE与EntE/CorefE提供动作、否定、参与者的局部改写；可以作为同一事实最小变化的训练目标。
- **错引资料编号：缺直接监督。** 每条只有一个对话材料，没有passage1/2/3的多来源引用标签；指代错误可帮助角色对应，但不能冒充13575、14229那类跨段来源归属训练数据。

当前规范导出为 [train_pairs_review.jsonl](train_pairs_review.jsonl)，保留全部2400条，未筛成训练集；[manifest.json](manifest.json)明确标记 **review-only、许可未明确、尚未做与本项目材料的隔离检查**。后续接入前需澄清使用范围、做材料层面的隔离并复核拟用局部对。现有FAVA/PsiloQA/QA训练均未修改。

实际执行 `inspect_train.py` 最终exit0。原train文件和规范导出已逐字段回放核对，全部原文、人工修正、consistency和1379个人标位置保持不变；没有新增正式真/假标签。
