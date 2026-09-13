# 最小事实配对：现有数据能支持到哪里

**可以准备“同资料、同上下文、只换一个事实片段”的小型配对实验，但不能直接把7482条FAVA修正投影全部标成正确答案。** 现有文件能定位改动；不能独立认证篡改前原文的身份，也没有整篇事实均被资料支持的证明。最窄、结构合格的待核查池为 **498条**，尚未成为新训练标签。

本轮只读取已下载的官方训练文件、固定7482候选及作者代码/论文。没有读取本地cal/test、模型分数，未下载评测数据、训练、运行GPU或修改现有辅助数据。全量统计脚本最后实际exit0，结果见 [FAVA_PAIR_COUNTS.json](FAVA_PAIR_COUNTS.json)，逐行依据在 [PAIRABILITY_INDEX.jsonl](PAIRABILITY_INDEX.jsonl)。

## 1. FAVA原生成流程有配对，但发布文件丢了原文身份

FAVA为COLM2024正式论文。作者生成脚本先要求GPT-3.5根据Wikipedia材料改写3–5句，再逐类插错；中间输出保留`evidence`、`diversified_passage`、`errored_passage`等字段。这里的`diversified_passage`才是插错前文本。但“只依据材料”是一条生成指令，代码没有随后逐事实核验，也未验证每次插错是否遵守只改标记区的要求。[论文](https://openreview.net/forum?id=dJMTn3QOWO)、[作者生成代码](https://github.com/abhika-m/FAVA/blob/7ec08c2dcbff016a7b6af93152994600b646f0a5/training/generate_train_data.py#L306)

后处理仅导出核查prompt和带编辑标记的completion，不保留`diversified_passage`或其独立哈希。当前锁定发布版本的 **30073行全部只有prompt/completion两字段**；公开仓库元数据也只列README、training和人工评测annotations，没有该中间文件。因此无法把反向编辑结果与真正生成前原文逐字核验。本轮未打开annotations。[后处理代码](https://github.com/abhika-m/FAVA/blob/7ec08c2dcbff016a7b6af93152994600b646f0a5/training/process_train_data.py#L23)、[发布训练文件](https://huggingface.co/datasets/fava-uw/fava-data/blob/f4e40415d525b18bcb49ba241f26fd8e12eeb606/training.json)

标签方向也要注意：生成阶段`delete`是原词、`mark`是插入错词；后处理把两者交换。因此**当前completion的delete是待删错误，mark是建议恢复内容**。没有mark的contradictory/invented通常是删整句，不提供一个语义相反的正确句。[作者标签转换](https://github.com/abhika-m/FAVA/blob/7ec08c2dcbff016a7b6af93152994600b646f0a5/utils.py#L38)

## 2. 固定7482条的实测条件

| 条件 | 实际数量 | 能证明 / 不能证明 |
|---|---:|---|
| 原错误文本可从markup精确重建 | 7482条 | 标签位置按解析游标确定，不用`find`猜重复片段。 |
| 保留独立篡改前原文字段/哈希 | 0条 | 不能认证投影就是原始`diversified_passage`；不是说所有投影都错。 |
| 修正投影与错误文本不同 | 7476条 | 6条没有有效改变，不能形成差异对。 |
| 至少一个非空替换 | 5498条 | 可构造字符串替换，但原词/新词真伪仍需核查。 |
| 至少一对单一mark/delete实体或关系、两侧各1–3空白分词 | 5275条，10040处 | 只是形式上的小修改；可能一答多错、类型错标。 |
| **整答只有一个上述短替换，且投影的所有变化都限于该节点** | **498条** | 最小候选池；仍不是498条已核真/假对。 |
| 顶层所有错误节点投影后都有变化 | 7404条 | 不能推出全部错误已修复；78条还有bare/no-op节点。 |
| 投影含类型节点之外的编辑 | 11条 | 不能只看类型标签数就断言“单改动”。 |
| 含嵌套类型节点 | 84条 | 需单列，不能默认每个节点都是独立不重叠替换。 |
| 修正后为空/纯空白 | 10条 | 不能把空回答当有依据的正常正例。 |
| 非空整篇投影逐字包含于某个reference | 0条 | 窄字符串检查；不匹配不代表不受支持。 |
| 只压缩空白后整篇包含于某个reference | 2条 | raw20029、22236；只证明文本包含，不恢复生成前身份或保证外部真实性。 |

**“原文身份可追溯 + 编辑位置可证 + 全文语义支持已认证”三项同时被现有发布字段证明的条数为0。全文实际受支持的总数未知，不能把这个0解释为没有正确答案。** 本轮没有新增逐条事实认证标签。10个完全空的类型标签不计有效编辑；其中两个空subjective标签不包含原文风险字符，也未改变既有四类候选过滤。

498条条件可直接由审计索引复现：`top_typed_nodes==1`、`entity_relation_1to3word_pairs==1`、`all_projection_changes_confined_to_typed_nodes==true`。原先499条单节点候选中，raw5274还有类型标签外的`registration↔vehicle`编辑，故收严为498。所有旧7482训练条目仍原样保留。

三个已读训练例说明语义核查不能省略（完整原材料在 [TRAIN_ONLY_EXAMPLES.json](TRAIN_ONLY_EXAMPLES.json)）：

- **raw12：可用方向。** 资料说日本海军策略是通过赢得一次决战取胜；标记给出`winning↔losing`，还有`well-trained↔poorly-trained`。这类局部反义关系有明确资料依据，但一答有两处修改，不能直接说整答只变一个事实。
- **raw6：修正并不保证真。** 投影仍写“these audits often lead to more inaccuracies and manipulation of votes”，五段材料没有支持这个结论；其原替换也被标entity。这反驳了“反转标记就自动得到正确答案”的做法。
- **raw24：类型标签也可能不符合目标。** `boring↔captivating`被标entity，实际上是在评价剧情是否吸引人，不适合当客观事实反例，即使结构上属于最小替换池。

## 3. 最小可执行方案：核一个事实，不先声称整篇都真

**优先方案是对498条结构候选逐条核查目标事实，建立局部配对，暂不构造整答真/假标签。**

1. 从原markup按游标读出替换两侧与字符范围，核两版的替换范围之外逐字相同；资料始终为原5段，问题仍为空，不添原作者没有的问题。
2. 针对这一个事实记录支持片段及其原文坐标：恢复侧确被资料支持；错误侧与同一实体、事件、时间条件下的资料冲突。只有未找到错误侧依据，不等于已证明它矛盾。主观措辞、仅语法变化、来源间有冲突、关系仍歧义的条目隔离并保留原因。
3. 给局部片段“支持/冲突/不能判”的独立核查结果；其他未检查内容仍未知，**不把所有未改字符补成严格负风险**。如需整答正负BCE，必须另核整答全部断言，本轮数据条件不足以自动做这一步。
4. 成对目标比较同一事实两版的局部风险，保留两侧各自字符坐标；变长替换后不能直接沿用原token下标。来自同一原回答/reference组的全部版本放同组，沿既有来源隔离规则。不得借校准分数筛选样本。

这和当前FAVA辅助训练不同：当前只给错误答案及各位置银标签；新问题要求模型在**资料和上下文相同、只换关系/实体的两种版本**之间作差别判断。是否提高EC召回尚待实验。本轮仅给条件和审核索引，未导出正式配对训练集或改变标签。

## 4. PsiloQA的golden_answer不能直接补成最小真/假对

官方`generate_qa.py`把wiki材料交给LLM生成问题和答案；解析器只检查JSON结构/非空，未做原文抽取或事实验证。其筛选逻辑主要过滤问题质量和生成回答的拒答，不等于逐条认证golden_answer。论文称GPT-4o，而当前源码默认设置可被环境覆盖，不能仅由默认值反推发布行的真实生成器。[官方问题生成](https://github.com/s-nlp/PsiloQA/blob/95233aefcfbdfc2b0876ffc7d45729d7fc69bd2d/psilo/dataset/generate_qa.py)、[筛选代码](https://github.com/s-nlp/PsiloQA/blob/95233aefcfbdfc2b0876ffc7d45729d7fc69bd2d/psilo/dataset/filter_samples.py)、[Findings EMNLP2025论文](https://aclanthology.org/2025.findings-emnlp.626/)

现有train的标准答案与待检回答可能在内容、长度、措辞上大幅不同，不能归因于同一事实的最小修改。它最多是**另行核查后的弱训练目标来源**：可帮助确定拟修改的事实值，再核对原wiki和目标句。不能把它直接当整篇全真的监督，也**不能把golden_answer加到现有检测器输入**。当前candidate/model_inputs白名单和训练不改。

## 5. 只补一项已有人工配对资源

**DialSummFactCorr / Reference Matters，ACL2023主会长文**更直接：人工对同一对话下的模型摘要标错误词/短语、分类型，并用尽量少的改动写出修正版。论文§6.3按对话划分，每个语料1200/400/400，两个语料合计训练2400条对应600个对话；不是2400个独立来源。包含实体、谓词等错误；FERRANTI从原版/修正版对齐提取替换、删除、插入，仍不保证每条只有一个改动。[论文§3、§5、§6.3](https://aclanthology.org/2023.acl-long.779.pdf)

作者仓库已提供数据下载链接和对齐工具；本轮只核论文及README，**未下载/打开任何数据分区，尚未验证实际字段、训练文件及最终可用数**。后续若用，只考虑作者新划分的train，按对话组保持版本同组，再做本项目来源隔离与字符核验。它来源于对话摘要、并非联网QA，适合作为局部关系改写辅助监督，不替代公共QA评测。[官方仓库及数据入口](https://github.com/kite99520/DialSummFactCorr)、[官方对齐工具](https://github.com/kite99520/DialSummFactCorr/tree/main/errant)

因此不必重新凭空编一个大数据集：**先核498条FAVA最小替换是否能成为可靠局部配对；若要现成的人工原版/修正版，再核DialSummFactCorr的train。** 不能把“增加同样银标签”当成已经实现成对事实监督。
