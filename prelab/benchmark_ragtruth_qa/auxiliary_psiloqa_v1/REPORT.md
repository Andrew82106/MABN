# PsiloQA 自动辅助候选导出

**已导出 15,847 答、6,862 个文章/材料组，尚未加入训练。** 来源是此前锁定的官方 PsiloQA 英文 train；没有新下载或读取 validation/test。

| 项目 | 数量 |
|---|---:|
| 原英文 train | 16,115 |
| 字符或 HAL 标记失败 | 263 |
| 同输入标签冲突的整组记录 | 6（3 组） |
| 两项排除的交集 | 1 |
| 字符筛选后额外隔离冲突 | 5 |
| 合计排除 | 268 |
| 最终候选 | **15,847** |
| 带自动风险标签 / 无标记 | **15,212 / 635** |
| 坐标通过、complexity 为空且保留 | 4 |

没有选择冲突组中的某一份，也没有修文、移动标签、补标或统一 complexity。原 6,888 个文章组及全部成员去向都有记录；26 组没有保留成员。现有 source-only 隔离结果为 0，直接复用，未重新扫描材料。

输入只取原 `wiki_passage`、`question`、`llm_answer`，分别映射为 `retrieved_passages`、`question`、`original_response`。`model_inputs.jsonl` 除不含模型信息的行号 ID 外仅有这三个原文字段。候选标签保留原始 `[start,end)` 字符对，`text` 只按同一原回答切片派生。`golden_answer`、生成器及原含模型名 ID、标记文本等全部放在单独的 `provenance.jsonl`，不拼接到检测器输入。

最终有 27,996 个自动风险片段，共 2,755,496 个片段字符。回答字符数中位 183、95 分位 1,092.7、最大 3,615；原材料字符数中位 347、最大 10,318。完整分布在 `manifest.json`。数据只有自动二类风险标签，没有可冒用的人标四类错误分类。

风险回答比例约 95.99%，来自无资料生成、自动标注和拒答预过滤的构造流程，**不是部署环境的风险发生率**。无标记位置也只是自动标注下的负例，不能声称已验证正确。字符对齐通过不等于语义标签完全可靠。

产物：

- `candidate_fit.jsonl`：候选输入、原标签、稳定 source/group ID。
- `model_inputs.jsonl`：严格三字段输入白名单。
- `provenance.jsonl`：全部原英文行的独立出处及版本/原文哈希。
- `all_english_decisions.jsonl`、`excluded_candidates.jsonl`：逐条保留/排除原因。
- `article_group_index.jsonl`：全部原文章组、保留和排除成员。
- `original_review_failure_ledger.jsonl`、`original_article_groups.jsonl`：既有账本逐字节副本。
- `DATA_PROTOCOL.json`、`design_freeze.json`、`DATA_CHECK.json`、`complete.json`：方案、来源锁定及验证。

导出和核对均 actual exit 0：原三字段逐值一致，标签范围/切片一致，候选 ID 和输入唯一，剩余同输入标签分歧为 0，所有原组成员可追溯。未改现有训练、校准、测试或三份总状态文档。

后续已获准的工作仅为 CPU tokenization：完整材料＋SEP＋原问题＋SEP＋原回答；Llama raw BPE 只提供辅助坐标，不是这些生成器的历史轨迹。使用现有严格字符/NFC 映射，保留无标记答案；失败和超 8,192 输入必须逐条记录、不能静默截断。单独的 `tokenization_v1` 报告给出实际结果，仍不表示训练授权或已训练。
