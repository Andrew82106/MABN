# `Original:` 人标说明转纠错 pair 的独立可行性审计

## 结论

**不能把 `Original:` 批量当成可直接替换的金标准。** 它通常是“纠错线索”或“证据说明”，不是与错误片段语法同型的正确文本。全量 4,381 个 conflict span 中，严格解析到一个非空 `Original:` 的有 4,159 个（94.9%）；但固定分层抽审 100 条只有 14 条可原样替换，62 条需要重写，24 条应排除。

由于四层各抽 25 条，而总体被 Data2txt Evident Conflict 主导，按总体层权重回代的“可直接替换”描述性比例只有 **2.7%**。这不是置信区间或全量金标准，只说明“解析成功率”严重高估“可用 pair 率”。

## 数据和边界

- 只读输入：`auxiliary_human_v1/candidate_fit.jsonl`，SHA256 `4ad2008960abfd1835c3bddd49a1f0d794c729d051e70070d0b793525078f4c0`。
- 共 9,678 个 fit 回答、4,381 个 conflict span；分层为 Data2txt EC 3,680、Data2txt SC 58、Summary EC 572、Summary SC 71。
- 本审计只读上述 fit 文件和固定人工复核表；未读取 calibration/test，未训练，未调用模型/GPU，未改 baseline。
- 因禁止读取 calibration/test，本报告只能审计 fit 内部的重复和分组风险，不能声称已排除外部分割泄漏。

## 全量机械检查

### 1. 解析与证据对齐

严格解析状态：

```json
{"missing": 202, "multiple_strict": 1, "one_strict_nonempty": 4159, "strict_empty": 1, "variant_only": 18}
```

对已解析项，证据对齐层级为：

```json
{"exact_normalized": 2101, "exact_raw": 390, "lexical_strong": 75, "not_aligned": 881, "semantic_review_candidate": 633, "structured_exact": 79}
```

`exact_*` 是文字级命中；`structured_exact` 是 Data2txt 键值命中；`lexical_strong` 是冻结词汇阈值；`semantic_review_candidate` 只是待人工复核，**不代表语义已被证明**。

### 2. bad/good 长度差和说明性措辞

- bad/good 词数比中位数：1.000，P10–P90 为 0.500–2.500；good/bad 词数比中位数：1.000。
- bad 词数中位数 4.0，`Original:` 词数中位数 3.0。
- good/bad 词数比低于 0.5：661；高于 2：366；高于 4：80。
- 含任一冻结说明/样式标记：2,660；schema/布尔键值样式：2,531。
- 字面措辞计数：`source` 15，报告式 `states/stated/says ... that/quote` 57，`not mentioned` 16，`instead` 1，`rather than` 2，`according to` 5。

这些信号会形成明显捷径：分类器可能学到“正确端更长、像引用、含 schema 键值或说明词”，而不是学到事实是否被证据支持。

### 3. 最小替换捷径

- bad/good 各至多改 1 个词：718；各至多改 2 个词：1,435。
- 纯数字变化：39；纯布尔变化：331；纯否定/方向词变化：54。
- good 可从 source 精确复制、bad 不能：2,144。若直接造 pair，模型可用“哪边更像 source”取巧。
- bad 本身也能在 source 精确找到：383。这些项可能依赖上下文、标注边界或语义判断，不能机械当成单片段反事实。

### 4. 语法代理并不可靠

冻结字符串代理把 302 条判为“表面可替换”；在 100 条人工复核上，其识别真正 direct pair 的 TP/FP/FN/TN 为 12/19/2/67，precision=0.387，recall=0.857。因此它只能做候选预筛，不能自动产金标。

## 100 条固定分层人工规则复核

抽样规则：四个 task×conflict 层各取 `sha256(seed|span_id)` 最小的 25 条，合计 100 条；样本与决定逐条保存在 `MANUAL_REVIEW.jsonl`，并完整嵌入 `RESULTS.json`。

人工规则要求把 `Original:` 原样替换回精确 offset：100 条中只有 19 条语法可接受、79 条不可接受、2 条不确定；再排除语义等价、证据不足和不完整纠错后，只剩 14 条 direct pair。

| 层 | direct | needs rewrite | reject |
|---|---:|---:|---:|
| Data2txt EC | 0 | 22 | 3 |
| Data2txt SC | 1 | 12 | 12 |
| Summary EC | 4 | 19 | 2 |
| Summary SC | 9 | 9 | 7 |

主要失败形态：

1. Data2txt 的 `Original:` 经常是 `"OutdoorSeating": false` 一类 schema 证据。事实可能对，但不能直接插进自然语言回答。
2. Summary 常给整句 source 引文，而错误标签只覆盖短片段；直接替换会破坏语法、重复主语或重复上下文。
3. 一些 SC 是近义改写或仍被 source 支持，例如 `high winds/fierce winds`、`unnamed/unidentified`，不构成稳定纠错 pair。
4. `source states`、`not mentioned`、`instead` 等是给标注者看的解释，进入目标端会泄露标签。

## 重复、冲突和分组风险

- 标准化 bad/good 重复键：270；重复冗余行：1,218；跨 group 重复键：242。
- 同一 prompt+bad 对应多个 good 的键：2。
- 与其他 released label 重叠的 conflict span：50；处在多 conflict 回答中的 span：2,360。
- 处在含多个 released label 回答中的 conflict span：3,278。只改一个 span 就把整答当“纠正后答案”，会留下未修错误。
- 任何训练/测试划分都必须按 `group_id`，并同时封锁 source/prompt/answer 身份；不能随机拆 pair。

## 严格纳入规则

只有同时满足以下条件，才进入**待复核候选池**：

1. 保留原始 offset、response/source/group 身份；offset 与标签文字逐字一致。
2. 恰有一个严格、非空的 `Original:`；排除缺失、多值、variant-only 和内嵌 `Generative/AIGC` 标记污染。
3. good 被 source 明确支持，且姓名、数字、布尔值、否定和方向都一致；纯“语义候选”必须人工复核。
4. good 与 bad 在句法角色、时态、单复数、大小写和标点上可替换；插入原位置后自然、无重复。
5. good 修正整个 conflict span，不删除其他受证据支持的信息；bad 若也被 source 支持则排除或升级复核。
6. 与其他标签不重叠。若做整答 pair，回答中的所有错误必须同时修正；否则排除整答训练。
7. Data2txt schema 值不得原样作为自然语言目标；若使用，必须先经过独立冻结的 verbalizer，再重新做证据和语法复核。
8. 排除含 `source states/not mentioned/instead/correct/should be` 等说明性语言、括号占位、ellipsis 或 reviewer 元话语的目标。
9. 排除近义等价、标注含混和 questionable SC；不能为了数量把它们当反事实。
10. 标准化去重；按 group/source/prompt/answer 联合隔离。数字、实体、布尔、否定等最小编辑另行分层报告，并控制两端长度/样式。

按冻结机械条件，全量只剩 272 个 span 候选、去重后 263 个；若额外要求整答只有一个 released label，则为 150 个、去重后 146 个。**这些仍不是金标，必须人工确认语义和语法。**

## 可复现性

运行：

```powershell
prelab/.venv/Scripts/python.exe prelab/benchmark_ragtruth_qa/research/human_original_pair_feasibility_v1/audit.py final
```

脚本 SHA256：`c646d583580170af61b8796ac4bebbf7cb28fda8cdb66a0228305e36a8bdb1f5`；人工复核表 SHA256：`bb1bc87ffa441c81cb55dc1a3f8b2e4ebb3ee81224413eeed67760de43282d70`。
