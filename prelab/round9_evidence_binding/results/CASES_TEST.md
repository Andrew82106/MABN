# Round9：全部测试回答的定位复查

本文件依照冻结 `inputs.jsonl` 的原始顺序，完整列出测试集 **40 题组、80 条回答**，没有按结果挑选案例。只读取已冻结回答、参考标注、分数和阈值；没有重跑模型或调整边界。

## 字段与计数口径

- 问题、条件来自输入；原始回答是生成记录的 `response`，完整保留编号和额外解释。`complete` 为资料完整，`partial` 为目标条目仍出现但删除关键事实；这些条件标签没有送入模型。
- “参考风险片段”来自冻结 `risk_spans`。这些标注由助手初标、独立复核和裁决形成，**不是研究者人工金标**。回答与当前展示资料不一致或没有依据可记风险，即使偶然答对也不因此豁免。
- 区间采用完整回答中的 Unicode 字符下标 **`[start,end)`，从 0 开始、右端不含**；BPE 编号也是从 0 开始。引号内用 JSON 转义显示原文，保留前导空格和换行。
- 报警条件严格为 **`score >= threshold`**，展示所有达到阈值的原始 BPE 范围，含纯标点、空白词元，以及拒答、未决和解析范围外文本。没有将报警向完整词语或整句话扩展。
- 仅为显示，把重叠、相邻，或只隔标点/空白的报警范围合并；不跨越任何未报警的字母或数字。显示补入的标点/空白不参与计数。每段保留原始报警 BPE 编号，可直接核查。
- **TP/FP/FN 按 BPE 词元计数，不按字符、单词或报警段计数。**直接使用冻结分数文件的 `main_eligible` 与 `gold`：TP=可评风险词元报警；FP=可评非风险词元报警；FN=可评风险词元未报警。没有根据显示区间重新计算标签。
- 表中“可评 / 其他含字 / 纯标点空白”是报警词元数：第一项进入主指标；后两项只供描述。“含字”沿用冻结 `lexical` 字段，表示含字母或数字。拒答、未决没有主指标，记为“—”，不能视为 FP。
- 每条完整回答只有一个待答项。已决项以外的编号、解释等文本仍展示报警，但不进入该项 TP/FP/FN。

## 固定阈值与覆盖

阈值在测试前冻结；选择目标为验证集词元微平均 F1，平手依次优先精确率和更高阈值。

| 方法 | 冻结阈值 | 测试 TP | FP | FN |
|---|---:|---:|---:|---:|
| lb_coarse | 0.993983032945922 | 48 | 83 | 69 |
| lb_fine | 0.7988980460548466 | 59 | 28 | 58 |
| binding_fine | 0.7513038137814632 | 64 | 36 | 53 |

80 项中，63 项进入主指标（45 项无风险、18 项有风险），15 项拒答排除，2 项未决。共有 2,086 个生成 BPE；主指标计 1,318 个词元，其中 117 个风险词元。下面每条的 TP/FP/FN 加总与冻结正式指标一致。

## 全部 80 条回答

### 01. ragognize_train_0158__complete

- 输入行：`0`（从 0 开始）；类别：`time`；条件：`complete`。
- 问题：In what month and year was Jaylan Pearman's Perth Glory scholarship deal extended for another season?

**原始回答**

```text
Jaylan Pearman's Perth Glory scholarship deal was extended for another season in June 2024.
```

**参考标注**

已决、无风险（有资料依据）；参考风险片段为空。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | 无 | 0 / 0 / 0 | 0 | 0 | 0 |
| lb_fine | 无 | 0 / 0 / 0 | 0 | 0 | 0 |
| binding_fine | 无 | 0 / 0 / 0 | 0 | 0 | 0 |

生成记录 SHA256：`30164fb07d8eb48245b2b29d11f7f02ccb8e5fbf9dd563090086b44dbd08c06d`。

### 02. ragognize_train_0158__partial

- 输入行：`1`（从 0 开始）；类别：`time`；条件：`partial`。
- 问题：In what month and year was Jaylan Pearman's Perth Glory scholarship deal extended for another season?

**原始回答**

```text
The search results do not provide information about Jaylan Pearman's Perth Glory scholarship deal extension.
```

**参考标注**

拒答，排除主指标；没有可评参考风险片段。

冻结标注说明：Accurately declines the missing extension date; current target passage only concerns the 2025 youth team.

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,3)</code> <code>&quot;The&quot;</code>（BPE 0）<br><code>[10,55)</code> <code>&quot; results do not provide information about Jay&quot;</code>（BPE 2–8）<br><code>[66,68)</code> <code>&quot;&#x27;s&quot;</code>（BPE 12） | 0 / 9 / 0 | — | — | — |
| lb_fine | <code>[18,51)</code> <code>&quot; do not provide information about&quot;</code>（BPE 3–7） | 0 / 5 / 0 | — | — | — |
| binding_fine | <code>[18,51)</code> <code>&quot; do not provide information about&quot;</code>（BPE 3–7） | 0 / 5 / 0 | — | — | — |

生成记录 SHA256：`1db024c959b89ae4de228e2010438557f0e5eb2c8431cdc02eec0f21d70b12ee`。

### 03. ragognize_train_0519__complete

- 输入行：`10`（从 0 开始）；类别：`time`；条件：`complete`。
- 问题：On what date was The Safe House scheduled for release in French cinemas?

**原始回答**

```text
1. The Safe House was scheduled for release in French cinemas on 19 March 2025.
```

**参考标注**

已决、无风险（有资料依据）；参考风险片段为空。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |
| lb_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |
| binding_fine | 无 | 0 / 0 / 0 | 0 | 0 | 0 |

生成记录 SHA256：`18d2f19444346452d5f0d8f6a8070e1793889d39bd91280c0b1cb83c10248f4b`。

### 04. ragognize_train_0519__partial

- 输入行：`11`（从 0 开始）；类别：`time`；条件：`partial`。
- 问题：On what date was The Safe House scheduled for release in French cinemas?

**原始回答**

```text
The Safe House was scheduled for release in French cinemas on an unspecified date, while "The Ties That Bind Us" was released on 19 February 2025.
```

**参考标注**

已决、无风险（有资料依据）；参考风险片段为空。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,3)</code> <code>&quot;The&quot;</code>（BPE 0）<br><code>[61,76)</code> <code>&quot; an unspecified&quot;</code>（BPE 11,12）<br><code>[82,88)</code> <code>&quot; while&quot;</code>（BPE 15）<br><code>[141,142)</code> <code>&quot;2&quot;</code>（BPE 32） | 5 / 0 / 0 | 0 | 5 | 0 |
| lb_fine | <code>[61,81)</code> <code>&quot; an unspecified date&quot;</code>（BPE 11–13） | 3 / 0 / 0 | 0 | 3 | 0 |
| binding_fine | <code>[61,81)</code> <code>&quot; an unspecified date&quot;</code>（BPE 11–13） | 3 / 0 / 0 | 0 | 3 | 0 |

生成记录 SHA256：`f6f281b46434b630f11d9572b85e33a63678d6fc132435f16f6f4309a3565992`。

### 05. ragognize_train_0860__complete

- 输入行：`22`（从 0 开始）；类别：`time`；条件：`complete`。
- 问题：In what year did the Tianjin Federation of Trade Unions receive the national People’s Satisfactory Civil Service Collective award?

**原始回答**

```text
1. The Tianjin Federation of Trade Unions received the national People’s Satisfactory Civil Service Collective award in 2022.
```

**参考标注**

已决、无风险（有资料依据）；参考风险片段为空。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | 无 | 0 / 0 / 0 | 0 | 0 | 0 |
| lb_fine | 无 | 0 / 0 / 0 | 0 | 0 | 0 |
| binding_fine | 无 | 0 / 0 / 0 | 0 | 0 | 0 |

生成记录 SHA256：`25b3a1c691044a497cd34e13c92a726c6af627aa58166c1c54e07fdb2bdffa70`。

### 06. ragognize_train_0860__partial

- 输入行：`23`（从 0 开始）；类别：`time`；条件：`partial`。
- 问题：In what year did the Tianjin Federation of Trade Unions receive the national People’s Satisfactory Civil Service Collective award?

**原始回答**

```text
There is no information provided about the Tianjin Federation of Trade Unions receiving a national People’s Satisfactory Civil Service Collective award in the given search results.
```

**参考标注**

拒答，排除主指标；没有可评参考风险片段。

冻结标注说明：Accurately reports that the supplied union passage gives no award information.

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,5)</code> <code>&quot;There&quot;</code>（BPE 0）<br><code>[8,11)</code> <code>&quot; no&quot;</code>（BPE 2）<br><code>[23,32)</code> <code>&quot; provided&quot;</code>（BPE 4）<br><code>[87,98)</code> <code>&quot; a national&quot;</code>（BPE 15,16）<br><code>[105,107)</code> <code>&quot;’s&quot;</code>（BPE 18）<br><code>[109,120)</code> <code>&quot;atisfactory&quot;</code>（BPE 20,21）<br><code>[145,154)</code> <code>&quot; award in&quot;</code>（BPE 25,26）<br><code>[158,179)</code> <code>&quot; given search results&quot;</code>（BPE 28–30） | 0 / 13 / 0 | — | — | — |
| lb_fine | <code>[0,5)</code> <code>&quot;There&quot;</code>（BPE 0）<br><code>[145,151)</code> <code>&quot; award&quot;</code>（BPE 25） | 0 / 2 / 0 | — | — | — |
| binding_fine | <code>[0,5)</code> <code>&quot;There&quot;</code>（BPE 0）<br><code>[23,32)</code> <code>&quot; provided&quot;</code>（BPE 4）<br><code>[145,151)</code> <code>&quot; award&quot;</code>（BPE 25）<br><code>[158,164)</code> <code>&quot; given&quot;</code>（BPE 28） | 0 / 4 / 0 | — | — | — |

生成记录 SHA256：`d4472a723f120f6ed70841596cd21da0517f532e349889587126e20c93bd2e5c`。

### 07. ragognize_train_1298__complete

- 输入行：`46`（从 0 开始）；类别：`time`；条件：`complete`。
- 问题：In what year was OmiSoore Dryden appointed to the Dalhousie University School of Nursing?

**原始回答**

```text
1. In 2025, OmiSoore Dryden was appointed to the Dalhousie University School of Nursing.
```

**参考标注**

已决、无风险（有资料依据）；参考风险片段为空。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |
| lb_fine | 无 | 0 / 0 / 0 | 0 | 0 | 0 |
| binding_fine | 无 | 0 / 0 / 0 | 0 | 0 | 0 |

生成记录 SHA256：`e8603016fb97e7981f0c0248599d186313117fb0cae639e5976893580fa63108`。

### 08. ragognize_train_1298__partial

- 输入行：`47`（从 0 开始）；类别：`time`；条件：`partial`。
- 问题：In what year was OmiSoore Dryden appointed to the Dalhousie University School of Nursing?

**原始回答**

```text
The search results do not provide a specific year for OmiSoore Dryden's appointment to the Dalhousie University School of Nursing.
```

**参考标注**

拒答，排除主指标；没有可评参考风险片段。

冻结标注说明：Correctly says appointment year is not provided.

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,44)</code> <code>&quot;The search results do not provide a specific&quot;</code>（BPE 0–7）<br><code>[49,53)</code> <code>&quot; for&quot;</code>（BPE 9）<br><code>[62,66)</code> <code>&quot; Dry&quot;</code>（BPE 14）<br><code>[83,86)</code> <code>&quot; to&quot;</code>（BPE 18）<br><code>[90,95)</code> <code>&quot; Dalh&quot;</code>（BPE 20,21）<br><code>[98,129)</code> <code>&quot;ie University School of Nursing&quot;</code>（BPE 23–27） | 0 / 18 / 0 | — | — | — |
| lb_fine | <code>[21,35)</code> <code>&quot; not provide a&quot;</code>（BPE 4–6）<br><code>[118,121)</code> <code>&quot; of&quot;</code>（BPE 26） | 0 / 4 / 0 | — | — | — |
| binding_fine | <code>[21,35)</code> <code>&quot; not provide a&quot;</code>（BPE 4–6）<br><code>[118,121)</code> <code>&quot; of&quot;</code>（BPE 26） | 0 / 4 / 0 | — | — | — |

生成记录 SHA256：`9c745a1a92491a10ec08d836183b646eeb4237736b6d92559cea312309c2e8d5`。

### 09. ragognize_train_1593__complete

- 输入行：`52`（从 0 开始）；类别：`time`；条件：`complete`。
- 问题：On what date did After Party premiere in Czech cinemas?

**原始回答**

```text
1. After Party premiered in Czech cinemas on 14 November 2024.
```

**参考标注**

已决、无风险（有资料依据）；参考风险片段为空。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |
| lb_fine | 无 | 0 / 0 / 0 | 0 | 0 | 0 |
| binding_fine | 无 | 0 / 0 / 0 | 0 | 0 | 0 |

生成记录 SHA256：`2ac68e690cbcbc50da33f42fc26e873abf0aa2277676817dffe82b53973cd6fa`。

### 10. ragognize_train_1593__partial

- 输入行：`53`（从 0 开始）；类别：`time`；条件：`partial`。
- 问题：On what date did After Party premiere in Czech cinemas?

**原始回答**

```text
1. After Party premiered in Czech cinemas on an unspecified date in 2024.
```

**参考标注**

未决，排除主指标；不强制标出风险片段。

冻结标注说明：2024 Czech film 是影片年份，不严格等于在捷克影院首映年份；回答在拒绝精确日的同时肯定地区首映在 2024。建议未决以裁定影片年与特定市场上映年的推导界限。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0）<br><code>[14,24)</code> <code>&quot; premiered&quot;</code>（BPE 4）<br><code>[33,47)</code> <code>&quot; cinemas on an&quot;</code>（BPE 7–9）<br><code>[64,68)</code> <code>&quot; in &quot;</code>（BPE 12,13） | 0 / 6 / 1 | — | — | — |
| lb_fine | <code>[47,59)</code> <code>&quot; unspecified&quot;</code>（BPE 10）<br><code>[64,70)</code> <code>&quot; in 20&quot;</code>（BPE 12–15） | 0 / 4 / 1 | — | — | — |
| binding_fine | <code>[47,59)</code> <code>&quot; unspecified&quot;</code>（BPE 10）<br><code>[64,70)</code> <code>&quot; in 20&quot;</code>（BPE 12–15） | 0 / 4 / 1 | — | — | — |

生成记录 SHA256：`6c85dd2311b9d25b27aba1b3f3dc027b522ffbb1e367e16ad1861d438412ac06`。

### 11. ragognize_train_1681__complete

- 输入行：`56`（从 0 开始）；类别：`time`；条件：`complete`。
- 问题：On what date were the seedings for the 2025 Belgian Darts Open confirmed?

**原始回答**

```text
1. The seedings for the 2025 Belgian Darts Open were confirmed on 7 February.
```

**参考标注**

已决、无风险（有资料依据）；参考风险片段为空。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |
| lb_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |
| binding_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |

生成记录 SHA256：`de424f6f47b3f7bef4f47a1ebf05fdef40f282316d698d5e1e5bd8d99ecf19e8`。

### 12. ragognize_train_1681__partial

- 输入行：`57`（从 0 开始）；类别：`time`；条件：`partial`。
- 问题：On what date were the seedings for the 2025 Belgian Darts Open confirmed?

**原始回答**

```text
1. The seedings for the 2025 Belgian Darts Open were not confirmed in the provided search results.
```

**参考标注**

拒答，排除主指标；没有可评参考风险片段。

冻结标注说明：Says confirmation information absent in given sources; supplies no invented date.

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0）<br><code>[6,11)</code> <code>&quot; seed&quot;</code>（BPE 3）<br><code>[47,52)</code> <code>&quot; were&quot;</code>（BPE 16）<br><code>[56,66)</code> <code>&quot; confirmed&quot;</code>（BPE 18）<br><code>[73,98)</code> <code>&quot; provided search results.&quot;</code>（BPE 21–24） | 0 / 7 / 1 | — | — | — |
| lb_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0）<br><code>[56,66)</code> <code>&quot; confirmed&quot;</code>（BPE 18）<br><code>[73,82)</code> <code>&quot; provided&quot;</code>（BPE 21）<br><code>[89,98)</code> <code>&quot; results.&quot;</code>（BPE 23,24） | 0 / 4 / 1 | — | — | — |
| binding_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0）<br><code>[73,82)</code> <code>&quot; provided&quot;</code>（BPE 21）<br><code>[89,98)</code> <code>&quot; results.&quot;</code>（BPE 23,24） | 0 / 3 / 1 | — | — | — |

生成记录 SHA256：`562058a13f6f254ceca37423a6fcf942b7adafbc5a246d89e3893af0875adaf6`。

### 13. ragognize_train_2020__complete

- 输入行：`64`（从 0 开始）；类别：`time`；条件：`complete`。
- 问题：On what date was Vishnu Priya released?

**原始回答**

```text
1. Vishnu Priya was released on 21 February 2025.
```

**参考标注**

已决、无风险（有资料依据）；参考风险片段为空。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |
| lb_fine | 无 | 0 / 0 / 0 | 0 | 0 | 0 |
| binding_fine | 无 | 0 / 0 / 0 | 0 | 0 | 0 |

生成记录 SHA256：`d864c61ac099091f6315d4224ba4e1872509caa8bd1a1b40532912e484e379f9`。

### 14. ragognize_train_2020__partial

- 输入行：`65`（从 0 开始）；类别：`time`；条件：`partial`。
- 问题：On what date was Vishnu Priya released?

**原始回答**

```text
1. Vishnu Priya was released on an unspecified date in 2025.
```

**参考标注**

拒答，排除主指标；没有可评参考风险片段。

冻结标注说明：Leaves date unspecified, adding no month/day to the given 2025 film year.

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0）<br><code>[28,46)</code> <code>&quot; on an unspecified&quot;</code>（BPE 8–10）<br><code>[54,56)</code> <code>&quot; 2&quot;</code>（BPE 13,14） | 0 / 5 / 1 | — | — | — |
| lb_fine | <code>[31,51)</code> <code>&quot; an unspecified date&quot;</code>（BPE 9–11）<br><code>[55,58)</code> <code>&quot;202&quot;</code>（BPE 14–16） | 0 / 6 / 0 | — | — | — |
| binding_fine | <code>[31,51)</code> <code>&quot; an unspecified date&quot;</code>（BPE 9–11）<br><code>[54,59)</code> <code>&quot; 2025&quot;</code>（BPE 13–17） | 0 / 7 / 1 | — | — | — |

生成记录 SHA256：`5f9668618e6e011abe40adbd93af0cb6a7b47f4924b1592cf50da2ffc4d80809`。

### 15. ragognize_train_2218__complete

- 输入行：`76`（从 0 开始）；类别：`time`；条件：`complete`。
- 问题：In what month and year did Luxbox acquire the sales rights to The Message?

**原始回答**

```text
1. Luxbox acquired the sales rights to The Message in January 2025.
```

**参考标注**

已决、无风险（有资料依据）；参考风险片段为空。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |
| lb_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |
| binding_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0）<br><code>[62,63)</code> <code>&quot;2&quot;</code>（BPE 14） | 1 / 1 / 0 | 0 | 1 | 0 |

生成记录 SHA256：`35812db9e5ca3d8c31e17d770bcb704017ce7ac620d5f76073c27584aa58e38f`。

### 16. ragognize_train_2218__partial

- 输入行：`77`（从 0 开始）；类别：`time`；条件：`partial`。
- 问题：In what month and year did Luxbox acquire the sales rights to The Message?

**原始回答**

```text
1. Luxbox acquired the sales rights to The Message in August 2024.
```

**参考标注**

已决、有风险；与当前资料的关系：<code>unsupported</code>。

- <code>[54,65)</code> <code>&quot;August 2024&quot;</code>

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0）<br><code>[6,9)</code> <code>&quot;box&quot;</code>（BPE 3）<br><code>[18,22)</code> <code>&quot; the&quot;</code>（BPE 5）<br><code>[38,53)</code> <code>&quot; The Message in&quot;</code>（BPE 9–11）<br><code>[60,62)</code> <code>&quot; 2&quot;</code>（BPE 13,14）<br><code>[63,64)</code> <code>&quot;2&quot;</code>（BPE 16） | 7 / 1 / 1 | 2 | 5 | 3 |
| lb_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0）<br><code>[53,62)</code> <code>&quot; August 2&quot;</code>（BPE 12,14）<br><code>[63,64)</code> <code>&quot;2&quot;</code>（BPE 16） | 3 / 1 / 0 | 3 | 0 | 2 |
| binding_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0）<br><code>[53,62)</code> <code>&quot; August 2&quot;</code>（BPE 12,14）<br><code>[63,64)</code> <code>&quot;2&quot;</code>（BPE 16） | 3 / 1 / 0 | 3 | 0 | 2 |

生成记录 SHA256：`153cfb22edc6b4fe49f3d5627ae3311e65ea139484f8b8a597e9e474d187c77b`。

### 17. ragognize_test_0130__complete

- 输入行：`82`（从 0 开始）；类别：`quantity`；条件：`complete`。
- 问题：Approximately how many people did Belden Tri-State Building Materials employ as of 2023?

**原始回答**

```text
1. As of 2023, Belden Tri-State Building Materials employs approximately 50 people.
```

**参考标注**

已决、无风险（有资料依据）；参考风险片段为空。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |
| lb_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |
| binding_fine | 无 | 0 / 0 / 0 | 0 | 0 | 0 |

生成记录 SHA256：`3da6325a13228879b0de54ef1c24beac0aa999a70fbd430f975637daddb9edb9`。

### 18. ragognize_test_0130__partial

- 输入行：`83`（从 0 开始）；类别：`quantity`；条件：`partial`。
- 问题：Approximately how many people did Belden Tri-State Building Materials employ as of 2023?

**原始回答**

```text
1. The search results do not provide specific employment numbers for Belden Tri-State Building Materials as of 2023.
```

**参考标注**

拒答，排除主指标；没有可评参考风险片段。

冻结标注说明：Declines the missing employee count.

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0）<br><code>[13,21)</code> <code>&quot; results&quot;</code>（BPE 4）<br><code>[64,68)</code> <code>&quot; for&quot;</code>（BPE 11）<br><code>[110,112)</code> <code>&quot; 2&quot;</code>（BPE 20,21） | 0 / 4 / 1 | — | — | — |
| lb_fine | <code>[24,45)</code> <code>&quot; not provide specific&quot;</code>（BPE 6–8）<br><code>[56,68)</code> <code>&quot; numbers for&quot;</code>（BPE 10,11） | 0 / 5 / 0 | — | — | — |
| binding_fine | <code>[24,45)</code> <code>&quot; not provide specific&quot;</code>（BPE 6–8）<br><code>[56,68)</code> <code>&quot; numbers for&quot;</code>（BPE 10,11） | 0 / 5 / 0 | — | — | — |

生成记录 SHA256：`51e477ae41120870f8c1e867a5d10bac83de2b2b01a881907defdbecda9b2c97`。

### 19. ragognize_test_0275__complete

- 输入行：`86`（从 0 开始）；类别：`quantity`；条件：`complete`。
- 问题：As of February 2025, how much did FireAid organizers expect proceeds from donations, sales and sponsorships to exceed?

**原始回答**

```text
1. As of February 2025, FireAid organizers expected proceeds from donations, sales, and sponsorships to exceed $100 million.
```

**参考标注**

已决、无风险（有资料依据）；参考风险片段为空。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | 无 | 0 / 0 / 0 | 0 | 0 | 0 |
| lb_fine | 无 | 0 / 0 / 0 | 0 | 0 | 0 |
| binding_fine | 无 | 0 / 0 / 0 | 0 | 0 | 0 |

生成记录 SHA256：`590e899b407a9b903f778f753298775c699a20cc29e469bbffdb54152cb581c8`。

### 20. ragognize_test_0275__partial

- 输入行：`87`（从 0 开始）；类别：`quantity`；条件：`partial`。
- 问题：As of February 2025, how much did FireAid organizers expect proceeds from donations, sales and sponsorships to exceed?

**原始回答**

```text
1. FireAid organizers expected proceeds from donations, sales, and sponsorships to exceed $1 million as of February 2025.
```

**参考标注**

已决、有风险；与当前资料的关系：<code>unsupported</code>。

- <code>[90,100)</code> <code>&quot;$1 million&quot;</code>

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0）<br><code>[82,92)</code> <code>&quot; exceed $1&quot;</code>（BPE 17–19） | 2 / 1 / 1 | 1 | 1 | 1 |
| lb_fine | 无 | 0 / 0 / 0 | 0 | 0 | 2 |
| binding_fine | 无 | 0 / 0 / 0 | 0 | 0 | 2 |

生成记录 SHA256：`4245132fcd2696890bb2017edd30fc2a5de4c74130270a6e8843930d0bb6b084`。

### 21. ragognize_train_1396__complete

- 输入行：`102`（从 0 开始）；类别：`quantity`；条件：`complete`。
- 问题：How many people attended the maiden edition of Adedoyin Oseni’s Evening of Worship in Sheffield?

**原始回答**

```text
1. The maiden edition of Adedoyin Oseni’s Evening of Worship in Sheffield recorded an attendance of 250 persons.
```

**参考标注**

已决、无风险（有资料依据）；参考风险片段为空。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | 无 | 0 / 0 / 0 | 0 | 0 | 0 |
| lb_fine | 无 | 0 / 0 / 0 | 0 | 0 | 0 |
| binding_fine | 无 | 0 / 0 / 0 | 0 | 0 | 0 |

生成记录 SHA256：`a1987ff06af3438d64625d11a7eaf428d40d88a043da08765e32dd1159882868`。

### 22. ragognize_train_1396__partial

- 输入行：`103`（从 0 开始）；类别：`quantity`；条件：`partial`。
- 问题：How many people attended the maiden edition of Adedoyin Oseni’s Evening of Worship in Sheffield?

**原始回答**

```text
1. The search results do not provide information about the number of people who attended the maiden edition of Adedoyin Oseni’s Evening of Worship in Sheffield.
```

**参考标注**

拒答，排除主指标；没有可评参考风险片段。

冻结标注说明：当前只有 Oseni 学历和另一酒店火灾人数，回答准确拒绝给该音乐会入场人数。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,2)</code> <code>&quot;1.&quot;</code>（BPE 0,1）<br><code>[13,21)</code> <code>&quot; results&quot;</code>（BPE 4）<br><code>[135,138)</code> <code>&quot; of&quot;</code>（BPE 29）<br><code>[146,159)</code> <code>&quot; in Sheffield&quot;</code>（BPE 31,32） | 0 / 5 / 1 | — | — | — |
| lb_fine | <code>[1,2)</code> <code>&quot;.&quot;</code>（BPE 1） | 0 / 0 / 1 | — | — | — |
| binding_fine | <code>[1,2)</code> <code>&quot;.&quot;</code>（BPE 1）<br><code>[24,54)</code> <code>&quot; not provide information about&quot;</code>（BPE 6–9） | 0 / 4 / 1 | — | — | — |

生成记录 SHA256：`34154e804ee18f8635cd16b8c534361ee5bb85598eb91a87dc1892b80bfe60a2`。

### 23. ragognize_test_1645__complete

- 输入行：`110`（从 0 开始）；类别：`quantity`；条件：`complete`。
- 问题：How many Ligier JS P325 cars were expected to be built before the 2025 racing season?

**原始回答**

```text
1. Twenty-five Ligier JS P325 cars were expected to be built before the 2025 racing season.
```

**参考标注**

已决、无风险（有资料依据）；参考风险片段为空。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | 无 | 0 / 0 / 0 | 0 | 0 | 0 |
| lb_fine | 无 | 0 / 0 / 0 | 0 | 0 | 0 |
| binding_fine | 无 | 0 / 0 / 0 | 0 | 0 | 0 |

生成记录 SHA256：`1905be6b12c53e1d9d77d5c8a9ad8754c946870f6b6fa4bd4d916d8234e847c6`。

### 24. ragognize_test_1645__partial

- 输入行：`111`（从 0 开始）；类别：`quantity`；条件：`partial`。
- 问题：How many Ligier JS P325 cars were expected to be built before the 2025 racing season?

**原始回答**

```text
1. No specific number of Ligier JS P325 cars were expected to be built before the 2025 racing season.
```

**参考标注**

未决，排除主指标；不强制标出风险片段。

冻结标注说明：No specific number ... were expected 可读为生产计划本身没有具体数量（当前未证），也可读为语境省略的未给预计数（拒答）；没有 in provided information 澄清，无法可靠锁定其主张，保留两种读法，不强圈 span。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0）<br><code>[49,61)</code> <code>&quot; expected to&quot;</code>（BPE 15,16）<br><code>[86,100)</code> <code>&quot; racing season&quot;</code>（BPE 26,27） | 0 / 5 / 0 | — | — | — |
| lb_fine | <code>[2,5)</code> <code>&quot; No&quot;</code>（BPE 2）<br><code>[14,21)</code> <code>&quot; number&quot;</code>（BPE 4） | 0 / 2 / 0 | — | — | — |
| binding_fine | <code>[2,5)</code> <code>&quot; No&quot;</code>（BPE 2）<br><code>[14,21)</code> <code>&quot; number&quot;</code>（BPE 4） | 0 / 2 / 0 | — | — | — |

生成记录 SHA256：`2efb892faafe2c91ab18753ca7e0164c985afccb8e0b2b60b9008c53db570017`。

### 25. ragognize_test_2095__complete

- 输入行：`114`（从 0 开始）；类别：`quantity`；条件：`complete`。
- 问题：How many hectares of new woodland were initially planned for the Western Forest?

**原始回答**

```text
1. The Western Forest initially planned for 2,500 hectares of new woodland.
```

**参考标注**

已决、无风险（有资料依据）；参考风险片段为空。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |
| lb_fine | 无 | 0 / 0 / 0 | 0 | 0 | 0 |
| binding_fine | 无 | 0 / 0 / 0 | 0 | 0 | 0 |

生成记录 SHA256：`b1854e7773556ad6a5b86784b06fbdf4b37875ca2464a5719735b756ecdf134e`。

### 26. ragognize_test_2095__partial

- 输入行：`115`（从 0 开始）；类别：`quantity`；条件：`partial`。
- 问题：How many hectares of new woodland were initially planned for the Western Forest?

**原始回答**

```text
1. The initial plan for the Western Forest included planting woodlands on 20,000 hectares.
```

**参考标注**

已决、有风险；与当前资料的关系：<code>unsupported</code>。

- <code>[74,89)</code> <code>&quot;20,000 hectares&quot;</code>

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,2)</code> <code>&quot;1.&quot;</code>（BPE 0,1）<br><code>[14,19)</code> <code>&quot; plan&quot;</code>（BPE 4）<br><code>[42,51)</code> <code>&quot; included&quot;</code>（BPE 9）<br><code>[60,65)</code> <code>&quot; wood&quot;</code>（BPE 11）<br><code>[70,78)</code> <code>&quot; on 20,0&quot;</code>（BPE 13–18）<br><code>[80,89)</code> <code>&quot; hectares&quot;</code>（BPE 21） | 8 / 1 / 3 | 4 | 4 | 2 |
| lb_fine | <code>[65,70)</code> <code>&quot;lands&quot;</code>（BPE 12）<br><code>[75,78)</code> <code>&quot;0,0&quot;</code>（BPE 16,18）<br><code>[80,89)</code> <code>&quot; hectares&quot;</code>（BPE 21） | 4 / 0 / 0 | 3 | 1 | 3 |
| binding_fine | <code>[65,70)</code> <code>&quot;lands&quot;</code>（BPE 12）<br><code>[73,74)</code> <code>&quot; &quot;</code>（BPE 14）<br><code>[75,78)</code> <code>&quot;0,0&quot;</code>（BPE 16,18）<br><code>[80,89)</code> <code>&quot; hectares&quot;</code>（BPE 21） | 4 / 0 / 1 | 3 | 1 | 3 |

生成记录 SHA256：`1d0b4ab3bc955ac5d0a120a139043caa4a796c78b91ce0772fc748a3efb330dc`。

### 27. ragognize_train_1861__complete

- 输入行：`148`（从 0 开始）；类别：`quantity`；条件：`complete`。
- 问题：In TROPION-Breast01, what proportion of the chemotherapy comparison group received eribulin rather than datopotamab deruxtecan?

**原始回答**

```text
1. In TROPION-Breast01, 60% of the chemotherapy comparison group received eribulin rather than datopotamab deruxtecan.
```

**参考标注**

已决、无风险（有资料依据）；参考风险片段为空。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0）<br><code>[26,34)</code> <code>&quot;% of the&quot;</code>（BPE 15–17） | 2 / 1 / 1 | 0 | 2 | 0 |
| lb_fine | <code>[7,10)</code> <code>&quot;ROP&quot;</code>（BPE 4） | 1 / 0 / 0 | 0 | 1 | 0 |
| binding_fine | 无 | 0 / 0 / 0 | 0 | 0 | 0 |

生成记录 SHA256：`ea99fe70ad3d48a13efab78ed5e42c85536baac498a5966490d02899c6622e9f`。

### 28. ragognize_train_1861__partial

- 输入行：`149`（从 0 开始）；类别：`quantity`；条件：`partial`。
- 问题：In TROPION-Breast01, what proportion of the chemotherapy comparison group received eribulin rather than datopotamab deruxtecan?

**原始回答**

```text
1. The search results do not provide information on the proportion of the chemotherapy comparison group receiving eribulin rather than datopotamab deruxtecan in TROPION-Breast01.
```

**参考标注**

拒答，排除主指标；没有可评参考风险片段。

冻结标注说明：准确拒绝给化疗分配比例；患者入组条件与地方选举投票率都不能回答。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,2)</code> <code>&quot;1.&quot;</code>（BPE 0,1）<br><code>[24,55)</code> <code>&quot; not provide information on the&quot;</code>（BPE 6–10）<br><code>[66,73)</code> <code>&quot; of the&quot;</code>（BPE 12,13）<br><code>[134,138)</code> <code>&quot; dat&quot;</code>（BPE 23）<br><code>[152,154)</code> <code>&quot;te&quot;</code>（BPE 29）<br><code>[162,165)</code> <code>&quot;ROP&quot;</code>（BPE 33）<br><code>[170,176)</code> <code>&quot;reast0&quot;</code>（BPE 36–38） | 0 / 14 / 1 | — | — | — |
| lb_fine | <code>[1,2)</code> <code>&quot;.&quot;</code>（BPE 1）<br><code>[24,51)</code> <code>&quot; not provide information on&quot;</code>（BPE 6–9）<br><code>[162,165)</code> <code>&quot;ROP&quot;</code>（BPE 33） | 0 / 5 / 1 | — | — | — |
| binding_fine | <code>[1,2)</code> <code>&quot;.&quot;</code>（BPE 1）<br><code>[24,51)</code> <code>&quot; not provide information on&quot;</code>（BPE 6–9）<br><code>[162,165)</code> <code>&quot;ROP&quot;</code>（BPE 33） | 0 / 5 / 1 | — | — | — |

生成记录 SHA256：`b92e5b8d0b3ff29b6958c761d534998f71fb03ffb3c1172fa9a059e5231f46a1`。

### 29. ragognize_train_1953__complete

- 输入行：`150`（从 0 开始）；类别：`quantity`；条件：`complete`。
- 问题：According to the Fiscal Policy Office study associated with the Integrated Entertainment Business Bill, to what amount was visitor spending projected to increase?

**原始回答**

```text
1. Visitor spending was projected to increase from 40,000 Baht to 60,000 Baht.
```

**参考标注**

已决、无风险（有资料依据）；参考风险片段为空。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | 无 | 0 / 0 / 0 | 0 | 0 | 0 |
| lb_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |
| binding_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |

生成记录 SHA256：`f73b524d7a86046645452f21a619232974895c6a1d578317e4561f2336b23ee0`。

### 30. ragognize_train_1953__partial

- 输入行：`151`（从 0 开始）；类别：`quantity`；条件：`partial`。
- 问题：According to the Fiscal Policy Office study associated with the Integrated Entertainment Business Bill, to what amount was visitor spending projected to increase?

**原始回答**

```text
I couldn't find a direct answer to the question about visitor spending projections in the provided search results. The search results do not contain information related to the Fiscal Policy Office study or visitor spending associated with the Integrated Entertainment Business Bill.
```

**参考标注**

拒答，排除主指标；没有可评参考风险片段。

冻结标注说明：两句均限定当前资料未给研究/访客消费，未说法案主体未出现。可见只有法案程序与球员收入，没有访客消费预测，属准确拒答。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;I&quot;</code>（BPE 0）<br><code>[8,10)</code> <code>&quot;&#x27;t&quot;</code>（BPE 2）<br><code>[15,34)</code> <code>&quot; a direct answer to&quot;</code>（BPE 4–7）<br><code>[38,53)</code> <code>&quot; question about&quot;</code>（BPE 9,10）<br><code>[82,85)</code> <code>&quot; in&quot;</code>（BPE 14）<br><code>[89,118)</code> <code>&quot; provided search results. The&quot;</code>（BPE 16,17,18,20）<br><code>[125,133)</code> <code>&quot; results&quot;</code>（BPE 22）<br><code>[136,175)</code> <code>&quot; not contain information related to the&quot;</code>（BPE 24–29）<br><code>[182,281)</code> <code>&quot; Policy Office study or visitor spending associated with the Integrated Entertainment Business Bill&quot;</code>（BPE 31–43） | 0 / 33 / 0 | — | — | — |
| lb_fine | <code>[0,1)</code> <code>&quot;I&quot;</code>（BPE 0）<br><code>[31,34)</code> <code>&quot; to&quot;</code>（BPE 7）<br><code>[160,171)</code> <code>&quot; related to&quot;</code>（BPE 27,28）<br><code>[222,233)</code> <code>&quot; associated&quot;</code>（BPE 37） | 0 / 5 / 0 | — | — | — |
| binding_fine | <code>[0,1)</code> <code>&quot;I&quot;</code>（BPE 0）<br><code>[24,34)</code> <code>&quot; answer to&quot;</code>（BPE 6,7）<br><code>[160,171)</code> <code>&quot; related to&quot;</code>（BPE 27,28）<br><code>[222,233)</code> <code>&quot; associated&quot;</code>（BPE 37） | 0 / 6 / 0 | — | — | — |

生成记录 SHA256：`298d6564e5aed0e1c8e53053d55581e625694c9e1a1bec530689c60996a2f0a5`。

### 31. ragognize_train_2295__complete

- 输入行：`158`（从 0 开始）；类别：`quantity`；条件：`complete`。
- 问题：How many votes did Reyna Walters-Morgan receive on the third ballot in her 2025 DNC vice-chair election?

**原始回答**

```text
1. Reyna Walters-Morgan received 210 votes on the third ballot in her 2025 DNC vice-chair election.
```

**参考标注**

已决、无风险（有资料依据）；参考风险片段为空。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |
| lb_fine | 无 | 0 / 0 / 0 | 0 | 0 | 0 |
| binding_fine | 无 | 0 / 0 / 0 | 0 | 0 | 0 |

生成记录 SHA256：`52e8f9d14471380f8b76d60695ff81ac68ee50ed40c36154ac3037492ead9a29`。

### 32. ragognize_train_2295__partial

- 输入行：`159`（从 0 开始）；类别：`quantity`；条件：`partial`。
- 问题：How many votes did Reyna Walters-Morgan receive on the third ballot in her 2025 DNC vice-chair election?

**原始回答**

```text
1. Reyna Walters-Morgan received 12,035 votes on the third ballot in her 2025 DNC vice-chair election.
```

**参考标注**

已决、有风险；与当前资料的关系：<code>unsupported</code>。

- <code>[33,45)</code> <code>&quot;12,035 votes&quot;</code>

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0）<br><code>[8,16)</code> <code>&quot; Walters&quot;</code>（BPE 4）<br><code>[18,34)</code> <code>&quot;organ received 1&quot;</code>（BPE 6–9）<br><code>[36,37)</code> <code>&quot;0&quot;</code>（BPE 12）<br><code>[52,65)</code> <code>&quot; third ballot&quot;</code>（BPE 18,19）<br><code>[86,92)</code> <code>&quot;-chair&quot;</code>（BPE 29） | 8 / 1 / 1 | 2 | 6 | 4 |
| lb_fine | <code>[33,37)</code> <code>&quot;12,0&quot;</code>（BPE 9,10,12）<br><code>[39,45)</code> <code>&quot; votes&quot;</code>（BPE 15）<br><code>[58,65)</code> <code>&quot; ballot&quot;</code>（BPE 19） | 5 / 0 / 0 | 4 | 1 | 2 |
| binding_fine | <code>[33,38)</code> <code>&quot;12,03&quot;</code>（BPE 9–13）<br><code>[39,45)</code> <code>&quot; votes&quot;</code>（BPE 15）<br><code>[58,65)</code> <code>&quot; ballot&quot;</code>（BPE 19）<br><code>[72,73)</code> <code>&quot; &quot;</code>（BPE 22） | 6 / 0 / 2 | 5 | 1 | 1 |

生成记录 SHA256：`e14203ef1afe69be7521f4778a4562e782340cdd4fe96806d215e3e5d3085fc2`。

### 33. ragognize_test_0452__complete

- 输入行：`162`（从 0 开始）；类别：`location`；条件：`complete`。
- 问题：In which city was Matt Smiley born?

**原始回答**

```text
1. Matt Smiley was born in Montreal, Quebec.
```

**参考标注**

已决、无风险（有资料依据）；参考风险片段为空。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |
| lb_fine | 无 | 0 / 0 / 0 | 0 | 0 | 0 |
| binding_fine | 无 | 0 / 0 / 0 | 0 | 0 | 0 |

生成记录 SHA256：`a6dd2b0f13ffd26e641ed397ceca2c2654baac303d38625fcb1d261191b03dcb`。

### 34. ragognize_test_0452__partial

- 输入行：`163`（从 0 开始）；类别：`location`；条件：`partial`。
- 问题：In which city was Matt Smiley born?

**原始回答**

```text
1. Matt Smiley was born in Toronto.
```

**参考标注**

已决、有风险；与当前资料的关系：<code>unsupported</code>。

- <code>[27,34)</code> <code>&quot;Toronto&quot;</code>

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0）<br><code>[10,26)</code> <code>&quot;iley was born in&quot;</code>（BPE 4–7） | 4 / 1 / 0 | 0 | 4 | 1 |
| lb_fine | <code>[26,34)</code> <code>&quot; Toronto&quot;</code>（BPE 8） | 1 / 0 / 0 | 1 | 0 | 0 |
| binding_fine | <code>[26,34)</code> <code>&quot; Toronto&quot;</code>（BPE 8） | 1 / 0 / 0 | 1 | 0 | 0 |

生成记录 SHA256：`296fa00c07cbc87f00a10eacd34397768f14072fdbd76eed99ca492b407417e8`。

### 35. ragognize_test_0889__complete

- 输入行：`170`（从 0 开始）；类别：`location`；条件：`complete`。
- 问题：In which Nigerian state is Smart City F.C. based?

**原始回答**

```text
1. Smart City F.C. is based in Lagos state.
```

**参考标注**

已决、无风险（有资料依据）；参考风险片段为空。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |
| lb_fine | 无 | 0 / 0 / 0 | 0 | 0 | 0 |
| binding_fine | 无 | 0 / 0 / 0 | 0 | 0 | 0 |

生成记录 SHA256：`e3650526dc1d84703a1a77b925916fe1078c4caf749716d7fe6b6912e274c7a4`。

### 36. ragognize_test_0889__partial

- 输入行：`171`（从 0 开始）；类别：`location`；条件：`partial`。
- 问题：In which Nigerian state is Smart City F.C. based?

**原始回答**

```text
1. Smart City F.C. is based in a Nigerian state, but the specific state is not mentioned in the provided search results.
```

**参考标注**

拒答，排除主指标；没有可评参考风险片段。

冻结标注说明：只重复问题限定的 Nigerian state 并说明具体州名未给，无新增州名；另一美国画家州名未被移接。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0）<br><code>[18,21)</code> <code>&quot; is&quot;</code>（BPE 7）<br><code>[27,32)</code> <code>&quot; in a&quot;</code>（BPE 9,10）<br><code>[41,48)</code> <code>&quot; state,&quot;</code>（BPE 12,13）<br><code>[52,56)</code> <code>&quot; the&quot;</code>（BPE 15）<br><code>[65,78)</code> <code>&quot; state is not&quot;</code>（BPE 17–19）<br><code>[104,119)</code> <code>&quot; search results&quot;</code>（BPE 24,25） | 0 / 11 / 1 | — | — | — |
| lb_fine | <code>[30,41)</code> <code>&quot; a Nigerian&quot;</code>（BPE 10,11）<br><code>[52,56)</code> <code>&quot; the&quot;</code>（BPE 15） | 0 / 3 / 0 | — | — | — |
| binding_fine | <code>[30,47)</code> <code>&quot; a Nigerian state&quot;</code>（BPE 10–12）<br><code>[52,65)</code> <code>&quot; the specific&quot;</code>（BPE 15,16） | 0 / 5 / 0 | — | — | — |

生成记录 SHA256：`b93b3aa901d48206483abd69e2a256bb7880e426ae54e28d7615d628037b71e3`。

### 37. ragognize_test_1420__complete

- 输入行：`182`（从 0 开始）；类别：`location`；条件：`complete`。
- 问题：From which metropolitan area is Sean Curran of the Secret Service?

**原始回答**

```text
1. Sean Curran of the Secret Service is from the New York City area.
```

**参考标注**

已决、无风险（有资料依据）；参考风险片段为空。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | 无 | 0 / 0 / 0 | 0 | 0 | 0 |
| lb_fine | 无 | 0 / 0 / 0 | 0 | 0 | 0 |
| binding_fine | 无 | 0 / 0 / 0 | 0 | 0 | 0 |

生成记录 SHA256：`416269acb446b1bf3f1b30bcbcaa4feb3ff507683b7fb7dd6c6c897662fda089`。

### 38. ragognize_test_1420__partial

- 输入行：`183`（从 0 开始）；类别：`location`；条件：`partial`。
- 问题：From which metropolitan area is Sean Curran of the Secret Service?

**原始回答**

```text
1. Sean Curran of the Secret Service is from the Washington, D.C. metropolitan area.
```

**参考标注**

已决、有风险；与当前资料的关系：<code>unsupported</code>。

- <code>[49,83)</code> <code>&quot;Washington, D.C. metropolitan area&quot;</code>

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[64,65)</code> <code>&quot;.&quot;</code>（BPE 16） | 0 / 0 / 1 | 0 | 0 | 5 |
| lb_fine | <code>[48,59)</code> <code>&quot; Washington&quot;</code>（BPE 12） | 1 / 0 / 0 | 1 | 0 | 4 |
| binding_fine | <code>[48,59)</code> <code>&quot; Washington&quot;</code>（BPE 12） | 1 / 0 / 0 | 1 | 0 | 4 |

生成记录 SHA256：`5ff937612a4c5a614ab432146c651a64b37e90d4d325a430f91f4b710ced3c44`。

### 39. ragognize_test_1866__complete

- 输入行：`190`（从 0 开始）；类别：`location`；条件：`complete`。
- 问题：In which city was Rose BC based for the inaugural 2025 Unrivaled season?

**原始回答**

```text
1. Rose BC was based in Miami, Florida for the inaugural 2025 Unrivaled season.
```

**参考标注**

已决、无风险（有资料依据）；参考风险片段为空。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |
| lb_fine | 无 | 0 / 0 / 0 | 0 | 0 | 0 |
| binding_fine | 无 | 0 / 0 / 0 | 0 | 0 | 0 |

生成记录 SHA256：`fafab8e6eea096b710f64fd6406b8383088f535edb3ab15529d284d7ee7e34ca`。

### 40. ragognize_test_1866__partial

- 输入行：`191`（从 0 开始）；类别：`location`；条件：`partial`。
- 问题：In which city was Rose BC based for the inaugural 2025 Unrivaled season?

**原始回答**

```text
1. Rose BC was based in an unspecified city for the inaugural 2025 Unrivaled season.
```

**参考标注**

拒答，排除主指标；没有可评参考风险片段。

冻结标注说明：unspecified city 没有给出任何城名，语境是未能确定所问城市，而非断言俱乐部事实上没有城市。当前资料未给城市，准确拒答。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,2)</code> <code>&quot;1.&quot;</code>（BPE 0,1）<br><code>[23,26)</code> <code>&quot; an&quot;</code>（BPE 7） | 0 / 2 / 1 | — | — | — |
| lb_fine | <code>[26,38)</code> <code>&quot; unspecified&quot;</code>（BPE 8） | 0 / 1 / 0 | — | — | — |
| binding_fine | <code>[26,38)</code> <code>&quot; unspecified&quot;</code>（BPE 8） | 0 / 1 / 0 | — | — | — |

生成记录 SHA256：`929df71792c854bcd3feb4648537db20e0ada85cf0fdf090c8846b3ec94516be`。

### 41. ragognize_test_2293__complete

- 输入行：`200`（从 0 开始）；类别：`location`；条件：`complete`。
- 问题：Which geographic region is the focus of Anjal Prakash’s research on water and climate issues?

**原始回答**

```text
1. The geographic region focus of Anjal Prakash’s research on water and climate issues is South Asia.
```

**参考标注**

已决、无风险（有资料依据）；参考风险片段为空。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |
| lb_fine | 无 | 0 / 0 / 0 | 0 | 0 | 0 |
| binding_fine | 无 | 0 / 0 / 0 | 0 | 0 | 0 |

生成记录 SHA256：`e7fabd5c44a0825c89a9f39e95f0ee6523524a3db60bc28e87696aa32fe99f22`。

### 42. ragognize_test_2293__partial

- 输入行：`201`（从 0 开始）；类别：`location`；条件：`partial`。
- 问题：Which geographic region is the focus of Anjal Prakash’s research on water and climate issues?

**原始回答**

```text
1. Anjal Prakash’s research on water and climate issues focuses on the Asia-Pacific region.
```

**参考标注**

已决、有风险；与当前资料的关系：<code>unsupported</code>。

- <code>[71,83)</code> <code>&quot;Asia-Pacific&quot;</code>

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0）<br><code>[8,36)</code> <code>&quot; Prakash’s research on water&quot;</code>（BPE 4–9）<br><code>[66,83)</code> <code>&quot; the Asia-Pacific&quot;</code>（BPE 15–17） | 9 / 1 / 0 | 2 | 7 | 0 |
| lb_fine | <code>[66,90)</code> <code>&quot; the Asia-Pacific region&quot;</code>（BPE 15–18） | 4 / 0 / 0 | 2 | 2 | 0 |
| binding_fine | <code>[66,90)</code> <code>&quot; the Asia-Pacific region&quot;</code>（BPE 15–18） | 4 / 0 / 0 | 2 | 2 | 0 |

生成记录 SHA256：`aee33a7f71cb7c2ccf0cb30f85439f79f540817a55ee18558b97969d6d11e60e`。

### 43. ragognize_train_0845__complete

- 输入行：`212`（从 0 开始）；类别：`location`；条件：`complete`。
- 问题：In which city did Professor Nuhu Yaqub die?

**原始回答**

```text
1. Professor Nuhu Yaqub died in Abuja.
```

**参考标注**

已决、无风险（有资料依据）；参考风险片段为空。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |
| lb_fine | 无 | 0 / 0 / 0 | 0 | 0 | 0 |
| binding_fine | <code>[31,35)</code> <code>&quot; Abu&quot;</code>（BPE 11） | 1 / 0 / 0 | 0 | 1 | 0 |

生成记录 SHA256：`42f8ceab7893f1730e335848605dbdb979eddbcd4b852750d1378bc6d9e4a162`。

### 44. ragognize_train_0845__partial

- 输入行：`213`（从 0 开始）；类别：`location`；条件：`partial`。
- 问题：In which city did Professor Nuhu Yaqub die?

**原始回答**

```text
1. Professor Nuhu Yaqub died in Quezon City.
```

**参考标注**

已决、有风险；与当前资料的关系：<code>unsupported</code>。

- <code>[32,43)</code> <code>&quot;Quezon City&quot;</code>

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0）<br><code>[28,31)</code> <code>&quot; in&quot;</code>（BPE 10） | 1 / 1 / 0 | 0 | 1 | 3 |
| lb_fine | <code>[31,43)</code> <code>&quot; Quezon City&quot;</code>（BPE 11–13） | 3 / 0 / 0 | 3 | 0 | 0 |
| binding_fine | <code>[17,19)</code> <code>&quot; Y&quot;</code>（BPE 6）<br><code>[31,43)</code> <code>&quot; Quezon City&quot;</code>（BPE 11–13） | 4 / 0 / 0 | 3 | 1 | 0 |

生成记录 SHA256：`9d9a580d9e89f5a716b8ffcf40b721a9e22348300c75b3029eceb324ccaa7b66`。

### 45. ragognize_train_1279__complete

- 输入行：`224`（从 0 开始）；类别：`location`；条件：`complete`。
- 问题：In which New York City borough was the Halleluya Festival’s 2025 edition scheduled to take place?

**原始回答**

```text
1. The Halleluya Festival’s 2025 edition was scheduled to take place in Brooklyn.
```

**参考标注**

已决、无风险（有资料依据）；参考风险片段为空。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |
| lb_fine | 无 | 0 / 0 / 0 | 0 | 0 | 0 |
| binding_fine | 无 | 0 / 0 / 0 | 0 | 0 | 0 |

生成记录 SHA256：`463f9a21b1cadd0f772df1becf29225b61b5fedf452a45ceb2fac31a64dc3a70`。

### 46. ragognize_train_1279__partial

- 输入行：`225`（从 0 开始）；类别：`location`；条件：`partial`。
- 问题：In which New York City borough was the Halleluya Festival’s 2025 edition scheduled to take place?

**原始回答**

```text
The search results provided do not contain information about the 2025 edition of the Halleluya Festival or its location in New York City. Therefore, I cannot answer the question based on the given information.
```

**参考标注**

拒答，排除主指标；没有可评参考风险片段。

冻结标注说明：准确说明资料没有2025纽约场次及具体地点；未否认资料提到一般节庆，也没有给猜测地点。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[3,27)</code> <code>&quot; search results provided&quot;</code>（BPE 1–3）<br><code>[30,42)</code> <code>&quot; not contain&quot;</code>（BPE 5,6）<br><code>[64,65)</code> <code>&quot; &quot;</code>（BPE 10）<br><code>[67,68)</code> <code>&quot;2&quot;</code>（BPE 13）<br><code>[103,119)</code> <code>&quot; or its location&quot;</code>（BPE 22–24）<br><code>[126,136)</code> <code>&quot; York City&quot;</code>（BPE 27,28）<br><code>[148,196)</code> <code>&quot; I cannot answer the question based on the given&quot;</code>（BPE 32–40） | 0 / 20 / 1 | — | — | — |
| lb_fine | <code>[30,60)</code> <code>&quot; not contain information about&quot;</code>（BPE 5–8）<br><code>[94,119)</code> <code>&quot; Festival or its location&quot;</code>（BPE 21–24）<br><code>[126,136)</code> <code>&quot; York City&quot;</code>（BPE 27,28）<br><code>[157,168)</code> <code>&quot; answer the&quot;</code>（BPE 34,35）<br><code>[177,183)</code> <code>&quot; based&quot;</code>（BPE 37） | 0 / 13 / 0 | — | — | — |
| binding_fine | <code>[18,60)</code> <code>&quot; provided do not contain information about&quot;</code>（BPE 3–8）<br><code>[94,119)</code> <code>&quot; Festival or its location&quot;</code>（BPE 21–24）<br><code>[126,136)</code> <code>&quot; York City&quot;</code>（BPE 27,28）<br><code>[157,168)</code> <code>&quot; answer the&quot;</code>（BPE 34,35）<br><code>[177,183)</code> <code>&quot; based&quot;</code>（BPE 37） | 0 / 15 / 0 | — | — | — |

生成记录 SHA256：`7d01b1793d9122af4524c00dfacf8d3ffc7cd1ad005cd90b6c192007bfdbf7ee`。

### 47. ragognize_train_2328__complete

- 输入行：`238`（从 0 开始）；类别：`location`；条件：`complete`。
- 问题：From which residential area was NM Ziaul Alam detained in March 2025?

**原始回答**

```text
1. NM Ziaul Alam was detained from the Panchlaish Residential area in March 2025.
```

**参考标注**

已决、无风险（有资料依据）；参考风险片段为空。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |
| lb_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |
| binding_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |

生成记录 SHA256：`b1c0eabf954cffd61a6318d0d46d767e68acd30e98711e4ede1fb2e0c011c39b`。

### 48. ragognize_train_2328__partial

- 输入行：`239`（从 0 开始）；类别：`location`；条件：`partial`。
- 问题：From which residential area was NM Ziaul Alam detained in March 2025?

**原始回答**

```text
There is no information provided about NM Ziaul Alam being detained in March 2025 or his residential area.
```

**参考标注**

拒答，排除主指标；没有可评参考风险片段。

冻结标注说明：当前只有退休官职和2015–2019任职情况，没有拘捕地点/事件信息，回答准确拒绝填值，未否认资料一般提及Alam。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,11)</code> <code>&quot;There is no&quot;</code>（BPE 0–2）<br><code>[23,32)</code> <code>&quot; provided&quot;</code>（BPE 4）<br><code>[77,78)</code> <code>&quot;2&quot;</code>（BPE 16）<br><code>[100,105)</code> <code>&quot; area&quot;</code>（BPE 23） | 0 / 6 / 0 | — | — | — |
| lb_fine | <code>[0,5)</code> <code>&quot;There&quot;</code>（BPE 0）<br><code>[38,41)</code> <code>&quot; NM&quot;</code>（BPE 6）<br><code>[47,52)</code> <code>&quot; Alam&quot;</code>（BPE 10） | 0 / 3 / 0 | — | — | — |
| binding_fine | <code>[0,5)</code> <code>&quot;There&quot;</code>（BPE 0）<br><code>[8,11)</code> <code>&quot; no&quot;</code>（BPE 2）<br><code>[23,41)</code> <code>&quot; provided about NM&quot;</code>（BPE 4–6）<br><code>[47,67)</code> <code>&quot; Alam being detained&quot;</code>（BPE 10–12）<br><code>[76,77)</code> <code>&quot; &quot;</code>（BPE 15） | 0 / 8 / 1 | — | — | — |

生成记录 SHA256：`e06e5fdf77a91632585e80c3c6567ef7b3cdbba5325e3cd9e859a457529a326a`。

### 49. ragognize_test_1722__complete

- 输入行：`248`（从 0 开始）；类别：`relation`；条件：`complete`。
- 问题：Which two writers does Tegwen Bruce-Deans cite as inspirations for her work?

**原始回答**

```text
1. Tegwen Bruce-Deans cites Mererid Hopwood and Ocean Vuong as inspirations for her work.
```

**参考标注**

已决、无风险（有资料依据）；参考风险片段为空。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |
| lb_fine | 无 | 0 / 0 / 0 | 0 | 0 | 0 |
| binding_fine | 无 | 0 / 0 / 0 | 0 | 0 | 0 |

生成记录 SHA256：`4956f059c0e93606c404c0aee662311f326b95e40e201f2d73eedfdde08b155f`。

### 50. ragognize_test_1722__partial

- 输入行：`249`（从 0 开始）；类别：`relation`；条件：`partial`。
- 问题：Which two writers does Tegwen Bruce-Deans cite as inspirations for her work?

**原始回答**

```text
Tegwen Bruce-Deans cites Marty Zambotto and Go-Jo as inspirations for her work.
```

**参考标注**

已决、有风险；与当前资料的关系：<code>unsupported</code>。

- <code>[25,39)</code> <code>&quot;Marty Zambotto&quot;</code>
- <code>[44,49)</code> <code>&quot;Go-Jo&quot;</code>

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,3)</code> <code>&quot;Teg&quot;</code>（BPE 0,1）<br><code>[12,30)</code> <code>&quot;-Deans cites Marty&quot;</code>（BPE 4–8）<br><code>[39,46)</code> <code>&quot; and Go&quot;</code>（BPE 12,13）<br><code>[52,65)</code> <code>&quot; inspirations&quot;</code>（BPE 17,18）<br><code>[69,73)</code> <code>&quot; her&quot;</code>（BPE 20） | 11 / 0 / 1 | 2 | 9 | 5 |
| lb_fine | <code>[0,1)</code> <code>&quot;T&quot;</code>（BPE 0）<br><code>[24,32)</code> <code>&quot; Marty Z&quot;</code>（BPE 8,9）<br><code>[39,43)</code> <code>&quot; and&quot;</code>（BPE 12）<br><code>[48,49)</code> <code>&quot;o&quot;</code>（BPE 15）<br><code>[52,59)</code> <code>&quot; inspir&quot;</code>（BPE 17） | 6 / 0 / 0 | 3 | 3 | 4 |
| binding_fine | <code>[0,1)</code> <code>&quot;T&quot;</code>（BPE 0）<br><code>[24,32)</code> <code>&quot; Marty Z&quot;</code>（BPE 8,9）<br><code>[39,46)</code> <code>&quot; and Go&quot;</code>（BPE 12,13）<br><code>[48,49)</code> <code>&quot;o&quot;</code>（BPE 15）<br><code>[52,59)</code> <code>&quot; inspir&quot;</code>（BPE 17）<br><code>[78,79)</code> <code>&quot;.&quot;</code>（BPE 22） | 7 / 0 / 1 | 4 | 3 | 3 |

生成记录 SHA256：`3d45745a51d9c14414067a26abd814c68b4fea851b3ad3333238e921c425e6dc`。

### 51. ragognize_train_0151__complete

- 输入行：`256`（从 0 开始）；类别：`relation`；条件：`complete`。
- 问题：What roles does Stacy Sims hold at Osmo Nutrition?

**原始回答**

```text
1. Stacy Sims holds the roles of co-founder and CEO at Osmo Nutrition.
```

**参考标注**

已决、无风险（有资料依据）；参考风险片段为空。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |
| lb_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |
| binding_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |

生成记录 SHA256：`a57ee18d944959e9367f1e8621b90e5653ba2f9e9d0a924a9fd00f72dd660889`。

### 52. ragognize_train_0151__partial

- 输入行：`257`（从 0 开始）；类别：`relation`；条件：`partial`。
- 问题：What roles does Stacy Sims hold at Osmo Nutrition?

**原始回答**

```text
1. Stacy Sims holds the roles of exercise physiologist, nutrition scientist, author, and women's health and fitness advocate at Osmo Nutrition.
```

**参考标注**

已决、有风险；与当前资料的关系：<code>unsupported</code>。

- <code>[33,54)</code> <code>&quot;exercise physiologist&quot;</code>
- <code>[56,75)</code> <code>&quot;nutrition scientist&quot;</code>
- <code>[77,83)</code> <code>&quot;author&quot;</code>
- <code>[89,124)</code> <code>&quot;women&#x27;s health and fitness advocate&quot;</code>

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0）<br><code>[32,41)</code> <code>&quot; exercise&quot;</code>（BPE 8）<br><code>[124,127)</code> <code>&quot; at&quot;</code>（BPE 25） | 2 / 1 / 0 | 1 | 1 | 12 |
| lb_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 13 |
| binding_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 13 |

生成记录 SHA256：`deb1f99e61ff410b36798812c41da73beeefd926be5744f6bbdc9f1ff91a05f5`。

### 53. ragognize_train_0380__complete

- 输入行：`264`（从 0 开始）；类别：`relation`；条件：`complete`。
- 问题：Who founded the Ukrainian organization Impact Force in 2021?

**原始回答**

```text
1. Nina Levchuk and Olga Diakova founded the Ukrainian organization Impact Force in 2021.
```

**参考标注**

已决、无风险（有资料依据）；参考风险片段为空。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |
| lb_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |
| binding_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |

生成记录 SHA256：`c2e9c0f29b393c69a1b65550b425c823c106471185ad2ddb7e27ab02233360ba`。

### 54. ragognize_train_0380__partial

- 输入行：`265`（从 0 开始）；类别：`relation`；条件：`partial`。
- 问题：Who founded the Ukrainian organization Impact Force in 2021?

**原始回答**

```text
1. Brahma Prakash Singh founded the Ukrainian organization Impact Force in 2021.
```

**参考标注**

已决、有风险；与当前资料的关系：<code>unsupported</code>。

- <code>[3,23)</code> <code>&quot;Brahma Prakash Singh&quot;</code>

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0）<br><code>[58,65)</code> <code>&quot; Impact&quot;</code>（BPE 11） | 1 / 1 / 0 | 0 | 1 | 5 |
| lb_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 5 |
| binding_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 5 |

生成记录 SHA256：`c782e878d63bbe46916b32080141e2aaed5f0407379ee7290b1495265f777c8a`。

### 55. ragognize_train_0642__complete

- 输入行：`276`（从 0 开始）；类别：`relation`；条件：`complete`。
- 问题：Who played bass in Hirax's live lineup for the Faster than Death tour?

**原始回答**

```text
1. Jose Gonzalez played bass in Hirax's live lineup for the Faster than Death tour.
```

**参考标注**

已决、无风险（有资料依据）；参考风险片段为空。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |
| lb_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |
| binding_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |

生成记录 SHA256：`e42cd0ac21109bb45b8e37e85a3e60aede8e5f86dac69c8be14fae4900d1f644`。

### 56. ragognize_train_0642__partial

- 输入行：`277`（从 0 开始）；类别：`relation`；条件：`partial`。
- 问题：Who played bass in Hirax's live lineup for the Faster than Death tour?

**原始回答**

```text
1. The search results do not provide information about who played bass in Hirax's live lineup for the Faster than Death tour.
```

**参考标注**

拒答，排除主指标；没有可评参考风险片段。

冻结标注说明：当前只给专辑/单曲发布，未给现场贝斯手，回答准确拒答；另一书会人名不能补证。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,2)</code> <code>&quot;1.&quot;</code>（BPE 0,1）<br><code>[28,36)</code> <code>&quot; provide&quot;</code>（BPE 7）<br><code>[77,81)</code> <code>&quot;ax&#x27;s&quot;</code>（BPE 15,16）<br><code>[101,108)</code> <code>&quot; Faster&quot;</code>（BPE 21）<br><code>[119,124)</code> <code>&quot; tour&quot;</code>（BPE 24） | 0 / 6 / 1 | — | — | — |
| lb_fine | <code>[0,2)</code> <code>&quot;1.&quot;</code>（BPE 0,1）<br><code>[21,48)</code> <code>&quot; do not provide information&quot;</code>（BPE 5–8）<br><code>[65,70)</code> <code>&quot; bass&quot;</code>（BPE 12） | 0 / 6 / 1 | — | — | — |
| binding_fine | <code>[0,2)</code> <code>&quot;1.&quot;</code>（BPE 0,1）<br><code>[24,48)</code> <code>&quot; not provide information&quot;</code>（BPE 6–8）<br><code>[65,70)</code> <code>&quot; bass&quot;</code>（BPE 12） | 0 / 5 / 1 | — | — | — |

生成记录 SHA256：`00d92a5e436bfbe63e0e9d78550025f3f75de5195203e4c8227870e8f6a16e0b`。

### 57. ragognize_train_1219__complete

- 输入行：`292`（从 0 开始）；类别：`relation`；条件：`complete`。
- 问题：Which company acquired Sara Khaki's documentary Cutting Through Rocks?

**原始回答**

```text
1. Autlook Filmsales acquired Sara Khaki's documentary Cutting Through Rocks.
```

**参考标注**

已决、无风险（有资料依据）；参考风险片段为空。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |
| lb_fine | <code>[0,6)</code> <code>&quot;1. Aut&quot;</code>（BPE 0,2） | 1 / 1 / 0 | 0 | 1 | 0 |
| binding_fine | <code>[0,6)</code> <code>&quot;1. Aut&quot;</code>（BPE 0,2） | 1 / 1 / 0 | 0 | 1 | 0 |

生成记录 SHA256：`70a221d5c87ba863f669fb478fd1f7cebcdeef263625cb9cd41da88644ff2583`。

### 58. ragognize_train_1219__partial

- 输入行：`293`（从 0 开始）；类别：`relation`；条件：`partial`。
- 问题：Which company acquired Sara Khaki's documentary Cutting Through Rocks?

**原始回答**

```text
1. Yalla Parkour acquired Sara Khaki's documentary Cutting Through Rocks.
```

**参考标注**

已决、有风险；与当前资料的关系：<code>unsupported</code>。

- <code>[3,16)</code> <code>&quot;Yalla Parkour&quot;</code>

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,4)</code> <code>&quot;1. Y&quot;</code>（BPE 0–2） | 1 / 1 / 1 | 1 | 0 | 3 |
| lb_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0）<br><code>[4,8)</code> <code>&quot;alla&quot;</code>（BPE 3） | 1 / 1 / 0 | 1 | 0 | 3 |
| binding_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0）<br><code>[4,8)</code> <code>&quot;alla&quot;</code>（BPE 3） | 1 / 1 / 0 | 1 | 0 | 3 |

生成记录 SHA256：`39996b2f8386f483db009c1b54b6c409a5bd6928c7f7bdd260c5b1c146f7065c`。

### 59. ragognize_train_1657__complete

- 输入行：`306`（从 0 开始）；类别：`relation`；条件：`complete`。
- 问题：What was the name of the law firm John Tracy Cagas established in Digos?

**原始回答**

```text
1. John Tracy Cagas established the law firm Cagas Law Office in Digos.
```

**参考标注**

已决、无风险（有资料依据）；参考风险片段为空。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |
| lb_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |
| binding_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |

生成记录 SHA256：`622463c67763546d5583bc11c6f89ab3d083191141b2dfd732ed2696933423e5`。

### 60. ragognize_train_1657__partial

- 输入行：`307`（从 0 开始）；类别：`relation`；条件：`partial`。
- 问题：What was the name of the law firm John Tracy Cagas established in Digos?

**原始回答**

```text
1. John Tracy Cagas established the law firm Cagas & Cagas in Digos.
```

**参考标注**

已决、有风险；与当前资料的关系：<code>unsupported</code>。

- <code>[45,58)</code> <code>&quot;Cagas &amp; Cagas&quot;</code>

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,2)</code> <code>&quot;1.&quot;</code>（BPE 0,1）<br><code>[15,19)</code> <code>&quot;agas&quot;</code>（BPE 5）<br><code>[35,44)</code> <code>&quot; law firm&quot;</code>（BPE 8,9）<br><code>[46,58)</code> <code>&quot;agas &amp; Cagas&quot;</code>（BPE 11–14） | 6 / 1 / 2 | 3 | 3 | 1 |
| lb_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0）<br><code>[46,58)</code> <code>&quot;agas &amp; Cagas&quot;</code>（BPE 11–14）<br><code>[63,67)</code> <code>&quot;igos&quot;</code>（BPE 17） | 4 / 1 / 1 | 3 | 1 | 1 |
| binding_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0）<br><code>[44,58)</code> <code>&quot; Cagas &amp; Cagas&quot;</code>（BPE 10–14）<br><code>[63,67)</code> <code>&quot;igos&quot;</code>（BPE 17） | 5 / 1 / 1 | 4 | 1 | 0 |

生成记录 SHA256：`01c19c007d7d7207d6acecfaf22d44fe6a99ba3302028ff66aa279e58143f427`。

### 61. ragognize_train_1664__complete

- 输入行：`310`（从 0 开始）；类别：`relation`；条件：`complete`。
- 问题：Which award did Chain Reactions Africa receive at LaPRIGA in 2024?

**原始回答**

```text
1. Chain Reactions Africa received the Public Sector PR Campaign of the Year award at LaPRIGA in 2024.
```

**参考标注**

已决、无风险（有资料依据）；参考风险片段为空。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |
| lb_fine | <code>[45,52)</code> <code>&quot; Sector&quot;</code>（BPE 9） | 1 / 0 / 0 | 0 | 1 | 0 |
| binding_fine | <code>[45,52)</code> <code>&quot; Sector&quot;</code>（BPE 9） | 1 / 0 / 0 | 0 | 1 | 0 |

生成记录 SHA256：`75269b99d1e1008583d89134cc9e65ce1d041c82defb448f10851bffc74f08bc`。

### 62. ragognize_train_1664__partial

- 输入行：`311`（从 0 开始）；类别：`relation`；条件：`partial`。
- 问题：Which award did Chain Reactions Africa receive at LaPRIGA in 2024?

**原始回答**

```text
1. Chain Reactions Africa received the Outstanding PR Agency of the Year award at the MarketingEdge Awards 2024.
```

**参考标注**

已决、无风险（有资料依据）；参考风险片段为空。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |
| lb_fine | 无 | 0 / 0 / 0 | 0 | 0 | 0 |
| binding_fine | 无 | 0 / 0 / 0 | 0 | 0 | 0 |

生成记录 SHA256：`fee2ca867c9f134c94f951d79c7bd2932deb75d28f579e479f384e6d0a3f3edb`。

### 63. ragognize_train_1773__complete

- 输入行：`312`（从 0 开始）；类别：`relation`；条件：`complete`。
- 问题：Which organization has Jonathan Hill on its Board of Directors?

**原始回答**

```text
1. Jonathan Hill serves on the Board of Directors for the Literary Arts.
```

**参考标注**

已决、无风险（有资料依据）；参考风险片段为空。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |
| lb_fine | 无 | 0 / 0 / 0 | 0 | 0 | 0 |
| binding_fine | 无 | 0 / 0 / 0 | 0 | 0 | 0 |

生成记录 SHA256：`8d8625ffd6f20e143f92267a42159c9a91302b3993356a9d8354edb388c6e168`。

### 64. ragognize_train_1773__partial

- 输入行：`313`（从 0 开始）；类别：`relation`；条件：`partial`。
- 问题：Which organization has Jonathan Hill on its Board of Directors?

**原始回答**

```text
1. Jonathan Hill is on the Board of Directors at the Pacific Northwest College of Art.
```

**参考标注**

已决、有风险；与当前资料的关系：<code>unsupported</code>。

- <code>[53,85)</code> <code>&quot;Pacific Northwest College of Art&quot;</code>

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0）<br><code>[11,16)</code> <code>&quot; Hill&quot;</code>（BPE 3）<br><code>[19,22)</code> <code>&quot; on&quot;</code>（BPE 5） | 2 / 1 / 0 | 0 | 2 | 5 |
| lb_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0）<br><code>[48,52)</code> <code>&quot; the&quot;</code>（BPE 11）<br><code>[60,70)</code> <code>&quot; Northwest&quot;</code>（BPE 13） | 2 / 1 / 0 | 1 | 1 | 4 |
| binding_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0）<br><code>[48,52)</code> <code>&quot; the&quot;</code>（BPE 11）<br><code>[60,78)</code> <code>&quot; Northwest College&quot;</code>（BPE 13,14）<br><code>[81,85)</code> <code>&quot; Art&quot;</code>（BPE 16） | 4 / 1 / 0 | 3 | 1 | 2 |

生成记录 SHA256：`94dbd5f3eb3cb27907c4f381d918be6b8e6a2accb14de7a7be7afe8aa0c240b4`。

### 65. ragognize_test_1644__complete

- 输入行：`324`（从 0 开始）；类别：`action`；条件：`complete`。
- 问题：What did the ethics commissioner recommend for Patrick Johnstone after reviewing his COP28 trip?

**原始回答**

```text
1. The ethics commissioner recommended that Patrick Johnstone receive additional ethics training after reviewing his COP28 trip.
```

**参考标注**

已决、无风险（有资料依据）；参考风险片段为空。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |
| lb_fine | 无 | 0 / 0 / 0 | 0 | 0 | 0 |
| binding_fine | <code>[120,121)</code> <code>&quot;2&quot;</code>（BPE 18） | 1 / 0 / 0 | 0 | 1 | 0 |

生成记录 SHA256：`9c653dae4dbb3d9c5de353decaa84fe045890ae3a3b1156f77bfe1197b07cf1e`。

### 66. ragognize_test_1644__partial

- 输入行：`325`（从 0 开始）；类别：`action`；条件：`partial`。
- 问题：What did the ethics commissioner recommend for Patrick Johnstone after reviewing his COP28 trip?

**原始回答**

```text
1. The ethics commissioner recommended that Patrick Johnstone provide a detailed accounting of his expenses and a public report on the benefits derived from his attendance at COP28.
```

**参考标注**

已决、有风险；与当前资料的关系：<code>unsupported</code>。

- <code>[62,180)</code> <code>&quot;provide a detailed accounting of his expenses and a public report on the benefits derived from his attendance at COP28&quot;</code>

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0）<br><code>[13,43)</code> <code>&quot; commissioner recommended that&quot;</code>（BPE 4–6）<br><code>[80,91)</code> <code>&quot; accounting&quot;</code>（BPE 13）<br><code>[98,156)</code> <code>&quot; expenses and a public report on the benefits derived from&quot;</code>（BPE 16–25）<br><code>[171,174)</code> <code>&quot; at&quot;</code>（BPE 28）<br><code>[178,179)</code> <code>&quot;2&quot;</code>（BPE 30） | 16 / 1 / 0 | 13 | 3 | 9 |
| lb_fine | <code>[61,143)</code> <code>&quot; provide a detailed accounting of his expenses and a public report on the benefits&quot;</code>（BPE 10–23）<br><code>[151,156)</code> <code>&quot; from&quot;</code>（BPE 25）<br><code>[178,179)</code> <code>&quot;2&quot;</code>（BPE 30） | 16 / 0 / 0 | 16 | 0 | 6 |
| binding_fine | <code>[61,143)</code> <code>&quot; provide a detailed accounting of his expenses and a public report on the benefits&quot;</code>（BPE 10–23）<br><code>[151,156)</code> <code>&quot; from&quot;</code>（BPE 25）<br><code>[178,179)</code> <code>&quot;2&quot;</code>（BPE 30） | 16 / 0 / 0 | 16 | 0 | 6 |

生成记录 SHA256：`37fca39f609e4b354d68417f7e010a03c106e3d10d99f85dcd6c926fc22d4d91`。

### 67. ragognize_test_1723__complete

- 输入行：`326`（从 0 开始）；类别：`action`；条件：`complete`。
- 问题：What was the goal of the Tesla Takedown movement that emerged in early 2025?

**原始回答**

```text
1. The goal of the Tesla Takedown movement that emerged in early 2025 was to economically impact Elon Musk and challenge his political influence.
```

**参考标注**

已决、无风险（有资料依据）；参考风险片段为空。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |
| lb_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |
| binding_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |

生成记录 SHA256：`1bcb7e249aff6efd9b4cda8133ab29c6ff6562081aa865858eb46e27d5c17717`。

### 68. ragognize_test_1723__partial

- 输入行：`327`（从 0 开始）；类别：`action`；条件：`partial`。
- 问题：What was the goal of the Tesla Takedown movement that emerged in early 2025?

**原始回答**

```text
1. The goal of the Tesla Takedown movement that emerged in early 2025 was to target Tesla, Inc. and its CEO, Elon Musk.
```

**参考标注**

已决、无风险（有资料依据）；参考风险片段为空。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0）<br><code>[69,76)</code> <code>&quot; was to&quot;</code>（BPE 19,20）<br><code>[83,89)</code> <code>&quot; Tesla&quot;</code>（BPE 22） | 3 / 1 / 0 | 0 | 3 | 0 |
| lb_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0）<br><code>[73,76)</code> <code>&quot; to&quot;</code>（BPE 20）<br><code>[83,89)</code> <code>&quot; Tesla&quot;</code>（BPE 22） | 2 / 1 / 0 | 0 | 2 | 0 |
| binding_fine | <code>[0,2)</code> <code>&quot;1.&quot;</code>（BPE 0,1）<br><code>[73,76)</code> <code>&quot; to&quot;</code>（BPE 20）<br><code>[83,89)</code> <code>&quot; Tesla&quot;</code>（BPE 22） | 2 / 1 / 1 | 0 | 2 | 0 |

生成记录 SHA256：`f08f612a02954636f80bd8bcccc77be2bcfcf1614d2de3fdea351a526f432aa4`。

### 69. ragognize_train_0951__complete

- 输入行：`360`（从 0 开始）；类别：`action`；条件：`complete`。
- 问题：Why were the primary elections for the 2025 Hattiesburg mayoral race canceled?

**原始回答**

```text
1. The primary elections for the 2025 Hattiesburg mayoral race were canceled because no one else challenged Toby Barker for the seat.
```

**参考标注**

已决、无风险（有资料依据）；参考风险片段为空。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0）<br><code>[39,42)</code> <code>&quot;att&quot;</code>（BPE 13） | 1 / 1 / 0 | 0 | 1 | 0 |
| lb_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |
| binding_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |

生成记录 SHA256：`aa14b07bc95c50e3976d73c7f4dd178a3bd0f2632f57269cbc3664e951e6e675`。

### 70. ragognize_train_0951__partial

- 输入行：`361`（从 0 开始）；类别：`action`；条件：`partial`。
- 问题：Why were the primary elections for the 2025 Hattiesburg mayoral race canceled?

**原始回答**

```text
There is no information provided about the cancellation of the primary elections for the 2025 Hattiesburg mayoral race in the given search results.
```

**参考标注**

拒答，排除主指标；没有可评参考风险片段。

冻结标注说明：Accurately says no cancellation details are supplied.

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,5)</code> <code>&quot;There&quot;</code>（BPE 0）<br><code>[8,11)</code> <code>&quot; no&quot;</code>（BPE 2）<br><code>[23,32)</code> <code>&quot; provided&quot;</code>（BPE 4）<br><code>[88,90)</code> <code>&quot; 2&quot;</code>（BPE 14,15）<br><code>[113,118)</code> <code>&quot; race&quot;</code>（BPE 25）<br><code>[131,146)</code> <code>&quot; search results&quot;</code>（BPE 29,30） | 0 / 7 / 1 | — | — | — |
| lb_fine | <code>[23,32)</code> <code>&quot; provided&quot;</code>（BPE 4） | 0 / 1 / 0 | — | — | — |
| binding_fine | <code>[23,32)</code> <code>&quot; provided&quot;</code>（BPE 4） | 0 / 1 / 0 | — | — | — |

生成记录 SHA256：`214093f43034d7d6d1220b0afe943372492d9ae175be742a7c4f04510f1104a4`。

### 71. ragognize_train_1035__complete

- 输入行：`364`（从 0 开始）；类别：`action`；条件：`complete`。
- 问题：What did Syed Ashhad Rashidi call for in response to alleged police misconduct in 2024?

**原始回答**

```text
1. Syed Ashhad Rashidi called for judicial inquiries into alleged police misconduct in response to alleged police misconduct in 2024.
```

**参考标注**

已决、无风险（有资料依据）；参考风险片段为空。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | 无 | 0 / 0 / 0 | 0 | 0 | 0 |
| lb_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0）<br><code>[11,14)</code> <code>&quot;had&quot;</code>（BPE 5） | 1 / 1 / 0 | 0 | 1 | 0 |
| binding_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0）<br><code>[11,14)</code> <code>&quot;had&quot;</code>（BPE 5） | 1 / 1 / 0 | 0 | 1 | 0 |

生成记录 SHA256：`75320fa571ead9c13923e942451063d4567089020a3a009ded1a9dd7aa4bf3d3`。

### 72. ragognize_train_1035__partial

- 输入行：`365`（从 0 开始）；类别：`action`；条件：`partial`。
- 问题：What did Syed Ashhad Rashidi call for in response to alleged police misconduct in 2024?

**原始回答**

```text
1. In 2024, Syed Ashhad Rashidi called for Jamiat Ulama-e-Hind representatives to assess the situation following communal violence in Bahraich and Sambhal.
```

**参考标注**

已决、无风险（有资料依据）；参考风险片段为空。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[52,55)</code> <code>&quot;ama&quot;</code>（BPE 20）<br><code>[59,62)</code> <code>&quot;ind&quot;</code>（BPE 23）<br><code>[148,151)</code> <code>&quot;amb&quot;</code>（BPE 38） | 3 / 0 / 0 | 0 | 3 | 0 |
| lb_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0）<br><code>[52,55)</code> <code>&quot;ama&quot;</code>（BPE 20） | 1 / 1 / 0 | 0 | 1 | 0 |
| binding_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0）<br><code>[46,55)</code> <code>&quot;iat Ulama&quot;</code>（BPE 18–20）<br><code>[112,121)</code> <code>&quot; communal&quot;</code>（BPE 30） | 4 / 1 / 0 | 0 | 4 | 0 |

生成记录 SHA256：`829ec9bfac0039920d49c232870199aa67f69a04df7ad67052ffa3e2d5be54bb`。

### 73. ragognize_train_1042__complete

- 输入行：`366`（从 0 开始）；类别：`action`；条件：`complete`。
- 问题：What reasons were given for creating Punggol GRC before the 2025 general election?

**原始回答**

```text
1. Punggol GRC was created before the 2025 general election to reflect the town's geographical location and accommodate population growth.
```

**参考标注**

已决、无风险（有资料依据）；参考风险片段为空。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |
| lb_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0）<br><code>[79,103)</code> <code>&quot;&#x27;s geographical location&quot;</code>（BPE 23–25） | 3 / 1 / 0 | 0 | 3 | 0 |
| binding_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0）<br><code>[74,103)</code> <code>&quot; town&#x27;s geographical location&quot;</code>（BPE 22–25） | 4 / 1 / 0 | 0 | 4 | 0 |

生成记录 SHA256：`61b0f821792a5478764cfe6b3230fdc9d4926988004a87ae02a89179870a0cf4`。

### 74. ragognize_train_1042__partial

- 输入行：`367`（从 0 开始）；类别：`action`；条件：`partial`。
- 问题：What reasons were given for creating Punggol GRC before the 2025 general election?

**原始回答**

```text
1. Before the 2025 general election, Punggol GRC was formed by merging the entirety of Punggol West SMC with the Punggol Coast and Punggol Shore wards of the Pasir Ris–Punggol GRC.
```

**参考标注**

已决、无风险（有资料依据）；参考风险片段为空。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0）<br><code>[153,161)</code> <code>&quot; the Pas&quot;</code>（BPE 47,48）<br><code>[169,172)</code> <code>&quot;ung&quot;</code>（BPE 53）<br><code>[175,177)</code> <code>&quot; G&quot;</code>（BPE 56） | 4 / 1 / 0 | 0 | 4 | 0 |
| lb_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |
| binding_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0）<br><code>[175,177)</code> <code>&quot; G&quot;</code>（BPE 56） | 1 / 1 / 0 | 0 | 1 | 0 |

生成记录 SHA256：`b783f2f93b16b024ea03e5d521705ff529114164cbcd5127203bc7b0f7d0d69f`。

### 75. ragognize_train_1543__complete

- 输入行：`376`（从 0 开始）；类别：`action`；条件：`complete`。
- 问题：What did the Bundestag decide on 30 January 2025 after debating the proposed prohibition of Alternative for Germany?

**原始回答**

```text
1. The Bundestag decided to forward the question of prohibiting Alternative for Germany to a parliamentary committee for further consideration on January 30, 2025.
```

**参考标注**

已决、无风险（有资料依据）；参考风险片段为空。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | 无 | 0 / 0 / 0 | 0 | 0 | 0 |
| lb_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |
| binding_fine | 无 | 0 / 0 / 0 | 0 | 0 | 0 |

生成记录 SHA256：`af88d0f9b02db9aa808e8e8906ba53ebc271658b15da6389d47ec7237fa3b4a7`。

### 76. ragognize_train_1543__partial

- 输入行：`377`（从 0 开始）；类别：`action`；条件：`partial`。
- 问题：What did the Bundestag decide on 30 January 2025 after debating the proposed prohibition of Alternative for Germany?

**原始回答**

```text
1. The Bundestag decided to advance a resolution calling for the Federal Constitutional Court to initiate an inquiry into the prohibition of Alternative for Germany on 30 January 2025.
```

**参考标注**

已决、有风险；与当前资料的关系：<code>unsupported</code>。

- <code>[17,48)</code> <code>&quot;decided to advance a resolution&quot;</code>

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[14,16)</code> <code>&quot;ag&quot;</code>（BPE 5） | 1 / 0 / 0 | 0 | 1 | 5 |
| lb_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 5 |
| binding_fine | 无 | 0 / 0 / 0 | 0 | 0 | 5 |

生成记录 SHA256：`71664d91bbf58850c26e2fb1610b25228127d910e03fac794ec056d538b101be`。

### 77. ragognize_train_2144__complete

- 输入行：`390`（从 0 开始）；类别：`action`；条件：`complete`。
- 问题：What clarification did Bradford Blackmon give about the seriousness of his Contraception Begins at Erection Act proposal after receiving threats?

**原始回答**

```text
1. Bradford Blackmon clarified that the Contraception Begins at Erection Act proposal was made in jest after receiving death threats.
```

**参考标注**

已决、无风险（有资料依据）；参考风险片段为空。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |
| lb_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |
| binding_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |

生成记录 SHA256：`5c1361a084d38c86ac3ef4a6976b8b759e4ea3e8d5758ecc3acf559079261a0c`。

### 78. ragognize_train_2144__partial

- 输入行：`391`（从 0 开始）；类别：`action`；条件：`partial`。
- 问题：What clarification did Bradford Blackmon give about the seriousness of his Contraception Begins at Erection Act proposal after receiving threats?

**原始回答**

```text
1. Bradford Blackmon clarified that the Contraception Begins at Erection Act was not intended to criminalize contraception but rather to address non-consensual fertilization.
```

**参考标注**

已决、有风险；与当前资料的关系：<code>unsupported</code>。

- <code>[81,173)</code> <code>&quot;not intended to criminalize contraception but rather to address non-consensual fertilization&quot;</code>

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0）<br><code>[11,17)</code> <code>&quot; Black&quot;</code>（BPE 3）<br><code>[30,39)</code> <code>&quot; that the&quot;</code>（BPE 6,7）<br><code>[72,80)</code> <code>&quot; Act was&quot;</code>（BPE 15,16）<br><code>[84,93)</code> <code>&quot; intended&quot;</code>（BPE 18）<br><code>[96,122)</code> <code>&quot; criminalize contraception&quot;</code>（BPE 20–22）<br><code>[126,173)</code> <code>&quot; rather to address non-consensual fertilization&quot;</code>（BPE 24–31） | 17 / 1 / 0 | 12 | 5 | 3 |
| lb_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0）<br><code>[20,35)</code> <code>&quot; clarified that&quot;</code>（BPE 5,6）<br><code>[72,173)</code> <code>&quot; Act was not intended to criminalize contraception but rather to address non-consensual fertilization&quot;</code>（BPE 15–31） | 19 / 1 / 0 | 15 | 4 | 0 |
| binding_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0）<br><code>[20,35)</code> <code>&quot; clarified that&quot;</code>（BPE 5,6）<br><code>[72,173)</code> <code>&quot; Act was not intended to criminalize contraception but rather to address non-consensual fertilization&quot;</code>（BPE 15–31） | 19 / 1 / 0 | 15 | 4 | 0 |

生成记录 SHA256：`85fb46e5ee76e38106af119e1649a663bd33e700fe1682bffbf509e0662fd109`。

### 79. ragognize_train_2314__complete

- 输入行：`398`（从 0 开始）；类别：`action`；条件：`complete`。
- 问题：What change to state income tax did House Bill 2601, introduced with Nico Delgado’s involvement, propose?

**原始回答**

```text
1. House Bill 2601, introduced with Nico Delgado's involvement, proposed to exempt minors making less than $50,000 from state income tax.
```

**参考标注**

已决、无风险（有资料依据）；参考风险片段为空。

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |
| lb_fine | <code>[0,1)</code> <code>&quot;1&quot;</code>（BPE 0） | 0 / 1 / 0 | 0 | 0 | 0 |
| binding_fine | 无 | 0 / 0 / 0 | 0 | 0 | 0 |

生成记录 SHA256：`d7850375eb0ae7014ad30c46ed5c7c136a45af894732d13081ed5452f4a88933`。

### 80. ragognize_train_2314__partial

- 输入行：`399`（从 0 开始）；类别：`action`；条件：`partial`。
- 问题：What change to state income tax did House Bill 2601, introduced with Nico Delgado’s involvement, propose?

**原始回答**

```text
House Bill 2601, introduced with Nico Delgado’s involvement, proposed a change to state income tax that aimed to retain younger professionals in Oklahoma.
```

**参考标注**

已决、有风险；与当前资料的关系：<code>unsupported</code>。

- <code>[104,153)</code> <code>&quot;aimed to retain younger professionals in Oklahoma&quot;</code>

**固定阈值报警**

| 方法 | 原文报警区间 | 报警词元：可评 / 其他含字 / 纯标点空白 | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| lb_coarse | <code>[0,11)</code> <code>&quot;House Bill &quot;</code>（BPE 0–2）<br><code>[12,13)</code> <code>&quot;6&quot;</code>（BPE 4）<br><code>[14,27)</code> <code>&quot;1, introduced&quot;</code>（BPE 6,8）<br><code>[47,69)</code> <code>&quot; involvement, proposed&quot;</code>（BPE 14–16）<br><code>[71,94)</code> <code>&quot; change to state income&quot;</code>（BPE 18–21）<br><code>[98,127)</code> <code>&quot; that aimed to retain younger&quot;</code>（BPE 23–27）<br><code>[141,144)</code> <code>&quot; in&quot;</code>（BPE 29） | 17 / 0 / 2 | 5 | 12 | 2 |
| lb_fine | <code>[71,78)</code> <code>&quot; change&quot;</code>（BPE 18）<br><code>[87,94)</code> <code>&quot; income&quot;</code>（BPE 21）<br><code>[109,127)</code> <code>&quot; to retain younger&quot;</code>（BPE 25–27） | 5 / 0 / 0 | 3 | 2 | 4 |
| binding_fine | <code>[71,78)</code> <code>&quot; change&quot;</code>（BPE 18）<br><code>[87,94)</code> <code>&quot; income&quot;</code>（BPE 21）<br><code>[109,127)</code> <code>&quot; to retain younger&quot;</code>（BPE 25–27） | 5 / 0 / 0 | 3 | 2 | 4 |

生成记录 SHA256：`0b77c62955676ca7e1303ddd99148a2567a9d0f0dce55d61efe11a24ee822d08`。

## 文件绑定与复算说明

本报告通过 `row_id` 连接输入、回答、标注和词元分数。每条标注的 `source_generation_sha256` 与实际生成文件及冻结清单逐一一致；词元编号、token ID、字符范围和原文片段也逐一核对。

关键字段：`inputs.questions[0] / condition / split`；`generation_records.response / response_token_offsets / response_token_ids`；`annotations_test.localization_status / original_risk / risk_spans`；`token_scores_test.start / end / lexical / main_eligible / gold / scores`；`freeze9.thresholds[method].threshold`。

| 文件 | SHA256 |
|---|---|
| [data/inputs.jsonl](<D:/Projects/Multi_Agent_Graph_Analysis/prelab/round9_evidence_binding/data/inputs.jsonl>) | `137059b3d05e8ee124d23ceed9a44c2182d6075e32a2328a269c7b9531b2f4d7` |
| [data/annotations_test.jsonl](<D:/Projects/Multi_Agent_Graph_Analysis/prelab/round9_evidence_binding/data/annotations_test.jsonl>) | `29807a6b524a3bef119c0b3ff3efcd39d3f3083d5394fa6f4873a1b1d4f78926` |
| [data/annotation_freeze.json](<D:/Projects/Multi_Agent_Graph_Analysis/prelab/round9_evidence_binding/data/annotation_freeze.json>) | `d5ddb893124fd2414c9e6ea483bde91e26a8b364aaff5c333fae60072cc5fa7c` |
| [data/freeze.json](<D:/Projects/Multi_Agent_Graph_Analysis/prelab/round9_evidence_binding/data/freeze.json>) | `4a46dc042e531076e6803d820f1576832c28a58a6641009b52c9f08436707def` |
| [results/token_scores_test.jsonl](<D:/Projects/Multi_Agent_Graph_Analysis/prelab/round9_evidence_binding/results/token_scores_test.jsonl>) | `e9894dea9e1fbfec853c14d14a5c2cd074d8892ffce5b82e47ad5756eb80447c` |
| [results/freeze9.json](<D:/Projects/Multi_Agent_Graph_Analysis/prelab/round9_evidence_binding/results/freeze9.json>) | `32384d9425c3145b7c72d640ee1fb730172e01adfa6e04c35a46ff3686a25236` |
| [results/metrics_test.json](<D:/Projects/Multi_Agent_Graph_Analysis/prelab/round9_evidence_binding/results/metrics_test.json>) | `46bbc1cb6cc0f88beedeaa16a95608f443b5c0daaeb7923772c0367074593001` |
| [results/test_complete9.json](<D:/Projects/Multi_Agent_Graph_Analysis/prelab/round9_evidence_binding/results/test_complete9.json>) | `357a51506e96988d950fcec68158b83e7c8c8445280b945c076592af7feea496` |

上面每条案例给出各自生成文件的 SHA256。80 文件清单摘要按展示顺序，以 `row_id + "\t" + SHA256 + "\n"` 拼接为 UTF-8 字节后计算：

`e2e51cf71e43e3dbd376ba24713cbd0c7ec543ca6e291d93a8651b2b0219e427`

只写入本 Markdown 报告；没有修改数据、参考标注、分数、阈值或源码。
