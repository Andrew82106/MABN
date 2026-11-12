# CERP-v2.1 结构门禁停止报告

状态：`stopped_structure_capacity_failed`。

v2.1 独立冻结后只读取了 v1.1 的无标签候选池，并让全部六类父候选接受同一套严格
局部对齐检查。没有读取 QA 对错标签、错误区间或窗口标签，没有加载模型或使用 GPU，
也没有改动正式 baseline、v1.1 或 v2.0。

## 冻结门禁结果

| 条件 | 实际值 | 阈值 | 结果 |
|---|---:|---:|---|
| 候选数 | 2,577 | >=10,000 | 失败 |
| 覆盖主张数 | 2,393 | >=5,000 | 失败 |
| 覆盖 group 数 | 548 | >=300 | 通过 |
| 至少 100 条的候选类型数 | 2 | >=3 | 失败 |
| 候选硬上限 | 2,577 | <=139,676 | 通过 |

最终类型为 `aligned_relation=2,382`、`citation=173`、`number=18`、
`negation_direction=2`、`entity=1`、`temporal=1`。115,483 条父候选中有 104,384 条
无法让其原区间和替换值唯一、精确地对应 SequenceMatcher 的局部替换块，说明父池中的
显式数字、日期、实体和方向候选绝大多数确实来自跨位置组合，而不是可回收的局部修复。

## 停止决定

容量门禁先失败，因此冻结的 `QC_SAMPLE_UNREVIEWED.jsonl` 未做研究代理判断，不能报告
clear rate。没有生成 NLI manifest 或 resume 入口，没有进入训练、融合或标签审计。

若继续迭代，不能再依赖 v1.1 的跨组合父池。下一版应直接从每个 label-blind
`claim/source_sentence` 对的 SequenceMatcher opcode 生成候选，并补充受约束的
insert/delete（尤其否定词）操作；否则数字、日期、实体和方向类型没有足够容量。
