# CERP-v3.0 结构门禁停止报告

状态：`stopped_structure_capacity_failed_route_terminated`。

v3 使用 237,739 个物理无标签 `claim/source` 对，直接从 SequenceMatcher opcode 生成
局部替换、否定插入和否定删除。没有读取旧修复候选、QA 金标、错误区间、窗口标签、
校准集或测试集；没有加载模型、使用 GPU 或修改正式 baseline。

## 冻结门禁结果

| 条件 | 实际值 | 阈值 | 结果 |
|---|---:|---:|---|
| 候选数 | 3,220 | >=10,000 | 失败 |
| 覆盖主张数 | 2,864 | >=5,000 | 失败 |
| 覆盖 group 数 | 543 | >=300 | 通过 |
| 至少 100 条的类型数 | 1 | >=3 | 失败 |
| 候选硬上限 | 3,220 | <=139,676 | 通过 |

类型分布为：`relation=3,064`、`entity=84`、`number=60`、`negation=8`、
`temporal=3`、`direction=1`。操作分布为 `replace=3,212`、`insert=8`、`delete=0`。

容量先失败，因此冻结的 60 条抽样保持未评审状态；没有报告代理 clear rate，没有准备
NLI manifest/resume，也没有进入训练、融合或标签审计。

按预注册停止规则，本结果终止纯规则反事实候选生成路线，不再修改阈值或追加规则版本。
