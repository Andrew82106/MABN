# 实验索引

[protocol.json](protocol.json) 是唯一主协议；[baselines.json](baselines.json) 是当前基线清单。
主开发结果见 [synthetic_mas/runs/independent_mas_journal_v1/REPORT.md](synthetic_mas/runs/independent_mas_journal_v1/REPORT.md)；
结果解释与限制见[论文汇报](../docs/manuscript/README.md)。

以下是现存资产的主题导航，不代表每个运行包均为有效、独立或成功的实验。本次保留全部 390 个运行包，未重算指标。

| 主题 | 运行包数 |
| --- | ---: |
| [A2A 安全基准](a2asecbench/) | 8 |
| [ATBench 外部迁移](atbench_transfer/) | 2 |
| [确认集准备](confirmation_preparation/) | 43 |
| [委派与授权](delegation/) | 50 |
| [失败定位](failure_attribution/) | 3 |
| [HarnessAudit 与原生守卫](harnessaudit/) | 34 |
| [大模型裁判对照](llm_judge/) | 15 |
| [MAST 外部迁移](mast_transfer/) | 1 |
| [消息与工作流](message_workflows/) | 22 |
| [模型机制检查](model_mechanisms/) | 71 |
| [隐私与信息传播](privacy_flow/) | 39 |
| [采购任务](purchase/) | 9 |
| [运行时预警](runtime_alarm/) | 5 |
| [运行成本与延迟](runtime_costs/) | 9 |
| [语义提取](semantic_extraction/) | 10 |
| [投稿证据审计](submission_audit/) | 5 |
| [合成 MAS 主实验](synthetic_mas/) | 64 |

## 每类实验的统一布局

- `configs/`：该类实验的参数设置。有实际配置时才建立，不创建空壳。
- `notes/`：该类实验的分析与边界说明。
- `runs/<运行ID>/`：该次运行的完整证据包，包括当时的输入、输出、模型、日志和源码快照。

只维护 [index.json](index.json) 这一份机器索引；它记录每个运行包的新位置与旧位置。新增运行先分配新 ID，完成后登记；不能覆盖旧目录。
结果是否能进入论文由协议与人工核验决定，不由目录名、日期或高分决定。

普通历史运行留在这里；只有整套退役系统进入 `archive/`。现有证据包内部不拆散，不重复复制到 `data/`。
