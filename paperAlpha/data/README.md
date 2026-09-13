# paperAlpha 数据目录

## 当前分区（2026-09-11）

- `raw/`：新论文的原始数据和运行记录，按来源或运行编号隔离，原始记录不覆盖。
- `processed/`：新论文的清洗、标注、特征等派生数据，记录对应原始数据及处理配置。
- `splits/`：数据集划分，记录来源与样本标识，不存原文或凭据。
- `pre_exp1/`：旧预实验数据，保持原路径，不混入新论文数据。
- `shared/`：旧实验的静态模板和 fixtures，复用前核对适用范围。

新目录目前为空，没有迁入 BayesTrace 数据集。图表、指标、报告和模型产物放到
`../results/`，方法代码放到 `../src/`。新数据不应包含密钥；API 运行记录写入前
去除认证信息，敏感内容按数据使用权限处理。

以下是旧实验的数据说明，状态未在本次整理中重新验收。
其中关于虚构数据和 API 返回的描述仅适用于原说明范围，不是对未来新数据的保证。

## 旧预实验说明

`shared/` 保存小型、可版本控制、完全虚构的静态模板、fixtures 与 schema。
`pre_exp1/` 保存 P0 与后续串行准入阶段的运行数据。P0 历史产物继续位于原有
五个子目录；P1 Phase A 产物隔离在 `pre_exp1/p1_benign/`。

生命周期：

- `raw/` 原始 append-only JSONL；
- `interim/` 可重建的重放和质量检查；
- `processed/` 验证后的 episode 结果；
- `manifests/` 每次运行的追踪信息；
- `reports/` 人类可读摘要。

本目录不包含真实秘密、凭据、个人信息或真实 API 返回。

P1 静态输入位于 `shared/p1_benign/`，包括固定八角色任务所需的 20 条虚构材料、
内部 mock 记录、期望公开报告和 schema。每条内部记录的假测试秘密只用于沙箱验证。

P1 生成数据位于：

```text
pre_exp1/p1_benign/
├── raw/
├── interim/
├── processed/
├── manifests/
└── reports/
```

`test_double` 与 `local_model_shakedown` 产物均明确
`eligible_for_scientific_analysis=false`。

P1 本地真实模型资格检查与 P1 Phase A 完全隔离：静态虚构任务在
`shared/p1_model_qualification/`，生成证据仅在
`pre_exp1/p1_model_qualification/` 的 `raw/`、`interim/`、`processed/`、
`manifests/` 与 `reports/` 中。它们同样明确
`eligible_for_scientific_analysis=false`，不得并入后续 20-episode 科学数据。

P1 manifest 的权威 artifact hash map 精确包含 events、outcomes、replay 和 run
report。validation JSON 与 validation report 是可重建派生产物；应以当前源码重新
运行 validator 的结果为准。hardening 后当前正式 dry-run 为
`P1-BENIGN-DRY-20260731T094841301886Z`（validation `99/99`、replay `2/2`）。

正式 P1 良性门槛批次使用独立科学层：工程 dry-run 仅写入
`pre_exp1/p1_scientific_benign/engineering_dry/`，真实 20-episode gate pilot 仅写入
`pre_exp1/p1_scientific_benign/pilot/`。pilot 标签为
`P1_AWAITING_MAIN_AGENT_ACCEPTANCE`、`P1_GATE_PILOT_ONLY`；它不属于确认性或因果
效应数据，且不得与 Phase A、qualification 或 P0 原始产物混合。当前真实 run 为
`P1-BENIGN-PILOT-QWEN3-20260731T132033188154Z`，当前四次返工审计的逐字段统计为 `74/80`
（原始报告的整条报告布尔统计为 `68/80`；该 68/80 仅为历史基线）。当前审计只写入
`pre_exp1/p1_scientific_benign/pilot/acceptance_repair/acceptance4-20260801T120000000000Z/`，并标记
`derived_from_run_id`、原始 hash、精确四 schema provenance、`no_provider_calls=true`、
`network_calls=0`、`p1_go=false`。当前交付说明见
`pre_exp1/p1_scientific_benign/REWORK4_DELIVERY.md`；`REWORK2_DELIVERY.md` 与 `REWORK3_DELIVERY.md` 仅为历史返工说明，不是当前交付；不得重跑既有 pilot 或进入 P2。
