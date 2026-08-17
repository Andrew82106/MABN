# paperAlpha

`paperAlpha` 是第一篇论文的独立工程目录。论文一的总体目标是区分“发生了通信”
与“消息对下游可验证危害产生了因果影响”，并比较 Original、Safe replacement
与 Drop，以区分真正的安全遏制和任务停摆。

P0 工具验证已经完成。**P1 良性任务可完成性 Phase A** 的工程实施和独立只读
旁路审计也已完成：在不加入风险种子、不进入后续因果处理的前提下，已搭建八角色
良性工作流、provider 接口、程序化终点、日志、validation、全轨迹 replay 和
provenance。随后已运行唯一一次真实 P1 benign gate pilot；原始证据保持不变，目前处于
验收返工中，不得重跑、不得进入 P2。

Phase A 的 test double dry-run 与单 episode 本地模型 shakedown 只验证工程链路，
不产生实验性研究结论；真实 pilot 只用于 P1 gate 验收：

```text
P1 scientific gate: P1_AWAITING_MAIN_AGENT_ACCEPTANCE（验收返工中；不得 P2）
```

## 目录职责

- `pre_exp1/`：第一个预实验的代码、冻结配置、测试、脚本与实现说明；
- `pre_exp1/p1_benign/`：P1 Phase A 独立代码、配置、提示、命令和测试；
- `data/shared/`：可复用的小型虚构静态数据、模板和 schema；
- `data/pre_exp1/`：`pre_exp1` 的全部运行输出。

生成数据不得写入 `pre_exp1/`。原始追加式日志位于
`data/pre_exp1/raw/`，manifest、派生数据和报告分别位于相应子目录。

## 安装

统一使用 Conda 环境 `multi_agent_graph`、Python 3.11.15。运行时没有第三方
依赖；测试需要 pytest。`environment.yml` 只声明项目所需的最小环境。

```powershell
cd D:\Projects\Multi_Agent_Graph_Analysis\paperAlpha
conda run -n multi_agent_graph python -B -m pip install -e ".[test]"
```

## 测试

```powershell
cd D:\Projects\Multi_Agent_Graph_Analysis\paperAlpha
conda run -n multi_agent_graph python -B -m pytest -p no:cacheprovider
```

## P0 smoke test

```powershell
cd D:\Projects\Multi_Agent_Graph_Analysis\paperAlpha
conda run -n multi_agent_graph python -B pre_exp1\scripts\run_p0_smoke.py
```

smoke test 覆盖正常安全、危险 Original、Safe 和 Drop 四条 scripted 轨迹。
它只检查预先规定的代码行为。

## 验证和重放

将 `<RUN_ID>` 替换为 smoke test 打印的运行编号：

```powershell
conda run -n multi_agent_graph python -B pre_exp1\scripts\validate_run.py --run-id <RUN_ID>
conda run -n multi_agent_graph python -B -m pre_exp1.cli replay-run --run-id <RUN_ID>
```

所有上述命令产生的持久运行数据都写到
`D:\Projects\Multi_Agent_Graph_Analysis\paperAlpha\data\pre_exp1`。

## P1 Phase A 入口

P1 的安全边界、六个命令、数据位置和分析资格见
[pre_exp1/p1_benign/README.md](pre_exp1/p1_benign/README.md)。

确定性工程 dry-run：

```powershell
conda run --no-capture-output -n multi_agent_graph python -B pre_exp1\p1_benign\scripts\run_p1_dry.py --episodes 2
```

P1 测试：

```powershell
conda run --no-capture-output -n multi_agent_graph python -B -m pytest pre_exp1\p1_benign\tests -p no:cacheprovider
```

当前 `run_p1_live.py` 必须安全失败；真实模型、解码和预算尚未冻结。讯飞 Astron
Coding Plan 只登记为 `interactive_coding_only`，不得用于自动化 P1 调用。

hardening 后当前正式工程 dry-run 为
`P1-BENIGN-DRY-20260731T094841301886Z`，validation `99/99`、replay `2/2`、
P1 tests `224/224`、P0 regression `57/57`；畸形 artifact 会结构化失败而不是
抛 traceback。
它仍然 `eligible_for_scientific_analysis=false`。

## P1 正式良性门槛批次入口

P1 真实模型资格检查通过后，正式良性门槛批次位于
`pre_exp1/p1_scientific_benign/`。它只使用冻结的本机 `qwen3:8b`，按
`P1-TASK-001` 至 `P1-TASK-020` 顺序运行一次 20-episode benign pilot；不运行风险
种子、Original/Safe/Drop、因果分析或 P2。确定性工程证据与真实 pilot 分别位于
`data/pre_exp1/p1_scientific_benign/engineering_dry/` 和
`data/pre_exp1/p1_scientific_benign/pilot/`。

```powershell
conda run --no-capture-output -n multi_agent_graph python -B pre_exp1\p1_scientific_benign\scripts\run_scientific_dry.py
conda run --no-capture-output -n multi_agent_graph python -B pre_exp1\p1_scientific_benign\scripts\run_scientific_pilot.py
```

正式数据仅用于 P1 gate pilot，完成后状态为
`P1_AWAITING_MAIN_AGENT_ACCEPTANCE`，不由实施 Agent 宣布 P1 Go。

既有真实 pilot：`P1-BENIGN-PILOT-QWEN3-20260731T132033188154Z`。公开
validation/replay 默认只读并按 run ID 自动路由；`--repair-id <REPAIR_ID>` 只在
`pilot/acceptance_repair/<REPAIR_ID>/` 生成派生审计，不覆盖原始 pilot artifact。
当前唯一交付说明是 [pre_exp1/p1_scientific_benign/REWORK4_DELIVERY.md](pre_exp1/p1_scientific_benign/REWORK4_DELIVERY.md)；
`DELIVERY.md`、`REWORK_DELIVERY.md`、`REWORK2_DELIVERY.md` 和 `REWORK3_DELIVERY.md` 仅为历史基线。
当前状态为 `p1_go=false`、返工待验收、既有 pilot 未重跑、不得进入 P2。当前四次返工派生审计 ID 为
`acceptance4-20260801T120000000000Z`。

## P1 真实模型资格检查入口

P1 Phase A 通过后，真实本地模型资格检查位于
`pre_exp1/p1_model_qualification/`。它只使用已安装的 `qwen3:8b` 与
`ministral-3:8b` 各一次三-episode 串行资格批次，数据与 Phase A 隔离，且始终
`eligible_for_scientific_analysis=false`。命令与安全边界见
[pre_exp1/p1_model_qualification/README.md](pre_exp1/p1_model_qualification/README.md)。
