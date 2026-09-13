# paperAlpha

## 项目宪章：任何后续 Agent 都必须先读这里

### 研究对象

本项目研究的是**现实中通过 API 运行的多智能体系统（MAS）的安全风险**。可用观测主要是运行时的请求、Agent 间消息、工具调用、权限/批准、记忆读写、状态变化和外部执行结果；不能假定能够取得 API 模型的权重或内部状态。

### 固定研究主线

项目不是普通文献综述，不是单纯扩大 BayesTrace 的节点数量，也不是无约束的因果发现。固定主线是：

> **风险知识库 → 知识驱动的分层图构建 → API Agent 运行时风险监测**

1. 系统整理危险行为、防御机制、成立条件、可观测证据和正常对照，形成有来源的 MAS 安全风险知识体系。
2. 按 MAS 的职责和工作流建立系统级 DAG；每个功能节点再由相关文献知识生成可解释的局部风险模板/子贝叶斯网络。
3. 通过状态、来源、授权、记忆、工具副作用和结果等类型化变量连接局部网络，形成分层、可组合的系统级风险图。
4. 利用 API 可见的运行记录学习或校准参数，输出风险评估、预警和证据链解释。

“图自己生长”只表示：**拓扑由工作流、文献证据和安全策略约束生成，参数由运行数据学习或校准**；不表示让模型自由生成不可验证的图。文献提供风险机制和结构依据，不等于已经证明因果关系。

### 预期贡献的固定表述

- **知识贡献**：面向 MAS 的结构化危险行为/防御知识体系，而非只列攻击名称。
- **方法贡献**：把上述知识转成可解释、可扩展、可组合的分层贝叶斯风险图构建机制。
- **验证贡献**：在 API Agent 运行轨迹上进行风险监测验证；条件允许时，发布带工作流、消息、工具、权限、记忆、外部效果和风险标签的数据集/基准。

数据集不是必然贡献：只有具备清晰标注规范、正常与危险对照、跨 Agent 关系及可复现实验协议时，才可作为独立成果。

### 术语和边界（不可误解）

- API 场景的主系统应称为**灰盒/外部可观测监测**，不能声称从文本恢复了模型内部白盒机制；本地模型探针只能作为可选增强或后续扩展。
- 风险入口、失败方式、危险动作和最终危害必须分开记录；“模型说要做”不等于“动作已执行”。
- 单 Agent 安全不等于 MAS 安全；重点关注委派、共享记忆、信息流、权限传递、组合危害、重复执行和全局资源约束。
- 综述和风险目录是图构建的知识底座，不应被误写成已经完成的实验、因果发现或绝对安全保证。
- 旧的 `pre_exp1/`、历史数据和共享实验平台属于可复用遗产，不能自动当作当前论文结果或当前方法已经实现。

后续任何新增代码、数据、实验或论文文字，都必须说明它位于上述主线的哪一层；如果改变研究对象、观测假设或核心贡献，先更新本节再继续。

## 当前用途（2026-09-13）

这里统一存放当前讨论的 **BayesTrace 向多智能体系统（MAS）安全风险评估拓展**
论文的代码、配置、数据和实验结果。`paperAlpha` 是工程代号，不表示大论文中的章节顺序。
具体研究切口和创新性仍在讨论，尚未冻结实验方案；目录整理不代表新方法已经实现。

实际路径：`D:\Projects\Multi_Agent_Graph_Analysis\paperAlpha`。

## 新工作放在哪里

| 目录 | 用途 |
|---|---|
| `src/` | 新论文的方法代码、特征提取与监测模块 |
| `configs/` | 可复现的模型、数据和实验配置；不保存密钥 |
| `scripts/` | 数据处理、训练、评估等运行入口 |
| `tests/` | 新代码的测试 |
| `research/` | 危险行为目录、来源与机制归纳；不等同于已冻结的实验方案 |
| `data/raw/` | 新工作的原始数据和运行记录；保留来源，不覆盖原始记录 |
| `data/processed/` | 清洗、标注和特征等派生数据 |
| `data/splits/` | 训练、验证、测试划分及其来源记录 |
| `results/` | 按实验名称和运行编号分开保存指标、图表、报告与模型产物 |

代码、配置与数据新目录目前只有占位文件；`research/` 已有文献整理。代码与生成数据分开；不同运行不互相覆盖。
原始数据、派生数据和运行结果默认不纳入 Git，划分文件只保存样本标识和来源，
不包含原文或凭据。需要共享小型数据时再明确选取。

现有 `pyproject.toml` 的安装和默认测试入口仍指向旧的 `pre_exp1`，
没有在本次整理中改成新方法入口；后续实现时再接入。
继续使用项目统一 Conda 环境 `multi_agent_graph`，不另建环境。

## 现有资产如何处理

- [research/README.md](research/README.md)：新增危险行为与风险机制目录（48 项、39 项来源），用于讨论后续研究范围。
- [pre_exp1/](pre_exp1/)：旧 P0、P1 和 P1v2 预实验代码、配置与测试，原位保留。
- [data/pre_exp1/](data/pre_exp1/)：上述实验的历史运行数据，原位保留，不当作新论文结果。
- [data/shared/](data/shared/)：旧实验的静态任务、模板和 schema，可按需要复用。
- [../shared_lab/](../shared_lab/)：共享实验平台，保持独立；是否用于新论文尚未决定，不复制整套平台。
- [../doc/ref_paper/my_work/](../doc/ref_paper/my_work/)：已有论文原文，继续保留在文献区。
- [../doc/ref_paper/mas_safety_2026-09-11/](../doc/ref_paper/mas_safety_2026-09-11/)：MAS 安全文献与调研，继续保留在文献区。

旧资产可供后续评估复用，但本次未重新验收，也未迁移 BayesTrace 原始实现或数据集。
其他论文目录不并入本工程。

---

## 以下为旧路线与操作说明（历史记录）

以下保留原说明，方便定位旧实验及其使用限制；其中“当前”、阶段状态和论文目标
均指当时的工作，不代表新论文已经选定该路线，也不构成重跑实验的授权。

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
