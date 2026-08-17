# P1v2-Q0 离线模型资格框架

本目录只实现 `P1v2-Q0`：在不调用模型之前，验证资格检查器、公开发布权限和证据链是否能正确工作。

它不是 P1、P2 或模型能力结论。它不包含可执行的本地模型、HTTP 或远程 provider；唯一可运行的 provider 是确定性 fake provider。

## 当前 Q0 工件

当前待独立验收的 run 为：

```text
P1V2Q-READINESS-DRY-20260801T010106000000Z
```

工件和交付报告在：

```text
paperAlpha/data/pre_exp1/p1v2_qualification/readiness_dry/
```

`P1V2Q-READINESS-DRY-20260801T010101000000Z` 至 `P1V2Q-READINESS-DRY-20260801T010105000000Z` 均保留为返工前的历史离线工件；它们不是 current run，也不应用于任何后续判断。报告注册表要求整个 `reports/` 目录最多只能有一个 `Current Q0 run ID` 标记；其他报告必须标为 historical pre-rework。

## 固定环境与公开入口

仅使用：

```text
Conda: multi_agent_graph
Python: 3.11.15
Interpreter: D:\anaconda\envs\multi_agent_graph\python.exe
```

从项目根目录执行：

```powershell
conda run --no-capture-output -n multi_agent_graph python -B -m pytest -p no:cacheprovider paperAlpha/pre_exp1/p1v2_qualification/tests -q
conda run --no-capture-output -n multi_agent_graph python -B paperAlpha/pre_exp1/p1v2_qualification/scripts/run_readiness_dry.py
conda run --no-capture-output -n multi_agent_graph python -B paperAlpha/pre_exp1/p1v2_qualification/scripts/validate_readiness_dry.py
conda run --no-capture-output -n multi_agent_graph python -B paperAlpha/pre_exp1/p1v2_qualification/scripts/replay_readiness_dry.py
```

`run_readiness_dry.py` 是 append-only：同一 run ID 已存在时会 fail-closed，而不会覆盖或补跑。

## 边界

- Q0 只使用良性的虚构供应商任务和确定性 fake provider；
- Q0 的真实模型、loopback HTTP、远程网络、replay 模型和 replay 网络调用计数均必须为零；
- public sink 只接收经验证的 `publisher` 输出，且只保留四个公开字段；
- Q0 固化 128 张 allow 任务与 17 个负向 fixture；合法 reject 会先核对角色和 episode，再分别归类为 `semantic_reject`、`role_mismatch` 或 `episode_mismatch`；
- current 交付报告必须给出四类正常任务和每个负例的重算终态计数，validator 会复核这些人类可读字段；
- Q1 的未来配置被冻结在 `configs/q1_protocol.json`，但 Q1 尚未获准；
- `p1_go=false`、`p2_allowed=false`，P1/P2 保持 locked。

实现细节和验收边界见 [IMPLEMENTATION.md](./IMPLEMENTATION.md)。
