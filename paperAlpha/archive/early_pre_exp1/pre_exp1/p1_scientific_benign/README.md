# P1 正式良性批次

本目录实现论文一 P1 的唯一正式 benign gate pilot：冻结 `qwen3:8b`，按
`P1-TASK-001` 至 `P1-TASK-020` 顺序运行 20 个无风险虚构供应商审查 episode。
它证明正常八角色工作流是否本来可完成，不是风险传播、Original/Safe/Drop、因果
效应或 P2 实验。

## 固定边界

- provider：本机 Ollama `qwen3:8b`，仅 loopback `http://127.0.0.1:11434`；
- temperature `0`、seed `20260731`、最大输出 `256`、thinking `false`；
- 每 episode 最多 9 次角色调用，20 episodes / 180 calls，串行、零重试、90 秒 timeout；
- 真实 run 至多一次，失败或中断的 manifest 与 append-only 事件必须保留；
- 未来真实 runner 必须先通过当前代码/冻结输入对应的 deterministic offline gate，并执行 60 分钟总时限；
- 不读取 `paperAlpha/.env`，不使用远程 API，不下载或变异模型；
- 只读复用 `data/shared/p1_benign/` 与 Phase A/qualification/P0 代码和配置。

## 命令

```powershell
conda run --no-capture-output -n multi_agent_graph python -B pre_exp1\p1_scientific_benign\scripts\run_scientific_dry.py
conda run --no-capture-output -n multi_agent_graph python -B pre_exp1\p1_scientific_benign\scripts\run_scientific_pilot.py
conda run --no-capture-output -n multi_agent_graph python -B pre_exp1\p1_scientific_benign\scripts\validate_scientific_run.py --run-id <RUN_ID>
conda run --no-capture-output -n multi_agent_graph python -B pre_exp1\p1_scientific_benign\scripts\replay_scientific_run.py --run-id <RUN_ID>
# 只写版本化派生修复审计，不覆盖原 pilot：
conda run --no-capture-output -n multi_agent_graph python -B pre_exp1\p1_scientific_benign\scripts\validate_scientific_run.py --run-id <RUN_ID> --repair-id <REPAIR_ID>
```

dry 证据写入 `data/pre_exp1/p1_scientific_benign/engineering_dry/`，真实 pilot 证据
写入 `data/pre_exp1/p1_scientific_benign/pilot/`。真实 pilot 的最终状态必须是
`P1_AWAITING_MAIN_AGENT_ACCEPTANCE`，实施 Agent 不宣布 P1 Go。当前真实 run 已完成；
公开 validation/replay 为只读，按 `P1-BENIGN-PILOT-QWEN3-*` 与
`P1-BENIGN-PILOT-DRY-*` 自动路由。`DELIVERY.md` 是首次实施历史基线，
`REWORK_DELIVERY.md` 是第一次返工历史交付；
`REWORK2_DELIVERY.md`、`REWORK3_DELIVERY.md` 是历史返工说明；当前唯一交付说明是
`REWORK4_DELIVERY.md`（当前审计 `acceptance4-20260801T120000000000Z`），其中保留
`p1_go=false`、既有 pilot 未重跑和不得 P2 的边界。
