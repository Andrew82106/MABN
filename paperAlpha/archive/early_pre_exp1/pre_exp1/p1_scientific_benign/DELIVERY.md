# P1 正式良性批次：历史首次交付/返工前基线

本文件只保留首次实施和返工前的历史证据，**不是当前交付说明**，不能替代主 Agent 验收。
第一次返工、二次返工和三次返工说明（`REWORK_DELIVERY.md`、`REWORK2_DELIVERY.md`、
`REWORK3_DELIVERY.md`）均为历史证据；当前唯一交付见 [REWORK4_DELIVERY.md](REWORK4_DELIVERY.md)。
历史数字不得解释为 current/latest/authoritative。

当前边界始终为：`p1_go=false`、不得 P2、既有真实 pilot 未重跑。

## 固定状态

```text
phase: P1_SCIENTIFIC_BENIGN
condition: benign_baseline
scientific_gate: P1_AWAITING_MAIN_AGENT_ACCEPTANCE
risk_seed_present: false
intervention_applied: false
```

## 实施文件

- `src/p1_scientific_benign/`：科学层核心、provider、runner、validation、replay、报告；
- `configs/scientific_benign.json`：20 task 顺序、qwen、预算和阈值；
- `scripts/`：dry、pilot、validation、replay；
- `tests/`：离线正向、负向、篡改和边界测试；
- `data/pre_exp1/p1_scientific_benign/engineering_dry/`：仅工程证据；
- `data/pre_exp1/p1_scientific_benign/pilot/`：仅真实 20-episode gate pilot。

## 环境与离线门

- 固定解释器：`D:\anaconda\envs\multi_agent_graph\python.exe`；Python `3.11.15`；
  pytest `8.4.2`；installed distribution `paperalpha-pre-exp1==0.2.0`。
- 首次实施时新层测试：`13 passed`；P1 Phase A：`224 passed`；qualification：`32 passed`；
  P0：`57 passed`。
- 只读历史 validation/replay：Phase A `PASS/2 matched`、qwen qualification
  `PASS`、ministral qualification `PASS`、P0 `PASS`；P0 raw 八项 SHA-256 全部匹配，
  未发现 cache/venv/pyc/egg-info 或后续阶段 artifact。
- 离线 dry run：`P1-BENIGN-PILOT-DRY-20260731T131719372119Z`，20/20 replay，
  180 transcript actions，engineering-only 标签；失败 dry 证据仍保留。

## 真实 P1 gate pilot

唯一真实 run：`P1-BENIGN-PILOT-QWEN3-20260731T132033188154Z`。

- 冻结 `qwen3:8b`、Ollama `0.24.0`、metadata hash
  `4d0f241b66d0e67f72e6ffbae82796ad7441c5df87bd27a2dee28c1ff480108f`，loopback
  endpoint；20 episodes，180 provider calls，串行、零重试，结构化输出 `179/180`。
- 报告发布 `19/20`；任务成功 `17/20`；原始报告按整条布尔值显示 `68/80`，返工后按
  权威 sink event 逐字段重算为 `74/80`；canary 泄漏
  `0`；内部/额外字段外发 `0`。
- 错误计数：`ProviderParseError=1`，scheduler/tool/unauthorized/timeout/retry 均为 `0`。
- transcript replay `20/20 matched`，live/network calls `0/0`；validation `FAIL`，唯一
  错误为 E004 的 provider/structured output 计数不足（9/8）。总时长约 `434.456 s`。

## P1 候选结论

历史首次交付曾记录“候选 No-Go（等待主 Agent 验收）”及当时的 68/80 口径；该描述仅是
返工前基线，不是当前最终研究判定。原始 pilot 的解析错误、179/180 structured outputs、
validation 失败和不得进入 P2 的边界仍须保留。

## 限制

不运行风险种子、Original/Safe/Drop、因果效应、贝叶斯主分析、P2 或任何后续阶段；
不读取 `.env`，不使用远程 API，不修改 `doc` 或上游运行数据。
