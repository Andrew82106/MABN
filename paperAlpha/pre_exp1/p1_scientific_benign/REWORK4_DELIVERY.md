# P1 第四次返工实施报告（供主 Agent 验收）

本文件是第四次最小范围返工交付说明，也是当前唯一 current/latest 交付；它不是新的
P1 实验，不宣布 P1 Go、No-Go 或进入 P2。源 pilot、acceptance1、acceptance2、
acceptance3 均保留原状；完成本报告后停止等待主 Agent 验收。

## 1. 固定环境与边界

- Conda：`multi_agent_graph`；Python：`3.11.15`；解释器：
  `D:\anaconda\envs\multi_agent_graph\python.exe`。
- 所有验证/测试命令均使用 `conda run --no-capture-output -n multi_agent_graph python`。
- 未调用 Ollama、任何 provider、远程 API 或网络；未读取、散列、复制或输出
  `paperAlpha/.env`；未重跑真实 pilot、未启动第二次 P1、未启动 P2。
- `p1_go=false`、`p2_allowed=false`、`pilot_rerun_allowed=false`；本轮只生成一个
  版本化派生审计目录，不改写源 pilot 或旧审计。

## 2. 最终冻结代码与 acceptance4

最终代码冻结后，仅通过公开入口执行一次：

```powershell
conda run --no-capture-output -n multi_agent_graph python -B pre_exp1\p1_scientific_benign\scripts\validate_scientific_run.py --run-id P1-BENIGN-PILOT-QWEN3-20260731T132033188154Z --repair-id acceptance4-20260801T120000000000Z
```

CLI 退出码为 `1`，原因是 source pilot 的历史 validation 仍按预期 FAIL；replay 为
20/20 且无 provider/network 调用。CLI 只新建以下一个目录，未覆盖 source artifact：

`paperAlpha/data/pre_exp1/p1_scientific_benign/pilot/acceptance_repair/acceptance4-20260801T120000000000Z/`

该目录的机器可检查身份字段为：

```text
artifact_kind=derived_repair_audit
is_derived_audit=true
is_new_p1_run=false
source_run_id=P1-BENIGN-PILOT-QWEN3-20260731T132033188154Z
source_execution_mode=local_model_pilot
execution_mode=derived_repair_audit
audit_mode=no_provider_reinterpretation
no_provider_calls=true
network_calls=0
```

`manifest.json` SHA-256 为
`0d33423be407bc09f72db1d561c29201f95c66470265d2093d5dcbf0889c4636`。
七个派生 artifact hash（均与 manifest 的 `artifact_hashes` 逐项相等）为：

| artifact | SHA-256 |
|---|---|
| `validation.json` | `02b582c287a98f5e6ec42d6a094cdd005f1156978257e92248de414339469417` |
| `replay.json` | `b18334bc32af558725cb7101b958744ad7e4f07fa494b8590203d5d9fa590ee8` |
| `run_report.md` | `2e9c72496b5c2f36c71013dbcefb78e1de04726b645b086cf0e9818b4829bcc5` |
| `p1_gate_report.md` | `c1990be7c0aa9b45ef2bcbd222daf49f91c9ab6eeccfb500db906b3f8b9142f6` |
| `source_hashes.json` | `0ceb3159bbc3b2bf75a101847cf671ec09d45627d658eb3196d6526ff49715f0` |
| `derived_audit_identity.json` | `aab4d14d0bf887da738cdc741a398c2099b639085d6f1ed54ddba3091156f332` |
| `DERIVED_AUDIT.md` | `4c1edced76d7906ac3b847d5868f54205ebab29562d91f6a374f7337016edd48` |

## 3. repair_tool_provenance 等值证据

在 acceptance4 生成后，以同一固定解释器重新调用生产
`capture_provenance()`，结果为严格 Python 对象相等：
`manifest["repair_tool_provenance"] == capture_provenance(DEFAULT_CONTEXT)` → **True**。

- 当前 `scientific_execution_tree_hash`：
  `b19976f240ae927729b977788cc0666f6fc575ff2ba3fdebecf3b125098cfd76`。
- 当前 `audit.py`（同时是 manifest 中
  `pre_exp1/p1_scientific_benign/src/p1_scientific_benign/audit.py`）SHA-256：
  `a066eafb990b9cc19800a6fd329ef0ac1ef60a49420bc2f25b72560a3b70510b`。
- installed distribution 证据为 `paperalpha-pre-exp1==0.2.0`；记录的 Python 为
  `3.11.15`，`sys_executable` 为固定解释器。

四个共享 schema 的精确 provenance 集合如下（不是目录模糊匹配）：

| 相对路径 | SHA-256 |
|---|---|
| `data/shared/p1_benign/schemas/event.schema.json` | `282fd3c0f190970fde61af6f42959772f9bcc26ade9d47199208c064bb7ad83d` |
| `data/shared/p1_benign/schemas/manifest.schema.json` | `4698f563639952a3492dc14fd739c748a66f3f10f455b2a3b5bec85748a4b786` |
| `data/shared/p1_benign/schemas/outcome.schema.json` | `e8853d16d1295eec82fe55d8db20bafa52665ceb9ea14a1ff8351ed2063fc8b3` |
| `data/shared/p1_benign/schemas/replay.schema.json` | `21356cf86c875978fe4cb46933c525d000f72d5b11005379b7d04322d467418e` |

四文件 tree hash：`ee9c47d5012061fb738255b7e6efc5f7c6a7a7118648cd69403231b6580ca927`。

## 4. 候选门槛（仅工程证据，不是研究结论）

派生审计从原始 sink/event 重算得到：20/20 完整 episode、19/20 恰好发布一次、
17/20 任务成功、四个公开字段 `74/80`、Canary `0/20`、内部字段/额外字段
`0/20`、跨 episode 状态污染 `0`、replay `20/20`、replay live/provider/network
`0/0/0`。历史 source validation 仍为 `False`（labels、旧 episode 计数、provenance、
schema provenance 四项历史错误），因此 candidate conclusion 保持
`候选返工（实现交付未验收）`；不得将这些数字解释为 P1 Go 或科学结论。

## 5. 测试与只读复核

固定环境回归测试（CLI 退出码均为 `0`）：

```text
pre_exp1/p1_scientific_benign/tests   28 passed
pre_exp1/p1_benign/tests              224 passed
pre_exp1/p1_model_qualification/tests 32 passed
pre_exp1/tests                          57 passed
```

定向身份测试包含完整 provenance 全量等值断言；schema literal truth-set 漂移、
in-flight deadline、preflight 失败、生产 replay fresh semantic tamper 等测试均通过。
只读 API 复核结果：当前 engineering dry validation/replay PASS；原始 pilot validation
按历史事实 FAIL、replay 20/20 PASS；Phase A `P1-BENIGN-DRY-20260731T094841301886Z`、
qwen qualification、ministral qualification 和 P0 smoke 的 validation/replay 均 PASS，
各自未产生 provider/network 调用。

## 6. 保护性检查与文档状态

source pilot 三个受保护 artifact 生成前后完全相等：

```text
events    8de3e520fe86b3b1be74104d2aa1762ef8ce7b8775be5033f9cc3c51f050d38e  1072786 bytes  2026-07-31T13:27:48.0168635Z
manifest  66c48af2d881f298bfa438cf7c3e8f07dde29b6a8a329382cc4d9525875f7ab1     11934 bytes  2026-07-31T13:27:48.5160994Z
outcomes  b8887f9a2cb175e2ad885b77f315f30e710ada3321e251a31647c633e7d6e2ee     28964 bytes  2026-07-31T13:27:48.0222967Z
```

旧 acceptance1/2/3 manifest hash 也未改变；新目录列表中仅新增
`acceptance4-20260801T120000000000Z`。`paperAlpha/README.md`、`data/README.md`、
科学层 README、`IMPLEMENTATION.md` 和 `data/registry.json` 均将 REWORK4/acceptance4
标为唯一 current；REWORK、REWORK2、REWORK3 及 acceptance1/2/3 仅作历史证据。
registry 保持 `p1_go=false`、`p2_allowed=false`、`pilot_rerun_allowed=false`。

## 7. 主 Agent 验收事项

请复核：公开 CLI 是否确实只生成 acceptance4；manifest 的
`repair_tool_provenance` 与当前 `capture_provenance()` 是否严格相等；四 schema 路径与
hash、派生身份字段、source/旧审计保护性 hash、文档 current/historical 唯一性，以及
不得 P2 / 不得重跑 pilot 的边界。实施 Agent 不作最终研究判断。

已停止，等待主 Agent 验收。
