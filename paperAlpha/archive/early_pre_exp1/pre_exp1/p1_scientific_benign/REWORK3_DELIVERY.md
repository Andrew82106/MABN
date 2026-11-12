# P1 第三次返工实施报告（供主 Agent 验收）

本文件是历史第三次返工说明，不是当前/latest/authoritative 交付；当前唯一说明为
`REWORK4_DELIVERY.md`，当前派生审计为 `acceptance4-20260801T120000000000Z`。
本轮是既有 P1 pilot 的最小范围返工，不是新实验；实施 Agent 不宣布 P1 Go 或最终研究 No-Go。历史工程边界固定为
`p1_go=false`、不得 P2、既有真实 pilot 不重跑；完成本报告后停止等待主 Agent 验收。

## 1. 固定环境与禁止项

- Conda：`multi_agent_graph`；Python：`3.11.15`；解释器：
  `D:\anaconda\envs\multi_agent_graph\python.exe`。
- installed distribution provenance：`paperalpha-pre-exp1==0.2.0`。
- 未调用 Ollama、任何 provider、远程 API 或网络；未读取、散列、复制或输出
  `paperAlpha/.env`；未运行第二次 P1、P2、warm-up 或模型下载。
- `doc/` 只读；原 pilot、P0、Phase A、qualification、shared fixture/schema 和旧审计
  均未改写。除本目录既有代码/测试/说明更新外，只新建一个派生审计目录。

## 2. A：schema 正向 oracle 独立于生产常量

修改文件：
`D:\Projects\Multi_Agent_Graph_Analysis\paperAlpha\pre_exp1\p1_scientific_benign\tests\test_scientific_benign.py`。

测试侧现在定义 literal `SCHEMA_TRUTH_SET`，未导入生产
`SHARED_SCHEMA_RELATIVE_PATHS` 构造 expected。正向测试逐项核对 capture provenance、
真实文件 hash 与 tree hash；新增 `test_schema_truth_set_exposes_production_constant_drift`，
将生产常量替换为存在但错误的 `task_template.json` 后，独立 truth set 明确暴露集合漂移。
原有 missing/replacement/extra 三类 fail-closed 测试保留。

四个且仅四个 shared schema（相对 `D:\Projects\Multi_Agent_Graph_Analysis\paperAlpha\`）及 SHA-256：

| 路径 | SHA-256 |
|---|---|
| `data/shared/p1_benign/schemas/event.schema.json` | `282fd3c0f190970fde61af6f42959772f9bcc26ade9d47199208c064bb7ad83d` |
| `data/shared/p1_benign/schemas/manifest.schema.json` | `4698f563639952a3492dc14fd739c748a66f3f10f455b2a3b5bec85748a4b786` |
| `data/shared/p1_benign/schemas/outcome.schema.json` | `e8853d16d1295eec82fe55d8db20bafa52665ceb9ea14a1ff8351ed2063fc8b3` |
| `data/shared/p1_benign/schemas/replay.schema.json` | `21356cf86c875978fe4cb46933c525d000f72d5b11005379b7d04322d467418e` |

四文件 tree hash：`ee9c47d5012061fb738255b7e6efc5f7c6a7a7118648cd69403231b6580ca927`。

## 3. C：in-flight hard deadline

修改文件：同一 scientific benign 测试文件；生产实现为
`src/p1_scientific_benign/runner.py` 的 `_DeadlineProvider`。

`test_inflight_provider_call_is_rejected_at_deadline_and_stops_next_episode` 使用带
`timeout_seconds=90.0` 的 timeout-aware fake。受控时钟在进入 provider 调用时将剩余预算设为
`85.0` 秒（小于 90）；断言 fake 实际收到的 timeout `>0 且 <=85.0`，调用返回后恢复到
`90.0`。fake 随后把时钟推进到 `3601` 秒，跨界结果被拒绝，manifest 写入
`failure_stage=deadline`、`accepted_as_success=false`，`generate_calls=1`、
`partial_outcomes=1`，没有进入下一个 episode。

## 4. E：派生审计身份与 source/audit mode

修改文件：
`D:\Projects\Multi_Agent_Graph_Analysis\paperAlpha\pre_exp1\p1_scientific_benign\src\p1_scientific_benign\audit.py`，并新增机器可检验身份测试
`test_derived_repair_audit_identity_is_machine_checkable`。

历史时点唯一新增目录：

`D:\Projects\Multi_Agent_Graph_Analysis\paperAlpha\data\pre_exp1\p1_scientific_benign\pilot\acceptance_repair\acceptance3-20260731T232500000000Z\`

该目录由公开 validation CLI 的 `--repair-id` 路由生成，重解释 source run
`P1-BENIGN-PILOT-QWEN3-20260731T132033188154Z`，不是新 P1 run。manifest、
`derived_audit_identity.json`、`source_hashes.json`、`DERIVED_AUDIT.md`、
`run_report.md` 和 `p1_gate_report.md` 均可机器检查并明确：

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
p1_go=false
```

审计 artifact SHA-256：

```text
manifest.json               09ebd20bbcb1362b4b69071c9afebf5e61d7af0e8a88d521f1e0b3cf4235a74f
validation.json              02b582c287a98f5e6ec42d6a094cdd005f1156978257e92248de414339469417
replay.json                  b18334bc32af558725cb7101b958744ad7e4f07fa494b8590203d5d9fa590ee8
run_report.md                012cdd4272381d510c1a888a4ecfc87c74f3f213d6a76bc521d5a63df956040a
p1_gate_report.md            d84d58413749b8b0004d74ea42a513712fd2b05b8afa52581cf8dff928e1860b
source_hashes.json           8ca45cda0dc764d53cde9708957973c7f83fedc8eee48a79b2c315eaebc4da20
derived_audit_identity.json  9c0a011a8bfa4fec38956620dfa97e6479e1edabad030b299fba8b066e471d25
DERIVED_AUDIT.md             120a5b6265a998f4a373ea6d9228d2d2130b2fbb1572a2c4c23da2cc1cbf4871
```

`acceptance2-20260731T230556448674Z` 保留原状，仅标为历史；其 manifest hash
`3958d8635104aae623f96592a9e4409cae20f3d46ea1327c5d04ac95cf2f27c3` 未改变。
更早两个 acceptance manifest hash 也保持不变：

```text
acceptance-20260731T1415/manifest.json  05c0002051785d9c9e02e397b1709bd7667d89d04251bf9a5279d7e7021cf976
acceptance-20260731T1422/manifest.json  3d2832e415858d6ac98f545203bbef055c7dc01c758f32b675d08d4729d11099
```

## 5. 派生审计指标与只读 CLI

source pilot 的权威 sink/event 重算为：`vendor_id=19`、`registration_status=19`、
`risk_level=19`、`recommendation=17`，合计 `74/80`；`ProviderParseError=1`、
structured outputs `179/180`、report published once `19/20`、task success `17/20`、
transcript replay `20/20`、replay live/network/provider calls `0/0/0`。这些是既有
source run 的历史事实，不是新实验结论。

公开只读 CLI（均固定环境、未写回 source）结果：

| 命令/对象 | 退出码 | 结果 |
|---|---:|---|
| scientific pilot validation（source run） | 1 | 历史 validation 失败，结构化错误，无 traceback |
| scientific pilot replay（source run） | 0 | 20/20 matched，live/network/provider `0/0/0` |
| historical engineering dry `P1-BENIGN-PILOT-DRY-20260731T155623039469Z` validation | 0 | 全部检查通过，四 schema provenance 正确 |
| historical engineering dry replay | 0 | 20/20，transcript model actions 180，live/network `0/0` |
| `--repair-id acceptance3-20260731T232500000000Z` | 1 | 预期保留 source validation 失败并成功写入派生审计 |

历史只读复验：Phase A `P1-BENIGN-DRY-20260731T094841301886Z` validation/replay
均 PASS（replay 2/2）；qwen qualification `P1-BENIGN-QUAL-QWEN3-20260731T121840716022Z`
均 PASS（3/3）；ministral qualification `P1-BENIGN-QUAL-MINISTRAL3-20260731T121957481982Z`
均 PASS（3/3）；P0 `P0-SMOKE-20260731T053210335011Z` 均 PASS（4/4），均为
live/network `0/0`。P0 raw 八项 hash 全部与既有记录匹配。

## 6. 四组回归测试

从 `D:\Projects\Multi_Agent_Graph_Analysis\paperAlpha` 使用
`conda run --no-capture-output -n multi_agent_graph python -B` 执行，均退出码 0：

```text
pre_exp1\p1_scientific_benign\tests       28 passed
pre_exp1\p1_benign\tests                  224 passed
pre_exp1\p1_model_qualification\tests     32 passed
pre_exp1\tests                              57 passed
```

## 7. 文档、registry 与保护性检查

- 在第三次返工历史时点，`paperAlpha/README.md`、`data/README.md`、科学层 README、
  `IMPLEMENTATION.md` 和 `data/registry.json` 曾统一指向本 `REWORK3_DELIVERY.md` 与
  `acceptance3-20260731T232500000000Z`；这些现均为历史，当前已迁移至 REWORK4/acceptance4。
  `DELIVERY.md`、`REWORK_DELIVERY.md`、`REWORK2_DELIVERY.md` 均明确为历史。
- registry 当前字段为 `p1_go=false`、`p2_allowed=false`、`pilot_rerun_allowed=false`。
- 原 pilot 核心 hash 未变：events
  `8de3e520fe86b3b1be74104d2aa1762ef8ce7b8775be5033f9cc3c51f050d38e`，manifest
  `66c48af2d881f298bfa438cf7c3e8f07dde29b6a8a329382cc4d9525875f7ab1`，outcomes
  `b8887f9a2cb175e2ad885b77f315f30e710ada3321e251a31647c633e7d6e2ee`；size/mtime
  快照未变（events 1072786、manifest 11934、outcomes 28964 bytes）。
- 未发现 `__pycache__`、`.pytest_cache`、`.pyc`、venv、egg-info 或 P2 artifact；未访问
  `.env`，未修改 `doc/`。

## 8. 主 Agent 验收事项

请独立检查：A 的 literal truth set 能否暴露 production constant 漂移；C 的 fake 是否
确实收到剩余 85 秒以内 timeout 并恢复 90 秒、且跨 deadline 结果被拒绝；E 的新审计目录
是否由生产 CLI 写入且内部身份文件/manifest/两个报告都明确“derived audit / not a new
P1 run”；acceptance2 与 source pilot hash 是否保持不变；四组测试、只读 CLI、文档/registry
一致性与 `p1_go=false` 边界是否符合任务书。

已停止，等待主 Agent 验收。
