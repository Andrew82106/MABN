# P1 scientific benign implementation

## Architecture

新层通过只读导入 Phase A 的冻结八角色 workflow、permissions、response contracts、
mock database/sink、20 条虚构 fixture 和 transcript provider 接口。新增层自己负责：

- `ScientificEventRecorder`：append-only JSONL，科学层 phase/data-role/资格标签；
- `runner.py`：engineering dry 与唯一 real qwen pilot 的边界、manifest-first 和 预算；
- `runner.py`：既有 real qwen pilot 已完成且不得重跑；实现仍在任何未来尝试前要求当前 provenance 对应的 deterministic offline gate，并把剩余总预算传入 provider、在 in-flight 调用越过 60 分钟时 fail-closed，留下结构化 deadline failure；
- `validation.py`：事件/hash、标签、权限终点、sink 重算、状态隔离、provenance、
  artifact boundary 和 replay 的 fail-closed 检查；
- `replay.py`：只读取已记录结构化模型输出，零 Ollama、零网络；
- `reporting.py`：按权威 sink event 逐字段输出频数、比例和唯一候选结论类别。
- `audit.py`：只读原 run 的版本化 derived repair audit；manifest、identity、source hash 与报告均明确 `execution_mode=derived_repair_audit`、`is_new_p1_run=false`，原始 pilot artifact 不可覆盖。

## Current delivery boundary

当前四次返工派生审计为 `acceptance4-20260801T120000000000Z`，唯一当前交付说明为
[`REWORK4_DELIVERY.md`](REWORK4_DELIVERY.md)。工程状态保持 `p1_go=false`、
`p2_allowed=false`、`pilot_rerun_allowed=false`；源 pilot 未重跑，不启动 P2。

## Labels and data boundary

engineering dry 使用 `engineering_dry_run`、`eligible_for_scientific_analysis=false`、
`scientific_gate=NOT_STARTED`。真实 pilot 使用 `p1_gate_pilot`、
`eligible_for_scientific_analysis=true`、`scientific_analysis_scope=P1_GATE_PILOT_ONLY`、
`scientific_gate=P1_AWAITING_MAIN_AGENT_ACCEPTANCE`；不允许确认性、因果或后续阶段分析。

## Read-only reuse

所有上游文件以逐文件 SHA-256 和树 hash 写入 manifest，并显式记录恰好四个 `data/shared/p1_benign/schemas/*.schema.json` 文件及其 tree hash；模板和源码保留在诚实命名的其他 provenance map。新代码不回写 Phase A、P0、
qualification、shared fixture 或 `doc`。所有 generated paths 都由 `require_generated_path`
限制在 `paperAlpha/data/pre_exp1/p1_scientific_benign/` 内。

## Validation and replay

validator 对 manifest/events/outcomes/replay 做严格 JSON、contract、hash、标签和权威
工具状态重算；任何结构或语义矛盾返回 `passed=false`、非空 errors 和 CLI 非零。replay
使用 `TranscriptProvider` 重执行调度、权限与 mock 状态，并比较规范化完整事件轨迹。语义篡改测试会实际调用生产 `replay_run()` 写出 fresh replay，再由 validator 检查权威终点矛盾；preflight drift 只在工程临时/派生目录留下结构化零调用失败记录。
所有新生成 replay、validation、报告和派生 manifest 都携带完整数据资格标签。
