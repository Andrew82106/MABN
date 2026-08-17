# P1v2-Q0 实现边界

## 冻结输入

- `configs/q1_protocol.json`：未来 Q1 的唯一候选配置；
- `schemas/response_contract.schema.json`：Q 专属严格 JSON 契约；
- `prompts/public_response_contract.txt`：实际未来使用的良性任务包装；
- `fixtures/`：32 张 Screen、96 张 Confirmation 任务卡和 17 个负向 case；
- `permissions/public_sink_rule.json`：publisher-only 的四字段公开端规则；
- `provenance/reference_contract_sources.json`：四个允许的 P1v2-A 契约来源路径与 SHA-256。

运行器只读取本目录自己的冻结输入；validator 独立比对四个允许的 P1v2-A 来源身份，以便发现来源漂移。

## 离线执行

`src/p1v2_qualification/fake_provider.py` 是唯一的 provider 实现。它只返回确定性 fixture 输出，或确定性异常/超时模拟；没有网络库、模型 CLI 或可执行 live adapter。

Q0 run 对 128 张正确任务卡和 17 个负向 case 建立 append-only event/transcript ledger。每个请求只有一个终态：

```text
model_output | model_output_rejected | model_call_failed
```

### Strict response branches

The frozen response schema has two strict, mutually exclusive JSON branches. The normal branch is exactly `role`, `episode_id`, `decision="allow_continue"`, and the four-field `public_report`. The reject branch is exactly `role`, `episode_id`, `decision="reject"`, and a non-whitespace `rejection_reason`; it cannot contain `public_report` or any extra field.

The `valid_reject` fixture is contract-valid and identity-aligned, but does not satisfy this batch's allow task card. It therefore ends as `model_output_rejected` with `failure_code=semantic_reject` and never reaches the public sink. The equally contract-valid `wrong_role_valid_reject` and `wrong_episode_valid_reject` fixtures are intentionally different: task alignment is checked first, so they end as `role_mismatch` and `episode_mismatch`, never as `semantic_reject`. This remains distinct from malformed JSON, unexpected fields, or an invalid `decision="reject"` paired with a public report.

只有验证通过的 publisher 记录可进入 mock public sink；sink 记录不包含任务 ID、角色或任何额外字段。

## 验证与重放

- validator 从 run ID 派生全部工件路径，拒绝不可信 manifest 路径；
- 静态输入、源代码树、入口脚本、来源身份、历史保护树和输出 hash 都独立复核；
- validator/replay 均为 fail-closed、结构化结果，且 replay 只消费既有 ledger；
- The validator also checks the derived `reports/` registry. Before the current run has been replayed, zero current reports is allowed; after report creation, exactly one `Current Q0 run ID` must exist and match this run. Every other report must carry its own historical run ID and the pre-rework, not-for-downstream-decisions warning.
- The delivery report must reproduce the recomputed Screen/coordinator, Screen/publisher, Confirmation/coordinator, and Confirmation/publisher counts, plus one terminal-count line for every negative fixture. Validator checks those human-readable report values against the ledger rather than trusting the report or outcomes JSON.
- 公开入口固定为项目指定根。测试副本只能通过显式 `test_mode=True` 在 `tests/.tmp_q0_*` 内运行，公开 CLI 无法注入外部路径。

测试不改写真实源码、schema、prompt、fixture 或历史根；它们只在自动清理的隔离副本中进行漂移检查。
