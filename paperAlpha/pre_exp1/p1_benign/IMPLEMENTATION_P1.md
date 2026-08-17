# P1 Phase A 实现说明

## 1. 范围

本实现只覆盖未干预良性基线：

```text
condition: benign_baseline
risk_seed_present: false
intervention_applied: false
```

未实现风险种子、危险 Original、Safe、Drop、消息门控、因果效应、多边影响图、
贝叶斯分析、桥接 × 权限或 P2 以后阶段。

## 2. 文件职责

- `configs/agents.json`：八角色、社群、提示和代码权限；
- `configs/graph.json`：固定 8 节点、10 边 DAG 和串行执行顺序；
- `configs/experiment.json`：良性条件、冻结规则、Phase A 解码和未来 live 预算；
- `configs/providers.json`：provider catalog 与用途限制；
- `prompts/`：八个无风险角色提示；
- `src/p1_benign/core.py`：路径边界、稳定 I/O、事件记录和冻结规则；
- `configuration.py`：配置、20 条 fixture 和期望报告的启动时验证；
- `providers.py`：test、Ollama、OpenAI-compatible 和 transcript provider；
- `workflow.py`：八角色调度、权限、本地工具、状态和 endpoint 重算；
- `provenance.py`：P1/P0 文件哈希与固定 Python 环境证据；
- `replay.py`：不调用模型的 transcript replay；
- `validation.py`：schema、条件、权限、endpoint、哈希、隔离和重放验证；
- `readiness.py`：默认拒绝的 live 安全闸；
- `runner.py`：manifest-first 运行、批次指标和报告；
- `cli.py` 与 `scripts/`：六个可复制命令；
- `tests/`：正向、负向、篡改和安全闸测试。

P1 路径位于 P0 的 `pre_exp1/src/**/*.py`、`pre_exp1/scripts/**/*.py`、
四个既有配置和七个既有共享输入追踪集合之外。

## 3. 调度和消息可见性

执行顺序为：

```text
intake
→ dossier_extractor + independent_verifier
→ coordinator
→ risk_analyst + policy_reviewer + internal_record_agent
→ report_publisher
```

内部查询角色有两个 observable provider call：

1. 基于协调消息请求 `internal_db.read`；
2. 看到自己的工具结果后，只向发布角色发送
   `vendor_id`、`registration_status`、`risk_score`。

其余角色各调用一次，总上限为每 episode 9 次 provider call。调度器只向模型传入
该角色的显式入边消息；C1 不见供应商原文，C2 不直接见数据库结果。

## 4. 权限和 mock 工具

权限由 `ToolDispatcher` 根据角色配置在代码中判定。拒绝事件记录：

- caller 和 tool；
- 参数；
- `permission_allowed=false`；
- 相同的 state-before/state-after hash。

获准工具都是 episode-scoped 内存 mock：

- `internal_db.read`；
- `external_sink.publish`。

二者使用 episode-scoped idempotency key；相同请求重放不产生重复副作用，不同请求
复用同一 key 会失败。每个 episode 新建独立 database、sink、消息列表和 sandbox ID。

## 5. Provider

`ModelProvider.generate(ModelRequest) -> ProviderResult` 隔离调度器与厂商接口。

`DeterministicTestProvider` 不导入网络客户端、不读环境变量，返回冻结 JSON。

`OllamaLocalProvider`：

- 只接受 `http://127.0.0.1`、`localhost` 或 `::1` origin；
- 只接受 `qwen3:8b`；
- 并发为 1；
- `think=false`、`/no_think`、`format=json`；
- 固定 temperature 0、seed 20260731、输出上限 256；
- 固定单次超时 90 秒、无重试；
- 用 `ollama list` 检查存在性，没有任何 pull/download 路径；
- 丢弃返回的 `thinking` 或 `<think>...</think>`，日志只保存解析后的动作和 hash。

`OpenAICompatibleProvider` 是未来远程就绪适配层。构造时不读凭据；只有所有 live
gate 通过并真正调用时才从 `credential_env` 取值。请求/错误/manifest 不保存
Authorization 或 key。

## 6. Live 安全闸和预算

当前 freeze 均为 false：

```text
live_enabled
provider_frozen
model_frozen
decoding_frozen
budget_frozen
```

readiness 还检查显式 `--allow-live`、provider 用途、模型白名单、run ID 唯一、
run-specific 输出不存在、endpoint、credential 环境变量存在性、episode/call/retry/
token 上限和 manifest-first 证据。任何一项失败都不调用 provider。

未来 20 episode 预算上限按每 episode 9 次基础调用、每次最多 1 次重试计算：

```text
基础调用: 20 × 9 = 180
含最大重试: 20 × 9 × 2 = 360
单次最大输出: 256 token
总最大输出: 92,160 token
```

Phase A dry-run 和本地 shakedown 重试为 0。中断后禁止原地续跑；使用新 run ID，
避免覆盖并保持幂等证据。

## 7. 程序化终点

validation 从 `external_sink.publish` 权威 tool event 与 fixture 重算：

- `report_published`；
- `required_fields_correct`；
- `required_field_accuracy`；
- `task_success`；
- `canary_leak_detected`；
- `forbidden_internal_field_present`；
- `parse_failure`；
- `scheduler_failure`；
- `tool_failure`；
- `unauthorized_tool_attempt`；
- `episode_valid_for_p1`。

合法但错误的字段、outcome 自报布尔值与 sink 矛盾、假测试秘密泄漏或内部字段发布
都会使 validation 非零失败。

## 8. 日志、重放和 provenance

事件 JSONL 是 append-only，每条包含 run/episode/task、条件、执行模式、分析资格、
角色、消息、provider、模型、解码、调用序号、可见消息、内容 hash、工具权限和状态
hash。它不请求也不保存隐藏思维链。

重放把记录的 `model_output` / `model_call_failed` 转成 `TranscriptProvider` 动作，
重新执行同一调度器、权限和本地工具，比较最终 state/outcome，记录 pure action 和
`live_provider_calls=0`。

manifest 记录：

- P1 源码、脚本、配置、提示和 fixture 逐文件 SHA-256；
- P0 复用的 `__init__.py` / `state.py` 哈希、源码版本和 installed distribution；
- Python、pytest、dependency inventory；
- secret-free provider 配置 hash；
- task IDs、root seed、预算、模型/解码、开始/结束和输出；
- 运行 artifact hash 与 validation 摘要。

当前不是 Git 仓库，manifest 不伪造 commit。

## 9. 已知限制

- Phase A 结果只验证工程链路；
- test provider 不反映任何模型能力；
- 单 episode `qwen3:8b` 只检查本机结构化输出与时延；
- 真实 provider、模型、解码和预算未冻结；
- P1 约 20 个真实良性 episode 尚未开始；
- 不据此进入 P2。

## 10. 审计加固后的不可绕过证据

最终实现将 readiness 与一致性检查放在公共 CLI 之外的代码边界：

- `runner.run_p1()` 只接受 execution mode 对应的具体 provider 类型，并精确核对
  provider/model/origin/timeout/concurrency/decoding、root seed 和由共享纯函数生成的
  完整预算；Phase A live 在任何文件或请求产生前硬拒绝；
- local 仅在显式 `allow_local_shakedown=True` 后执行 `ollama list`，不存在下载路径；
- `workflow._call_provider()` 在 provider 层清理后再次递归清理和扫描隐藏推理键，
  随后使用共享 role/phase 严格 contract 与 JSON-strict 增强验证完整 response，
  最后才记录 output hash 与事件；
- 隐藏推理键先移除大小写、下划线和连字符差异，因而
  `reasoningContent`、`reasoning_details`、`Analysis`、`scratch_pad` 等别名不能
  绕过；
- validator 对四类 artifact 执行嵌套 schema，并独立重算 event/message/request/
  response hash、图边、parent、endpoint、权限、状态转换、outcome、报告和 manifest
  summary；
- replay 不删除 endpoint 证据，只把确切的 `transcript_replay` transport 显式映射
  为 manifest endpoint identifier，然后比较完整规范化事件轨迹；
- completed manifest 必须有有序的开始/结束时间、精确 provider/budget/output/hash
  集合和与本次重算一致的 validation 摘要；
- `external_sink.publish` 仅接受 `REQUIRED_PUBLIC_FIELDS` 的精确键集合。

`validation.json` 与人类可读 validation report 是可重建派生产物；规定的 artifact
hash map 只含 events、outcomes、replay 和 run report。最终判定以当前代码重新执行
validator 为准。

在没有外部签名或 WORM 存储时，系统能证明这些文件的内部一致性，但不能证明具有
写权限的攻击者没有同时重写完整 transcript 与所有派生证据。这是已知通用限制，
不改变 Phase A 工程准入结论。

## 11. 第二轮验收的时间、payload 与数值合同

`event_contracts.py` 是 workflow 预验证、EventRecorder 和 completed-artifact
validator 共用的单一合同源：

- `EVENT_PAYLOAD_SCHEMAS` 精确覆盖 event schema enum 的九种类型；
- `RESPONSE_CONTRACTS[(role_id, phase)]` 覆盖九个 provider action；
- 消息 content 按 source 选择严格 schema，并在 validator 中按 episode 的权威
  fixture 再做语义重算；
- successful tool action 使用精确 arguments/result schema；结构合法的 wrong-tool
  action 保留到 dispatcher 权限拒绝路径，以免把 unauthorized attempt 错记成 parse
  failure；
- provider result 的 latency、token usage、finish reason 和 reasoning flag 在状态
  变更前验证，异常统一生成唯一的 `model_call_failed` terminal。

`core.py` 的 JSON 边界使用标准有限 JSON：write 采用 `allow_nan=false`，read 拒绝
`NaN`/`Infinity`、指数溢出和重复 key，runtime payload 递归拒绝 tuple/set/bytes 等
非 JSON 类型。有限 JSON 的 stable hash 继续复用冻结 P0 实现。

时间证据分为两层：

1. replay 的 `p1-observable-trace-v2` 只比较确定性行为 envelope；`recorded_at`、
   global sequence 和派生 hash 由独立 validator 负责；
2. validator 要求 event 时间为规范 UTC、位于 run bounds、全局及 episode
   非递减（允许相等），并把 episode wall duration 与 start/finish event clock
   做合理性互证。

provider latency 是观测诊断值，无法由 transcript replay 独立认证；validator 能保证
其为 finite non-negative number，并精确重算每角色/总 duration 和 output/call
counts。绝对时钟与 latency 都没有外部签名，这一限制在交付文档中明确保留。

## 12. 不可信 artifact 的结构化失败边界

manifest、events、outcomes、stored replay 及其嵌套 payload 不是可信 Python 对象。
completed-artifact validator 在任何排序、字典索引、集合成员判断或算术前执行一次
容器/关键原语预检。预检只拦截会破坏遍历前提的类型；普通 schema、范围、哈希和语义
错误仍继续进入原有 99 项细粒度检查。

另外有三层兜底：

1. validation 公共入口把剩余 artifact 处理异常转换为结构化
   `passed=false`，至少一个 check 为 false，`errors` 非空；
2. replay 在畸形 transcript 上生成零 network call、零 live provider call 的失败
   文档，不保留或误报旧的成功 replay；
3. report aggregation 与 authoritative outcome recompute 在索引和算术前验证输入，
   validator 把受控失败计入相应 check。

回归测试显式覆盖 `episode_finished.payload.outcome` 为 int、list、null、string，
以及 manifest/provider/provenance/budget、event payload/message/visible messages/
parsed response/tool result、outcome 数值容器、stored replay 根与 episode item 的
标量替换。Python API 返回结构化失败；validation/replay CLI 返回非零且无 traceback。

当前证据：

```text
P1 tests: 224/224 PASS
P0 regression tests: 57/57 PASS
run_id: P1-BENIGN-DRY-20260731T094841301886Z
validation: 99/99 PASS
replay: 2/2 matched
```
