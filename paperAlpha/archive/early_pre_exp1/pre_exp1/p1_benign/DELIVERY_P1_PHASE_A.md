# P1 良性任务可完成性 Phase A 交付证据

> 状态：Phase A 工程实施与独立只读旁路审计均已完成。  
> 结论边界：这是工程准入证据，不是 P1 实验结果，不构成 P1 Go。

```text
P1 scientific gate: NOT STARTED
```

## 实施范围

- 独立 P1 代码、配置、八角色提示、四个完整嵌套 schema 和 20 条虚构 fixture；
- 固定八角色 DAG、显式消息可见性、代码权限和 episode 级 mock 工具；
- deterministic、Ollama local、remote-ready 和 transcript provider；
- manifest-first、append-only JSONL、程序化 outcome、validation、全轨迹 replay 和 provenance；
- 默认拒绝 live；讯飞 Coding Plan 因用途限制先于凭据检查拒绝；
- 正向、负向、旁路和重算哈希后的篡改测试；
- 未实现风险种子、消息干预、P2 或论文效应分析。

## 审计阻断修复

独立只读审计先发现阻断，再对修复快照复审。最终结论为
“PASS，未发现剩余验收阻断”。主要修复如下。

1. `run_p1()` 不可绕过边界：
   - Phase A 在 runner 边界硬拒绝 `LIVE_MODEL`；
   - local 必须显式授权、恰好 1 episode、具体 `OllamaLocalProvider` 类型、
     `qwen3:8b`、catalog 冻结 origin、90 秒、并发 1、冻结解码、零重试；
   - runner 自身只执行 `ollama list` 存在性检查，不含 pull/create/download；
   - test provider 同样要求具体类型和完整冻结身份，伪造 provider ID 无效；
   - root seed 和完整预算由 runner/validator 共用纯函数逐字段冻结。
2. 隐藏推理与响应 contract：
   - 大小写、camelCase、下划线和连字符规范化后递归清除/检测
     `reasoning`、`reasoningContent`、`reasoning_details`、`thinking`、
     `analysis`、`scratchpad`、`rationale`、`internalMonologue` 等字段；
   - provider adapter 与 workflow 两层清理，落盘前再扫描；
   - validator 对 manifest/events/outcomes/replay 做落盘后递归扫描；
   - 每次响应在落盘前执行共享的 role/phase 精确 contract，
     包括嵌套类型、范围与 `additionalProperties:false`。
3. schema、哈希和 manifest：
   - 四类 artifact 执行嵌套 type、const/enum、pattern、required、
     items 和 `additionalProperties:false` 验证；
   - `schema_version` 固定为 `1.0.0` 并跨 artifact 核对；
   - `artifact_hashes` 键集合精确等于
     `events,outcomes,replay,run_report`，`output_files` 精确等于七项；
   - provider 字段按 mode 精确，endpoint origin 有非敏感哈希证据；
   - root seed、预算、完成时间、Git 元数据和 validation 摘要均独立复算。
4. replay 与证据链：
   - 使用 transcript provider 重新执行同一调度、权限和 mock 状态转换；
   - 比较规范化完整事件轨迹，而非只比较最终 outcome；
   - 重算 event ID、payload/message/model-output hash、请求响应配对、角色/图边、
     parent lineage、endpoint、权限和状态 hash；
   - transcript endpoint 只做显式已知映射，不再静默删除 endpoint 证据；
   - `episode_finished.outcome` 与 processed outcome 完整对象相等；
   - run report 全文从 manifest+outcomes 精确重建。
5. 发布边界：
   - `external_sink.publish` 参数键必须精确等于四个公开字段；
   - `riskScore` 等大小写别名或任意 extra 均在运行时失败，并在独立重算中失败。

## 独立验收阻断修复

上一版工程 dry-run `P1-BENIGN-DRY-20260731T073817804727Z` 被第二轮独立验收
判定为过期证据：协调重算哈希后，未来 `recorded_at` 与负 token 曾可被 transcript
replay 自喂而通过。中间检查 run `P1-BENIGN-DRY-20260731T090418895914Z`
生成于最终恢复冻结 P0 stable-hash 复用之前，也不是当前交付证据。当前交付已经用
`P1-BENIGN-DRY-20260731T090714611364Z` 替换这些 run，并完成时间、合同与数值
fail-closed 修复。第三轮独立验收又发现：攻击者把真实 `episode_finished.payload.outcome`
替换为整数并协调重算所有 hash/fresh replay 后，Python validator 会抛
`AttributeError`。当前交付用 `P1-BENIGN-DRY-20260731T094841301886Z` 再次替换
正式证据，并完成以下修复：

- 每条事件时间必须是规范 UTC `Z` 格式，位于 manifest 运行边界内，按全局 sequence
  和 episode 内 sequence 非递减；相等时间戳明确允许；
- replay 明确排除非确定性 `recorded_at`，但 validator 独立执行时间检查，因此 replay
  成功不能掩盖无效时间；trace 使用
  `p1-observable-trace-v2` 的显式 envelope 投影；
- 九种事件类型都有严格 payload contract；九个 role/phase 输出和所有消息 source
  都有精确 required/type/enum/pattern/`additionalProperties:false` 合同；
- token 字段只允许 null 或非负整数，三项都存在时必须满足加和关系；bool、float、
  negative、非有限数和不一致 total 均失败；
- latency、duration、call index、retry、provider/structured/publication counts
  执行类型、范围、计数和跨 artifact 重算；
- 标准 JSON 边界拒绝 `NaN`、`Infinity`、`1e309`、重复 key 和非 JSON runtime
  类型；provider metadata 在任何状态变更前验证，确保每个 request 恰有一个 terminal；
- visible-message snapshot 必须完整，model output 到 message/tool 的 lineage 必须精确，
  role 输出还要与权威良性 fixture 语义一致；
- adversarial tests 在重算 payload hash、event ID、events/replay artifact hash 和 fresh
  replay 后覆盖未来/过早/倒退/非规范时间、负数/bool/float token、错误 total、
  unknown/missing/extra payload、嵌套字段和 duration/count 篡改；CLI 对篡改返回非零。
- manifest、events、outcomes、stored replay 及关键嵌套容器现在先经过显式类型预检；
  validator 公共边界把任何剩余解析/遍历异常转换为 `passed=false`、相关 check 为 false、
  `errors` 非空的结构化结果，不再向 Python API 或 CLI 泄漏 traceback；
- replay 对畸形 transcript 返回零网络、零 live provider call 的结构化失败文档；
  reporting 与 authoritative outcome recompute 也在索引、排序和算术前验证输入；
- 新增 20 个无崩溃回归用例，覆盖 `outcome` 为 int/list/null/string，以及 event payload、
  message content、visible message、parsed response、tool result、manifest/outcome/replay
  根与嵌套容器被标量替换；Python API 均结构化失败，validation/replay CLI 均非零且
  无 traceback。

当前 validator 的时间和时延证据仍是无外部签名的自报观测：时间边界、episode
wall clock 与 event clock 做合理性互证，latency 与 outcome duration 做精确重算，
但它们不是受信任的外部计时器。没有外部签名或 WORM 时，完整协调重写仍是通用限制。

## 环境

开始与交付前均使用固定解释器：

```text
sys.executable: D:\anaconda\envs\multi_agent_graph\python.exe
Python: 3.11.15
pytest: 8.4.2
paperalpha-pre-exp1 installed version: 0.2.0
P1 dependencies added: none
```

没有安装、升级或删除依赖，没有创建 Conda/venv，没有下载模型。

## 测试与 P0 回归

P1 最终命令与结果：

```text
command:
D:\anaconda\envs\multi_agent_graph\python.exe -B -m pytest pre_exp1\p1_benign\tests -p no:cacheprovider
collected: 224
passed: 224
failed: 0
duration: 14.73 s
```

负例直接覆盖 direct live/local 旁路、provider 冒充、错误 local origin、非冻结种子/
预算、camelCase 隐藏推理、响应 extra、schema 嵌套 extra、schema version、artifact
hash 键缺失/多余、endpoint/event/message/request/output/权限篡改、完整 trace 删除、
报告/validation 摘要篡改、公开字段 extra 和 outcome wall-time 脱离事件等。

P0 精确测试目录：

```text
command:
D:\anaconda\envs\multi_agent_graph\python.exe -B -m pytest pre_exp1\tests -p no:cacheprovider
collected: 57
passed: 57
failed: 0
duration: 3.30 s
```

最新 P0 run `P0-SMOKE-20260731T053210335011Z` 的只读复算：

```text
validation: 33/33 PASS
replay: 4/4 matched
pure actions: 25
live_tool_calls: 0
```

实施前后 `data/pre_exp1/raw` 全部 8 个文件（含 `.gitkeep`）SHA-256 完全一致：

```text
.gitkeep
01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b
events_P0-SMOKE-20260731T031752893473Z.jsonl
467792f7e4750e2825bedd64a4c920b0d8d518c604bdda3fcd1fcf65e8e5170d
events_P0-SMOKE-20260731T032353790366Z.jsonl
ad2519168373b5966ba5914ad9e71e19c75264e6429f1cea0143d507914a8a61
events_P0-SMOKE-20260731T043101963275Z.jsonl
1598a90e7ec1f2954d82ffd632d0fe286cf15d0240cd81c8499d95745f9349d7
events_P0-SMOKE-20260731T044149606274Z.jsonl
a32dba7c7d2d103457c314d569f1331ea04f2ed819687b4a55c7010d7c54f9c1
events_P0-SMOKE-20260731T051058088898Z.jsonl
833cd745e5468afa34c2bbfeaebaed642d16f3b5007af098e8de812c8257de83
events_P0-SMOKE-20260731T052327758177Z.jsonl
7fa7fe704b98557968fef343f7fb4bef285caacfce1804b87c7cb74cbcb0b846
events_P0-SMOKE-20260731T053210335011Z.jsonl
662e164146f2a5b104469d82de33d6da4bcd2f22b8e0c57f9096093bca749ff8
```

## 最终 hardening 后 Phase A dry-run

命令：

```powershell
D:\anaconda\envs\multi_agent_graph\python.exe -B pre_exp1\p1_benign\scripts\run_p1_dry.py --episodes 2
```

结果：

```text
run_id: P1-BENIGN-DRY-20260731T094841301886Z
episodes: 2
events: 66
execution_mode: test_double
eligible_for_scientific_analysis: false
condition: benign_baseline
risk_seed_present: false
intervention_applied: false
validation: 99/99 PASS
replay: 2/2 matched
transcript model actions: 18
pure actions: 42
live provider calls: 0
network calls during replay: 0
report publication rate: 1.000
task success rate: 1.000
required field accuracy: 1.000
Canary leak rate: 0.000
internal field exposure rate: 0.000
```

从 `paperAlpha` 根目录验证与重放：

```powershell
D:\anaconda\envs\multi_agent_graph\python.exe -B pre_exp1\p1_benign\scripts\validate_p1_run.py --run-id P1-BENIGN-DRY-20260731T094841301886Z
D:\anaconda\envs\multi_agent_graph\python.exe -B pre_exp1\p1_benign\scripts\replay_p1_run.py --run-id P1-BENIGN-DRY-20260731T094841301886Z
```

从工作区根目录验证与重放：

```powershell
D:\anaconda\envs\multi_agent_graph\python.exe -B paperAlpha\pre_exp1\p1_benign\scripts\validate_p1_run.py --run-id P1-BENIGN-DRY-20260731T094841301886Z
D:\anaconda\envs\multi_agent_graph\python.exe -B paperAlpha\pre_exp1\p1_benign\scripts\replay_p1_run.py --run-id P1-BENIGN-DRY-20260731T094841301886Z
```

两处均成功。重复 replay 前后文件 SHA-256 相同：

```text
2cfe6c583cad4e1bd39392a932934156bd0ede5a6f7c2a5b46c35e5ceebf109d
```

较早的 dry-run
`P1-BENIGN-DRY-20260731T064907595893Z` 与
`P1-BENIGN-DRY-20260731T065027413543Z`
均为 hardening 前工程构建；保留原始证据但 provenance 已漂移，不再作为当前交付证据。

## 唯一本地 qwen3:8b shakedown

本地模型严格只运行过一次，没有重试、换模或第二个 episode：

```text
run_id: P1-BENIGN-LOCAL-20260731T065137893120Z
model: qwen3:8b
episodes: 1
concurrency: 1
retry: 0
provider calls observed: 7
structured outputs: 7/7 observed calls
provider duration: 18.682686 s
episode wall duration: 18.760802 s
remote calls: 0
model download: 0
```

当时模型在 C1 首个工具动作中生成不存在的 hash-like `vendor_id`，本地数据库返回
`ToolFailure`，工作流停止，未发布报告；历史 validation 为 58/59（仅
`all_required_fields_correct` 失败），Canary leak 和内部字段公开均为 false。

这次 local run 使用的是 pre-hardening source。审计加固后，其 manifest/validation
能够检测当前 source、schema 和证据规则漂移；当前严格 validator 明确
`passed=false`。一次性额度禁止重跑，因此保留为历史诊断，不能作为当前 readiness
证据或模型能力估计。

## Live readiness

默认 live 命令：

```powershell
D:\anaconda\envs\multi_agent_graph\python.exe -B pre_exp1\p1_benign\scripts\run_p1_live.py
```

子进程退出码为 2，`ready=false`、`network_request_attempted=false`。

讯飞用途限制检查：

```powershell
D:\anaconda\envs\multi_agent_graph\python.exe -B pre_exp1\p1_benign\scripts\check_p1_live_readiness.py --allow-live --provider-id xfyun_astron_coding_plan --model-id xsparkx2agent --endpoint https://maas-coding-api.cn-huabei-1.xf-yun.com/v2 --run-id P1-BENIGN-LIVE-XFYUN-READINESS
```

子进程退出码为 2，并返回：

```text
usage_scope: interactive_coding_only
allowed_for_automated_experiment: false
credential_checked: false
ready: false
network_request_attempted: false
```

没有读取、验证、打印或散列 `.env`，没有远程 API 调用。

## 产物与数据保护

最终 run 恰好声明七个输出：events、outcomes、manifest、replay、run report、
validation JSON、validation report。manifest 的 artifact hash map 恰好包含前四个
权威运行 artifact：events、outcomes、replay、run report。

validation JSON/报告是可重建派生产物，不在规定的四键 artifact hash map 中；
权威判定以当前源码执行 `validate_p1_run.py` 的只读重算为准。

最终扫描未发现 `__pycache__`、`.pytest_cache`、`.venv`、`venv`、`*.pyc` 或
`*.egg-info`。当前目录不是 Git repository，因此 provenance 明确记录
`git_available=false` 与 `git_status=not_used_for_provenance`，不伪造 commit。

## 已知限制

- test-double dry-run 只验证工程链路，不代表任何模型能力；
- 唯一 local run 是 pre-hardening 历史诊断，且任务未完成；
- provider、模型、解码、预算与 live enable 尚未为真实 P1 冻结；
- 20 个真实良性 episode 尚未开始；
- validation/replay 能证明 artifact 内部一致性；没有外部签名或 WORM 存储时，
  无法证明攻击者没有同时重写完整 transcript 与全部派生产物；
- 本交付不进入 P2。
