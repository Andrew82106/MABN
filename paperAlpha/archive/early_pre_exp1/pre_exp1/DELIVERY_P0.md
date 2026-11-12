# pre_exp1 二轮验收后小返工交付清单

日期：2026-07-31  
范围：三个 P0 阻断项及最终验收发现的 tool payload 证据链漏检补丁。

## 最终验收 payload 证据链补丁

本补丁只修改 `tool_read_internal_record` 与
`tool_publish_local_sink` 的 canonical payload 权威来源核对：

- read 事件从 `details.tool_name`、`details.idempotency_key` 和
  `replay_action.vendor_id` 重建完整 request，再与 canonical payload 的
  `request` 及 `result` 做整对象比较；
- read 同时核对 replay action 类型、agent、vendor、idempotency key、result，
  并要求 `request_fingerprint` 等于完整 canonical request 的稳定哈希；
- publish 事件从 `details.tool_name`、`details.idempotency_key` 和
  `replay_action.report` 重建完整 request，再与 canonical payload 做整对象比较；
- publish 同时核对 `local_only=true`、replay action 类型、agent、report、
  idempotency key，并要求 `request_fingerprint` 等于完整 canonical request 的
  稳定哈希。

新增隔离协调篡改测试覆盖：

- read canonical request 的 `vendor_id` 被修改并重算正确 payload hash；
- read canonical request 的 `idempotency_key` 被修改并重算正确 payload hash；
- publish canonical request 的 `idempotency_key` 被修改并重算正确 payload hash；
- read 与 publish 的 replay `request_fingerprint` 与 canonical request 不一致；
- 上述每个用例都检查 `passed=false`、
  `event_payload_hashes_match=false`、可定位错误和 CLI 退出码 1。

## 阻断项与实现

1. 合法但语义矛盾的篡改仍会通过
   - 每个 episode 以 `episode_started.treatment` 为主 treatment，核对全部中间
     event、门控 requested/applied treatment、completion、outcome 和 replay。
   - 从 replay 最终 publication sink 与该 episode 的虚构 `test_secret` 独立重算
     `leak_detected`。
   - 从 replay 最终 `task_status` 独立重算 `task_success`。
   - 将重算结果同时与 outcome、raw outcome action、replayed state、冻结 expected
     condition 及 manifest 汇总布尔值核对。
   - event 生成与 validation 共用 canonical payload 哈希规则；validation 逐事件
     重算并核对独立结构化字段。
2. installed distribution 版本证据失真
   - 在固定 Conda 环境执行 editable reinstall。
   - manifest 分别记录源码声明版本、实时读取的 installed distribution 版本、
     完整依赖 inventory、inventory hash 和条目数。
   - validation 检查三类版本证据内部一致。
3. 源码哈希漏掉执行脚本
   - source file hashes 覆盖 `pre_exp1/src/**/*.py` 与
     `pre_exp1/scripts/**/*.py`。
   - manifest 保存可定位的相对路径到 SHA-256 映射，并由该映射计算总 source
     tree hash。

## 新增文件

- `pre_exp1/src/pre_exp1/event_payload.py`：共享 canonical event payload 与哈希规则。
- `pre_exp1/DELIVERY_P0.md`：本交付清单。
- 新 run `P0-SMOKE-20260731T051058088898Z` 的 7 个正式产物，均位于
  `data/pre_exp1`。
- payload 补丁新 run `P0-SMOKE-20260731T053210335011Z` 的 7 个正式产物，
  均位于 `data/pre_exp1`。

## 修改文件

- `pre_exp1/src/pre_exp1/event_log.py`：写入 canonical payload 并共享哈希实现。
- `pre_exp1/src/pre_exp1/gate.py`：记录可独立核对的门控输入/输出。
- `pre_exp1/src/pre_exp1/models.py`：manifest 增加逐文件源码哈希字段。
- `pre_exp1/src/pre_exp1/provenance.py`：实时 distribution 版本、依赖 inventory、
  `src` + `scripts` 源码追踪。
- `pre_exp1/src/pre_exp1/sandbox.py`：向 manifest 写入逐文件源码哈希。
- `pre_exp1/src/pre_exp1/validation.py`：增加 treatment、泄漏、任务结果、payload、
  manifest 汇总、版本证据和逐文件源码的语义验证；最终补丁增加 read/publish
  完整 request 与 fingerprint 核对。
- `data/shared/schemas/run_manifest.schema.json`：同步 manifest 新证据字段。
- `pre_exp1/tests/conftest.py`：测试报告记录实际解释器和两个版本来源。
- `pre_exp1/tests/test_manifest_inputs.py`：验证环境证据与源码覆盖规则。
- `pre_exp1/tests/test_validation_tamper.py`：增加七类隔离负向篡改测试和五个
  read/publish 协调篡改测试实例。
- `pre_exp1/IMPLEMENTATION.md`：说明新的验证与 provenance 规则。
- `data/pre_exp1/reports/test_results_rework_2026-07-31.md`：本轮测试摘要。

## 精确运行命令

```powershell
conda run -n multi_agent_graph python -B -m pip install -e ".[test]"
conda run --no-capture-output -n multi_agent_graph python -B -m pytest -p no:cacheprovider
conda run --no-capture-output -n multi_agent_graph python -B pre_exp1\scripts\run_p0_smoke.py
conda run --no-capture-output -n multi_agent_graph python -B pre_exp1\scripts\validate_run.py --run-id P0-SMOKE-20260731T053210335011Z
conda run --no-capture-output -n multi_agent_graph python -B -m pre_exp1.cli replay-run --run-id P0-SMOKE-20260731T053210335011Z
conda run --no-capture-output -n multi_agent_graph python -B paperAlpha\pre_exp1\scripts\validate_run.py --run-id P0-SMOKE-20260731T053210335011Z
conda run --no-capture-output -n multi_agent_graph python -B -m pre_exp1.cli replay-run --run-id P0-SMOKE-20260731T053210335011Z
```

前五条在 `paperAlpha` 执行；后两条在工作区根目录执行。

## 实际环境与结果

- `sys.executable`：`D:\anaconda\envs\multi_agent_graph\python.exe`
- Python：`3.11.15`
- pytest：`8.4.2`
- source version：`0.2.0`
- installed distribution version：`0.2.0`
- 新增或变更依赖：无；仅重新安装现有 editable 项目
- pytest：collected 57，passed 57，failed 0
- payload 补丁新 run_id：`P0-SMOKE-20260731T053210335011Z`
- validation：从两个工作目录运行均 `passed=true`
- replay：从两个工作目录运行均 `passed=true`，pure actions 25，
  live tool calls 0

## 数据位置与保护

所有正式生成数据只写入：

`D:\Projects\Multi_Agent_Graph_Analysis\paperAlpha\data\pre_exp1`

负向篡改仅发生在测试临时副本。已有 run 未被删除、覆盖或用于篡改，包括
`P0-SMOKE-20260731T051058088898Z` 和主 Agent 独立验收生成的
`P0-SMOKE-20260731T052327758177Z`。

## 已知限制与未完成内容

- 当前仅为确定性 scripted/stub P0 sandbox，不接入真实 LLM、网络、数据库或外部
  发布服务。
- 仅使用虚构静态数据和假测试秘密。
- 未实现贝叶斯模型、SBM、POMDP、EVSI、处理效应估计或正式统计分析。
- 未运行正式实验，未修改研究方向；本结果不构成实验性研究结论。
- 最终验收仍由主 Agent 独立完成。
