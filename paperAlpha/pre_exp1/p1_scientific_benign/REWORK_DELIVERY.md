# P1 正式良性批次：第一次验收返工历史交付

本文件记录第一次**工程、验证、报告和文档返工**，是历史交付，不是当前交付，也不是新的 P1 实验。
本文件及二次、三次返工说明均为历史证据；当前唯一交付见 [REWORK4_DELIVERY.md](REWORK4_DELIVERY.md)。本轮返工没有调用
Ollama、远程 API 或任何 provider，没有重跑真实 batch，没有启动第二次真实 P1 或 P2，
也没有读取、散列或输出 `paperAlpha/.env`。`doc/` 全部保持只读。实施 Agent 不在此宣布
P1 Go、最终 No-Go 或进入 P2；本文件交给主 Agent 独立验收。

## 1. 修改内容与修复证据

### A. 公开入口和数据角色路由

- `src/p1_scientific_benign/core.py` 增加按 run ID 的唯一角色路由，并在读取 manifest 后交叉
  校验目录、`data_role`、`execution_mode`、run ID 前缀和完整标签；不一致即 fail-closed。
- `src/p1_scientific_benign/cli.py` 及公开脚本的 `validate`/`replay` 入口默认只读，不接受任意
  输出目录；`--repair-id` 只允许向新的版本化派生审计目录写入。
- 真实 pilot 的只读入口不会写回原 pilot，也不会创建根目录
  `p1_scientific_benign/interim/` 下的错误 validation 产物。工程 dry 和 pilot 均按各自目录
  正确解析。

### B. 字段门槛和候选结论

- `reporting.py` 从权威 `external_sink.publish` event args 与冻结的 expected reports 逐字段
  重算四个公开字段，而不是以整条报告布尔值乘四。
- 真实 pilot 的派生审计重算为：`vendor_id=19`、`registration_status=19`、`risk_level=19`、
  `recommendation=17`，合计 **74/80**（原报告的整条报告口径为 68/80）。
- 测试覆盖 3/4、2/4、0/4 部分正确单元，并只允许一个预定义候选类别。当前派生审计写入
  `候选返工（实现交付未验收）`，`p1_go=false`，不把它当作 Go 或第二个 batch。

### C. 标签、schema provenance 和说明一致性

- replay、validation、run report、gate report 和派生 manifest 统一携带 phase、condition、
  risk/intervention、data role、科学分析资格、`scientific_analysis_scope`、P1 gate 资格、
  confirmatory/causal 资格和 `scientific_gate`。
- provenance 现在覆盖实际执行源码、脚本、配置、图、角色、提示、权限、20 episode fixture、
  共享 schema、P0/Phase A/qualification 复用文件和固定环境；共享 schema 递归纳入 tree hash。
- 派生 manifest 明确记录历史事实：**原 pilot manifest 未覆盖 schema provenance，原 manifest
  未被改写**。`paperAlpha/README.md`、`data/README.md`、`data/registry.json`、本层
  `README.md`、`DELIVERY.md` 和 `IMPLEMENTATION.md` 已统一为当前真实 pilot、
  `P1_AWAITING_MAIN_AGENT_ACCEPTANCE`、返工中且不得 P2。

### D. fail-closed 防线和负向测试

- 语义篡改测试保持 JSONL 合法，并同步重算事件/文件 hash，使 replay 表面自洽，但让
  `task_success` 或权威终点与 outcome 矛盾；validator 会结构化失败。
- 新增数据角色错配、模型 metadata/version 漂移、缺离线 gate、受控时钟超 60 分钟等负向测试。
- 未来真实 runner 在任何 provider 调用前要求当前代码/冻结输入对应的 deterministic offline
  gate；已有真实 pilot 时拒绝第二次 run；总墙钟时限实际执行并在失败 manifest 中保留证据。

## 2. 固定环境与离线命令

所有 Python 均使用：

```text
Conda: multi_agent_graph
Interpreter: D:\anaconda\envs\multi_agent_graph\python.exe
Python: 3.11.15
pytest: 8.4.2
installed distribution: paperalpha-pre-exp1==0.2.0
```

以下命令均从 `D:\Projects\Multi_Agent_Graph_Analysis\paperAlpha` 根目录执行，离线回归均退出码 0：

```powershell
conda run --no-capture-output -n multi_agent_graph python -B -m pytest pre_exp1\p1_scientific_benign\tests -p no:cacheprovider  # 20 passed
conda run --no-capture-output -n multi_agent_graph python -B -m pytest pre_exp1\p1_benign\tests -p no:cacheprovider             # 224 passed
conda run --no-capture-output -n multi_agent_graph python -B -m pytest pre_exp1\p1_model_qualification\tests -p no:cacheprovider  # 32 passed
conda run --no-capture-output -n multi_agent_graph python -B -m pytest pre_exp1\tests -p no:cacheprovider                   # 57 passed
```

还完成了历史证据的只读函数级复验：P1 Phase A validation/replay、qwen qualification
validation/replay、ministral qualification validation/replay 和 P0 validation/replay 均通过；
没有启用写 manifest 或写 output 的选项。

当前离线门证据为 `P1-BENIGN-PILOT-DRY-20260731T141935514990Z`：validation 通过、replay
`20/20`、transcript model actions `180`，live/provider/network calls 均为 `0`；它位于
`engineering_dry/`，不是真实 P1 batch。

## 3. 原 pilot 的只读 CLI 复验

run ID：`P1-BENIGN-PILOT-QWEN3-20260731T132033188154Z`

```powershell
conda run --no-capture-output -n multi_agent_graph python -B pre_exp1\p1_scientific_benign\scripts\validate_scientific_run.py --run-id P1-BENIGN-PILOT-QWEN3-20260731T132033188154Z
# exit 1：历史 pilot validation FAIL；无 traceback、无写回

conda run --no-capture-output -n multi_agent_graph python -B pre_exp1\p1_scientific_benign\scripts\replay_scientific_run.py --run-id P1-BENIGN-PILOT-QWEN3-20260731T132033188154Z
# exit 0：20/20 matched，live/network/provider calls=0；无写回
```

validation 的失败项被结构化报告为历史标签不一致、179/180 structured outputs、provenance
drift 以及原 manifest 缺少 shared schema provenance；这些历史事实没有通过改写原始 artifact
消除。

## 4. 版本化派生修复审计

权威本轮审计目录：

```text
paperAlpha/data/pre_exp1/p1_scientific_benign/pilot/acceptance_repair/acceptance-20260731T1422/
```

其 manifest 记录 `derived_from_run_id`、所有原始 artifact 的实际 SHA-256、原始 manifest
记录的 hash、当前修复工具和共享 schema provenance、完整资格标签、`no_provider_calls=true`
与 `network_calls=0`。该目录不是新 batch，且不覆盖较早的派生目录
`acceptance-20260731T1415`；后者作为历史派生证据保留。

派生审计关键事实：

| 项目 | 结果 |
|---|---:|
| episodes | 20/20 |
| provider calls（原 pilot 事实） | 180 |
| structured outputs（原 pilot 事实） | 179/180 |
| `ProviderParseError` | 1 |
| report published once | 19/20 |
| task success | 17/20 |
| 四字段权威正确数 | **74/80** |
| 字段分项 | vendor_id 19；registration_status 19；risk_level 19；recommendation 17 |
| replay | 20/20 matched |
| 派生审计 live/network/provider calls | 0/0/0 |
| candidate conclusion | 候选返工（实现交付未验收） |
| `p1_go` | false |

## 5. 原始 pilot 字节保护

开始和交付复核得到相同 SHA-256；以下文件均未被覆盖、移动或删除：

| 原始 artifact | SHA-256（前） | SHA-256（后） |
|---|---|---|
| `raw/events_...132033188154Z.jsonl` | `8de3e520fe86b3b1be74104d2aa1762ef8ce7b8775be5033f9cc3c51f050d38e` | 同左 |
| `manifests/manifest_...132033188154Z.json` | `66c48af2d881f298bfa438cf7c3e8f07dde29b6a8a329382cc4d9525875f7ab1` | 同左 |
| `processed/outcomes_...132033188154Z.jsonl` | `b8887f9a2cb175e2ad885b77f315f30e710ada3321e251a31647c633e7d6e2ee` | 同左 |

此外，原 pilot replay、validation、run、gate、preflight 和 validation report 的 hash 也被
写入派生 `source_hashes.json` 并复核不变；P0 raw 八项 SHA-256 检查为全部匹配。

## 6. 候选门槛与未解决事项

本轮只交付返工实现与证据链修复。既有 pilot 的 `ProviderParseError=1`、structured output
`179/180`、validation 未通过和历史 provenance/标签缺口仍然存在；因此本文件不改变其研究
事实，不宣称 P1 Go，也不替主 Agent 宣布最终 No-Go。74/80 只是按修复后权威 sink/event
口径得到的候选门槛指标，不能抵消上述硬失败项。

请主 Agent 独立复验：公开 CLI 路由和只读性、派生审计 hash/标签/provenance、原始文件保护、
四套 pytest 退出码、P0 八项 hash、历史 validation/replay 结果，以及是否接受“候选返工（实现
交付未验收）”这一实现交付标记。返工后不开始真实模型批次、P2 或任何外部调用。

完成本文件后实施 Agent 停止，等待主 Agent 验收。
