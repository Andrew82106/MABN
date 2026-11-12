# P0 实现说明

## 范围

本实现是确定性 scripted/stub sandbox。它不连接模型 API、真实数据库、网络或
外部发布服务，不包含真实敏感数据，不实现处理效应估计、贝叶斯模型、SBM、
POMDP 或 EVSI。

## 执行模型

固定 DAG、节点、边、允许关系、门控边和执行顺序均从 `graph.json` 加载。
启动时验证节点/边唯一性、端点、拓扑顺序、门控边和 treatment 配置一致性；
每次发送前验证消息端点与允许边。四条 smoke 轨迹共享相同图驱动执行器：

- 正常安全：冻结 Original 请求只选取允许公开的字段；
- 危险 Original：冻结 Original 请求包含假测试秘密字段；
- Safe：危险请求在焦点边替换为冻结的公开字段模板；
- Drop：焦点消息替换为冻结的空操作消息，任务记录为信息不足。

## 权限与本地工具

权限映射在 `configs/agents.json` 声明，并由 `PermissionEnforcer` 在每次工具
调用前执行。只有内部数据查询 Agent 能读取虚构内部记录；只有报告发布 Agent
能写入内存 sink。拒绝会抛出 `PermissionDenied` 并写事件。

`LocalMockTools` 的状态属于当前 episode，可重置。读取和发布均需要幂等键；
相同键的重复调用返回已保存结果而不重复改变状态。所谓“发布”只是在当前
episode 的 Python 状态字典中加入记录。

## 门控

四种输入模板 `normal_original`、`dangerous_original`、`safe`、`drop` 都来自
`data/shared/treatment_templates/message_treatments.json`，具有稳定 ID、版本和
可重算内容哈希。`MessageGate` 的门控边来自已验证的 `graph.json`。Original
保持输入不变；Safe 和 Drop 使用同一冻结注册表。非门控边保持原样。

## 日志、状态与重放

原始事件按行追加到 `data/pre_exp1/raw/events_<run_id>.jsonl`。事件包含任务书
要求的公共字段；不会记录隐藏思维链。状态使用规范 JSON 和 SHA-256 计算稳定
哈希。episode 初始状态包含唯一的 `FAKE_TEST_SECRET::...` 字符串，clone 使用
深复制。

每条事件都在 `details.canonical_payload` 中保存可公开重算的规范 payload；
事件生成器与验证器共同使用同一规范化和 SHA-256 规则。验证器还会将该 payload
与事件类型对应的独立结构化字段核对，覆盖 episode start/completion、message
gate/delivered、tool read/publish、task status change 和 outcome recorded。

状态变化被编码为有限的纯数据 `replay_action`。重放器从 `episode_started`
事件恢复初始状态，再按顺序应用这些动作；它不会调用 scripted Agent 或
`LocalMockTools`。重放报告记录纯动作数量和实时工具调用数；失败注入测试进一步
证明 replay 不会触发数据库或发布工具。重放结果必须与原运行最终状态哈希一致。

## Provenance 与统一环境

manifest 记录可定位的配置和共享输入逐文件 SHA-256、冻结模板内容哈希、工作区
状态、随机种子及实时捕获的 Python 环境。源码追踪覆盖
`pre_exp1/src/**/*.py` 与 `pre_exp1/scripts/**/*.py`：先记录相对路径到
SHA-256 的映射，再由该映射稳定计算总 source tree hash。

Python 环境证据明确区分源码声明版本、通过
`importlib.metadata.version("paperalpha-pre-exp1")` 读取的已安装 distribution
版本，以及完整依赖 inventory、条目数和 inventory hash。验证器会重算并要求
三类版本证据内部一致。smoke 只允许在
`D:\anaconda\envs\multi_agent_graph\python.exe` / Python 3.11.15 下运行。
验证器重新计算当前输入、源码、环境，并核对 event、outcome、manifest、replay
的 run、episode、scenario、treatment、任务状态和最终哈希；泄漏与任务结果还会
分别从 replay 最终 publication sink 与最终 task status 独立重算。

## 数据生命周期

- `raw/`：首次创建后只追加的事件 JSONL；
- `manifests/`：运行追踪、配置和源码哈希、输出与验证结果；
- `interim/`：可由 raw 与代码重建的 replay/validation 结果；
- `processed/`：验证后生成的 episode outcome JSONL；
- `reports/`：smoke、验证和测试的人类可读摘要。

工作区没有 Git 仓库时，manifest 会记录 `git_available: false`，同时保留上述
逐文件源码哈希和确定性源码树总哈希。
