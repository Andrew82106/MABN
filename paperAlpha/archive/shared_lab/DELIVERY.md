# 共享实验平台科学实验准入层交付说明

状态：实现完成，等待主 Agent 独立验收；不得据此开始论文实验或声称平台已经可用于论文数据。

## 环境与范围

- Conda：`multi_agent_graph`
- 解释器：`D:\anaconda\envs\multi_agent_graph\python.exe`
- Python：`3.11.15`
- 全量测试：36 项通过，其中既有 22 项全部保留通过
- 新增依赖：0；仅使用 Python 标准库
- 执行边界：离线 fake transport、本地确定性场景工具，无模型、provider、HTTP、socket、
  GPU、Ollama、讯飞或 Codex gateway 调用
- 凭据边界：未读取、列举、散列或回显 `paperAlpha/.env`
- 修改边界：仅 `shared_lab/`；`paperAlpha/`、`doc/`、根 README、`.gitignore` 和历史工件
  未被本任务修改

## 三个准入接口

### 1. 默认拒绝的逐角色观察

- `src/lab_kernel/observation.py` 根据场景的 `observations` 声明投影公共路径。
- 每个角色只获得角色/任务提示、声明公共字段、自己的 inbox 和自己的工具结果。
- 私有记录、完整 state、其他角色消息及其他角色工具结果默认不可见。
- 投影和工具结果均深拷贝；Agent 修改观察对象不能改变内核状态。
- 受信任 observation hook 可运行，但最终输出只能保留内核基础字段和该角色显式声明的
  `hook_fields`，不能意外恢复全局状态。
- 场景加载器拒绝缺失角色策略、未知角色、重复/非法/不存在状态路径和非法 hook 字段。

### 2. 可信工具注册与内核授权

- `src/lab_kernel/tools.py` 实现严格工具注册表、封闭参数 schema、handler 加载、实时权限
  检查、结果摘要和允许效果验证。
- 模型侧 `AgentRequest` 没有 capability 字段。内核根据工具名从可信注册表推导唯一所需
  capability，调用者无法自报或伪造权限。
- `internal_record.read` 仅允许内部记录角色读取，假测试秘密只在成功调用后进入其私有
  `tool_results`。
- `public_report.publish` 仅允许发布角色调用，是模型控制路径写公开报告与完成状态的唯一入口。
- 未知、畸形、类型错误、额外参数、缺权和 handler 失败均 fail-closed，在业务状态变化前
  写入 `action_denied`。
- handler 使用状态深拷贝；只允许返回声明的效果种类。工具事件记录可信权限、请求、结果
  哈希摘要和确定性效果。

原有 `Kernel.dispatch(Action)` 继续服务于受信任内核、hook 和研究者控制的防御动作；模型
适配器只产生 `send_message` / `call_tool` 的 `AgentRequest`，不能到达 `set_value`、
`append_value` 或任意状态路径。

### 3. 可注入模型适配器

- `src/lab_kernel/model_agent.py` 定义 `ModelTransport` 与 `ModelAgent`；生产 provider 未实现。
- `src/lab_kernel/testing.py` 提供唯一已使用 transport：`FakeModelTransport`。
- 请求只包含冻结模型身份、角色/任务提示和隔离观察，使用 JSON object 响应约束。
- 严格解析拒绝非 JSON、重复键、NaN/Infinity、非单对象、多余字段、reasoning/refusal、
  原生 tool calls、未知动作、任意写状态、非法消息路由、未知/畸形/越权工具、错误模型身份、
  timeout 和基础设施失败。
- 工件策略显式支持 `none`、`hashes`、`raw`；默认 `hashes` 仅保存请求哈希、响应哈希和
  动作类型，不保存隐藏思维链，只有显式选择 `raw` 才保存可见请求与动作响应。

## 四条端到端轨迹

1. 正常完成：发布角色通过授权工具提交无风险报告；`task_success=true`、
   `hazard_detected=false`，终态可确定回放。
2. 显式传播：内部记录角色授权读取 canary，经允许的 `e10` 消息边发送给发布角色，再由
   发布工具写入报告；`task_success=true`、`hazard_detected=true`，程序判定器明确识别危害。
3. 必要边阻断：受信任防御动作切断 `e10` 后，读取仍可发生，但秘密消息被拒绝并记账；
   无风险报告正常完成，`hazard_detected=false`。
4. 直接旁路：普通角色尝试任意 `set_value`、内部读取、公开发布及未知工具，全部在状态变化
   前拒绝，终态保持不变并逐项产生 `action_denied`。

## 测试与篡改证据

公开命令：

```powershell
cd shared_lab
$env:PYTHONPATH = "$PWD\src;$PWD"
conda run --no-capture-output -n multi_agent_graph python -B -m unittest discover -s tests -v
```

结果：`Ran 36 tests`，`OK`（最终复跑耗时 `3.775s`；耗时随机器负载变化）。

关键覆盖：

- 现有 22 项账本、回放、溯源、权限映射、场景声明和三类 hook 测试继续通过；
- fake 模型的合法消息与合法工具请求均端到端执行，模型请求中不存在 canary；
- fake transport 失败矩阵覆盖畸形、越权、身份错误、timeout 和基础设施失败；
- 对工具名、actor、required capability、arguments、result hash 或 effect 进行协同篡改并重算
  整条事件链，可信收据验证和/或生产 `replay()` 仍 fail-closed；
- 回放审计为 `model_calls=0`、`tool_executions=0`，且原 transport/tool 执行计数不增加；
- 动态三角色、不同任务、不同图、不同工具的极小场景无需修改内核即可加载、运行和回放；
  它只证明解耦，不进入论文数据。

测试使用临时目录，工具 handler 只产生内存/临时账本中的确定性状态效果，没有网络、外部
发布或真实工具副作用。

## 已知限制与未实现项

- 没有真实模型 transport、live CLI、credential loader、重试/限流或 provider 兼容性证明。
- 没有运行任何论文实验，也没有生成论文数据或实验性结论。
- 收据是调用方持有的内容寻址可信锚，不是数字签名或远程不可变存证。
- 本轮只建立离线科学实验准入层候选；是否准入及后续范围必须由主 Agent 独立验收决定。

等待主 Agent 验收。
