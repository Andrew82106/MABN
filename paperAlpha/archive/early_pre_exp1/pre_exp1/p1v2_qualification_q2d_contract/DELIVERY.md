# Q2-D P1 动作契约离线准入交付（供主 Agent 验收）

## 1. 环境与范围

- sys.executable：D:\anaconda\envs\multi_agent_graph\python.exe
- Python：3.11.15
- 所有命令均使用 conda run --no-capture-output -n multi_agent_graph python -B ...。
- 本轮只实现离线 fake transport；没有模型/provider、网络、socket、HTTP、工具执行、公开发布或任何 P1/P2 run。
- 未读取、列举、哈希或回显 credential 文件；未创建 live CLI 或 credential loader。
- 未修改 doc/、长期路线、历史 P1、Q2-B 或 Q2-C 代码/数据，Q2-D 源码没有 import 历史包。

## 2. 实现内容

- 独立冻结 Q2D-P1-ACTION-CONTRACT-1.0.0、profile、18 张良性卡、18 个精确 schema、18 个 instruction 和 publisher 四案例 fixture。
- 严格 adapter 拒绝非 JSON、重复 key、NaN/Infinity、缺失/额外字段、错误类型/枚举/语义、native tool_calls/function_call、拒答和推理字段。
- fake transport 只返回内存中的预设响应，tool_execution_calls=0、network_calls=0、env_reads=0。
- Screen 全通过后才进入 Confirmation；Screen 失败会阻断 Confirmation。
- validator/replay 只持久化安全 hash、状态类别和计数，不保存输入、响应正文、工具参数、凭据或推理内容。

## 3. 测试结果

固定环境下：44 passed。

覆盖内容包括：

- 18 张卡逐卡成功；
- 九种动作位置与两阶段门控；
- Coordinator、Internal Record Agent、Report Publisher 的精确权限/参数；
- 非 JSON、重复 key、非有限数、缺失/额外字段、错误 kind/tool/参数/语义；
- native tool/function call、拒答、推理字段；
- HTTP/timeout、metadata 缺失、模型身份、deadline、预算；
- fake Canary 仅存在于内存，泄漏时 fail-closed；
- manifest schema、ledger 结果类别、provider identity、敏感检测和调用计数的单点/重链篡改；
- 无历史包 import、无 socket/urllib/requests live surface。

## 4. fake run 与 validation → replay → validation

唯一 fake run：

Q2D-FAKE-20260802T120000000000Z

三次公开 CLI 结果：

| 阶段 | 退出码 | 结果 |
|---|---:|---|
| validate | 0 | passed=true，18/18 |
| replay | 0 | passed=true，matched_cards=18 |
| validate | 0 | passed=true，18/18 |

安全计数：completion_calls=18、tool_execution_calls=0、network_calls=0、env_reads=0。
Screen 9/9 后 Confirmation 9/9，发布案例资产覆盖：
active/low/approve、active/medium/review、active/high/reject、inactive/reject。

## 5. 证据与隔离

- 运行目录：paperAlpha/data/pre_exp1/p1v2_qualification_q2d_contract/runs/Q2D-FAKE-20260802T120000000000Z/。
- 运行工件只有 manifest、ledger、validation、replay、run report；持久化内容不含响应正文、敏感值、工具参数或凭据材料。
- 受保护历史树前后字节不变；Q2-D 只写入本目录及对应 data 目录。
- 测试产生的 __pycache__、.pytest_cache 和隔离篡改副本已清理；保留唯一正式 fake run。

离线结论：ready_for_q2d_live。这只表示主 Agent 可以考虑另立任务书申请真实 Q2-D
调用；不表示 P1 Go、P2 Go 或任何研究结论。本交付完成后停止，等待主 Agent 验收。

