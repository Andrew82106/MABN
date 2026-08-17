# Supersession note (2026-08-01 rework)

`P1V2-READINESS-DRY-20260801T000001000000Z` is retained unchanged as a
historical pre-rework artifact and is not current. The current reworked
offline artifact is `P1V2-READINESS-DRY-20260801T000002000000Z`; see
`P1V2_A_REWORK1_DELIVERY.md`. This note does not make a P1/P2/model decision.

# P1v2-A 离线准入交付（历史 pre-rework 记录）

## 顶层结论（历史记录）

本交付是离线 `p1v2_readiness_dry` 准入工件，不是新的 P1 run，不是 P2，也不提供模型选择、
P1 Go 或 P2 Go 结论。本历史记录在当时使用的离线工件为
`P1V2-READINESS-DRY-20260801T000001000000Z`。

## 证明什么、怎么证明、产出什么

确定性 fixture 文本经过严格 JSON 契约：12 个案例中仅 1 个完整公开响应可以写入内存 mock
public sink；非 JSON、空输入、duplicate key、NaN、Infinity、缺字段、内部/fixture secret/
嵌套额外字段、角色错配和 episode 错配均有机器可读拒绝码且不发布。该历史工件冻结配置、
schema、prompt、fixture 和执行代码 provenance；manifest 哈希 events、outcomes、validation、
replay 与 report（manifest 自身因自指问题不进入自己的 hash map）。公开 validate/replay
均只读、fixture resolution 为 0。

## 离线运行与公开 CLI（历史记录）

- `run_readiness_dry.py --run-id P1V2-READINESS-DRY-20260801T000001000000Z`：退出码 0；
  `passed=true`、validation/replay 均 true。
- `validate_readiness_dry.py --run-id P1V2-READINESS-DRY-20260801T000001000000Z`：退出码 0；
  `passed=true`，fixture/model/external-provider/network 调用计数均为 0。
- `replay_readiness_dry.py --run-id P1V2-READINESS-DRY-20260801T000001000000Z`：退出码 0；
  `passed=true`、replayed case count=12，fixture/model/external-provider/network 调用计数均为 0。
- 运行未调用真实模型、provider 或 network；未读取环境凭据文件；没有 P2。

历史 preliminary artifact `P1V2-READINESS-DRY-20260801T000000000000Z` 未删除或覆盖。
在最终源码 provenance 下对它的 public validation 退出码为 1（`provenance_mismatch`、
`artifact_hash_structure_invalid`）；它不是当前工件，也未被表述为通过。

## 正反例与负向验证

独立测试覆盖：严格 JSON 拒绝、坏响应不能到达 mock sink、配置值变更实际改变 dry workflow、
新增 traceable config 文件导致 provenance FAIL、合法布尔/decision 篡改后即使重新计算
outcome hash 仍由 fresh production replay 拒绝、合法但错误的 SHA-256 provenance 值导致
FAIL、坏 artifact CLI 非零且无 traceback、CLI 实际 dispatch 到 production replay path、
validate/replay 的文件快照不变。

## 固定 Python 环境与依赖

- `sys.executable`：`D:\anaconda\envs\multi_agent_graph\python.exe`
- Python：`3.11.15`
- 新依赖：无。
- 开始与交付的 `python -m pip list --format=json` 均为 10 个 distribution；规范化 SHA-256
  均为 `645bee5241abc6c0fdd76c634a9d0c3f869e721507ff01bc6c16432deae95c79`。

## 回归测试

均使用 `conda run --no-capture-output -n multi_agent_graph python -B -m pytest ... -q -p no:cacheprovider`：

- `pre_exp1/p1v2_benign/tests`：24 passed，退出码 0；
- `pre_exp1/p1_benign/tests`：224 passed，退出码 0；
- `pre_exp1/p1_model_qualification/tests`：32 passed，退出码 0；
- `pre_exp1/tests`：57 passed，退出码 0。

## 旧 P1 保护性检查

`readonly_tree_before.json` 与 `readonly_tree_after.json` 的 tree SHA-256 均为
`3f5a36c4d547875586bd19329582d74e8fc2e489fc50765d0e05ffe48da6432b`；依赖清单比较通过。
以下受保护 hash 在交付复核时与开始前一致：

- source events：`8de3e520fe86b3b1be74104d2aa1762ef8ce7b8775be5033f9cc3c51f050d38e`
- source manifest：`66c48af2d881f298bfa438cf7c3e8f07dde29b6a8a329382cc4d9525875f7ab1`
- source outcomes：`b8887f9a2cb175e2ad885b77f315f30e710ada3321e251a31647c633e7d6e2ee`
- acceptance4 manifest：`0d33423be407bc09f72db1d561c29201f95c66470265d2093d5dcbf0889c4636`

## 明确未做的事情

未运行或选择任何模型；未重跑、覆盖、补写、重标或删除旧 P1/acceptance artifact；未创建 P2、
风险种子、Original/Safe/Drop 或因果分析；未修改 `doc` 或旧 P1 命名空间。

## 需要主 Agent 验收的事项

请独立复跑 README 的 current CLI、检查 manifest 全部身份字段与全生成物 hash、检查
`environment/` 中的前后只读树和依赖清单、复核四项保护性 hash，并决定是否仅进入模型选择评审。
本实施方不作该决定。
