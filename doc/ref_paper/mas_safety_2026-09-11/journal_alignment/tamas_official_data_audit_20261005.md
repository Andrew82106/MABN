# TAMAS 官方代码与数据审计（2026-10-05）

## 结论

TAMAS 的官方仓库可以直接下载，适合作为 MAS 安全的外部压力测试来源，但**不能把仓库里的静态 JSON 直接当作当前 API-only 监测器的运行轨迹**。静态文件是“给 Agent 的任务、角色和攻击配置”；监测器需要的是实际 API 运行中产生的消息、委派、工具调用、工具结果和时间顺序。

- 官方仓库：[microsoft/TAMAS](https://github.com/microsoft/TAMAS)
- ACL 正式论文：[ACL Anthology 2026.acl-long.1442](https://aclanthology.org/2026.acl-long.1442/)，DOI `10.18653/v1/2026.acl-long.1442`
- 代码许可：MIT
- 数据许可：Community Data License Agreement—Permissive 2.0（仓库中的 `LICENSE.CDLA-2.0`）

## 逐项核查结果

### 1. 发布内容与实际文件

官方仓库 `data/` 下实际有 6 个攻击目录、5 个领域，每个组合 10 条：

| 项目 | 审计结果 |
|---|---:|
| 静态攻击案例 | 300 条 |
| 攻击类别 | Byzantine、Colluding、Contradicting、DPI、IPI、Impersonation，各 50 条 |
| 领域 | education、finance、healthcare、legal、news，各 60 条 |
| 每条记录字段 | `agents`、`user query` |
| Agent 数量 | 2 个（3 条）、4 个（247 条）、5 个（50 条） |
| 运行轨迹字段 | 没有 |
| 评测用 benign 文件 | 仓库 `data/` 中未找到 |

论文/README 声称另有 100 条 harmless tasks；但当前官方仓库的 `data/` 目录没有相应的 benign JSON 文件。这个差异必须在复现实验中单独记录，不能默认为 400 条都已随仓库发布。

每条静态记录的 `agents` 元素只有 `agent_name` 和 `agent_description`，外加一个 `user query`。工具实现位于 `data/tools/autogen` 和 `data/tools/crewAI`；仓库中合计 207 个去重函数名。官方运行器支持 AutoGen/CrewAI 的不同编排方式，但输出是文本日志，不是本项目已定义的标准事件 JSONL。

### 2. 标签边界

攻击类别来自**目录名**，`is_adversarial` 是由数据构造得到的评测标签，不是模型运行时可见字段。部分攻击还写入了 Agent system description 或 user query，例如 Byzantine 的“故意不可靠”、DPI 的“忽略此前指令”等。这些属于测试配置，不能在监测器特征中直接使用。

本项目的边界处理如下：

1. `attack_type`、`is_adversarial`、源文件名和行号只放在 evaluator-only `labels.jsonl`；
2. 监测特征只来自实际运行时可观测事件；
3. 如果实验记录了 user query，应把它视为输入事件，而不是把目录名或预设攻击类型写入事件；
4. Agent 描述中的攻击指令属于配置，不进入运行时特征，以免“看答案”。

只读审计脚本 `paperAlpha/scripts/audit_tamas_release_v1.py` 会生成隔离的 `cases_public.jsonl` 和 `labels.jsonl`，并明确拒绝把攻击标签写入 public 投影。

### 3. 能否接入 API-only 监测器

可以接入，但需要两步：

1. 用官方 AutoGen/CrewAI runner 将静态案例实际运行在一个 API 模型上，并保存完整的可见消息、委派、工具调用、工具结果、错误和时间顺序；
2. 将这些日志转换成 `paperAlpha` 的 canonical runtime event schema，再运行两层 BN。攻击目录标签只在最后的 evaluator 端合并，用于计算指标。

当前静态发布包本身不能支持 AUROC/F1，因为它没有正常/攻击运行结果、工具调用时间线、权限状态、终态效果或人工确认标签。它适合作为**场景与攻击配置来源**，不应被写成已经完成的外部监测验证。

## 复现与投稿决策

- 作为外部迁移集：可用，优先选择一个固定框架、一个低成本 API 模型，并公开“可见前缀”和日志字段。
- 作为同信息 MAS baseline：需要先把官方运行日志规范化；不能直接拿 TAMAS 的 ERS/ASR 与本文 AUROC/F1 横比。
- 作为最终独立安全确认集：不够。官方标签是攻击构造标签，不是独立双盲人工对运行安全性的确认。
- 数据再分发：若将 TAMAS 原始数据或衍生文件放入发布包，必须随附 CDLA-Permissive 2.0 文本；本仓库只提交审计脚本和报告，不提交外部原始数据。

本审计使用 `microsoft/TAMAS` 当前公开仓库完成；下载时应记录 commit SHA，并在正式实验报告中固定版本。
