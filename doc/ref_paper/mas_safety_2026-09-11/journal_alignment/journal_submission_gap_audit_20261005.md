# 投稿门槛与实验缺口审计（2026-10-05）

本审计把当前结果与候选期刊的公开范围、近邻论文和可复现性要求逐项对齐。它不改模型、不改冻结主结果，也不把期刊论文的任务分数冒充本文 baseline。

## 1. 目标定位

### 首选：Expert Systems with Applications（ESWA）

Elsevier 官方范围明确包含风险评估、多智能体系统、知识管理和智能系统测试，并要求方法有真实创新、清晰的标准优化表述和有说服力的比较；官方页面还提醒不要只用已有概念换名或用隐喻包装算法。

因此本文应把贡献写成一个可验证的“观测证据融合机制”：

- 规范工作流线提供可读的角色、权限和预期关系；
- 运行时线只沿实际观察到的消息/委派传播证据；
- 两线的变量、未知证据上下界、路径解释和融合训练均可复算；
- MAS 系统级增益必须通过去掉跨 Agent 关系的同信息基线来证明。

官方范围：[ESWA](https://shop.elsevier.com/journals/expert-systems-with-applications/0957-4174)。

### 条件候选：Information Fusion

官方范围强调多源、多过程、多层信息融合，明确欢迎融合架构、算法、态势感知、不完整信息、实时计算和信息安全应用。因此本文只有在“规范证据 + 动态消息证据 + 不确定性边界”被形式化成真正的信息融合算法时才适合投该刊；如果只是把两个分数送入普通 Logistic 回归，不足以支撑该定位。

官方范围：[Information Fusion](https://shop.elsevier.com/journals/information-fusion/1566-2535)。

### 安全参照：IEEE TDSC

TDSC 更适合对照威胁模型、协作拓扑、攻击传播、低误报和可复现安全实验。它可以作为安全论证参照，但不能把攻击成功率直接和本文风险检测率混在一张数值表里。

官方入口：[IEEE TDSC](https://www.computer.org/csdl/journal/tq)。

中科院一区/二区目前仍没有取得学校采用版本的官方登录条目。现有机构页面和二级页面只能写成“候选证据”，不能在投稿稿件中写成已认证分区。

## 2. 当前已满足的门槛

| 门槛 | 当前证据 | 判断 |
|---|---|---|
| 双线模型可审计 | `independent_mas_journal_v1/MODEL_SPEC.md` 定义规范线、运行线、融合和禁止字段 | 已满足开发要求 |
| 同信息强基线 | runtime logistic、graph/local logistic、可靠性加权 Bayesian、规则/污点路径 | 已有；还不是所有近邻论文的源码复现 |
| 任务/拓扑隔离 | 4,000 条轨迹、186 topology families、五折 family-disjoint OOF | 已满足合成开发协议 |
| 指标完整性 | F1、AUROC、AUPRC、Brier、ECE、5% FPR、聚类 bootstrap | 已满足主要统计报告要求 |
| 白盒边界 | 不读取 API 权重/隐藏状态；输出因素、边、路径、上下界 | 已满足“监测器白盒”表述 |
| API 可运行性 | LANYUN 与本地端点均完成迁移/失败保留/拓扑 smoke | 已满足链路审计，未满足外部效度 |
| 可复现材料 | 公开投影、协议、manifest、失败记录、命令入口 | 基本满足；提交前需整理发布清单 |

## 3. 关键缺口（按投稿阻断程度排序）

### P0-1：缺少独立安全确认集（阻断最终安全结论）

当前主集标签来自合成生成器；两个 API 条件使用同一个手工 authority-payment simulator，标签仍是预先写入的策略谓词。它们能证明迁移和链路，不足以证明对真实 MAS 风险有效。

必须补：

1. 至少 100 个完整 episode、至少 4 个 workflow/topology family；
2. 至少两个独立模型/端点条件，端点只能作为分组变量；
3. benign 与 violating 样本在采集前冻结比例；
4. 两名独立标注者分别记录是否违规、首个违规事件、机制类别、证据 ID 和理由，先算 Cohen’s κ 再仲裁；
5. 监测器只能看到公开 workflow、公开策略和动作前日志；隐藏 prompt、标签、终态效果不能进入特征；
6. 失败请求和缺失事件必须保留，不能只保留成功样本。

协议草案已在 `paperAlpha/submission/EXTERNAL_CONFIRMATION_PLAN.md`，但当前尚未形成可用数据。

### P0-2：API-only 的端到端成本没有闭环

已有 API 传输 token/延迟和 Python 监测路径 p95，但尚未把“原始 API 文本 → 语义事件抽取 → 图更新 → BN 推理”作为一个完整流水线测量。投稿中不能把 1–2 ms 的监测器计算写成系统延迟。

必须分别报告：

- API 请求数、输入/输出 token、传输 p50/p95；
- 语义抽取模型、版本、提示词、抽取延迟和失败率；
- 图/BN 更新 p50/p95、进程级 RSS/峰值内存；
- 每 episode 总成本以及抽取失败时的 lower/upper/abstain 行为。

### P0-3：早期预警尚不足以作为卖点

当前固定低误报点的前缀及时召回较弱，且负例误报随前缀长度变化。论文只能说“支持前缀诊断并报告不确定性”，不能写成稳定的 early-warning guarantee。若要把预警作为主贡献，必须在独立确认集上固定 FPR 后报告 lead time、及时召回和误报置信区间。

### P1-1：原生 MAS 近邻尚未完成同信息量复现

SentinelAgent、Node Contribution Backpropagation、AgentMonitor 等可以作为原生 MAS 近邻，但当前结果主要是 protocol-compatible proxy 或目标不一致的迁移。主表应明确区分：

- 同信息、同标签、同划分的可运行 baseline；
- 只能作 related-work 实验参照的论文数值；
- 仅作上界/下界或目标不同的外部审计。

否则“超过文献方法”的表述不成立。

### P1-2：双线独立增益仍需更强的消融

已有 runtime-only、workflow-only 和融合结果，但应在最终表中固定报告：

- 去掉规范线；
- 去掉运行时关系传播，仅保留 Agent 独立分数；
- 去掉知识模板/权限因子；
- 去掉跨 Agent 边但保持事件数量和特征维数；
- 仅使用全轨迹统计的匹配 Logistic/树模型。

每个消融都必须用相同 family split、阈值和 bootstrap 单位。

### P1-3：拓扑外推和自适应攻击覆盖不足

合成集拓扑族数量足够做开发泛化，但 API smoke 只有少量 chain/fork/join/review。期刊版本至少应在四种拓扑分别报告性能、覆盖率和失败率，并加入通信方式变化（直接转发、摘要、投票/汇聚）或扰动边置信度的压力测试。攻击者适应性测试应明确监测器可见字段和攻击者不可见字段，不能把离线标签规则当作攻击结果。

### P1-4：不确定性尚不是校准安全保证

unknown/redacted 的 lower/point/upper 是观测缺失下的区间诊断，不是概率覆盖保证。若保留“置信度/不确定性”作为贡献，需在独立确认集报告 coverage-risk 曲线、abstention cost 和 ECE；否则正文将其称为“epistemic missing-provenance diagnostic”。

## 4. 投稿前最小验收表

以下条件全部通过后，才能把稿件称为“可投候选”；否则只能称为完整开发包。

| Gate | 通过条件 | 当前 |
|---|---|---|
| G1 独立标签 | ≥100 episodes、≥4 topology families、双标注 κ/仲裁记录 | 未通过 |
| G2 同信息 MAS baseline | 至少两个原生 MAS 近邻完成同输入复现，另列 proxy | 未通过 |
| G3 端到端成本 | API/抽取/图更新/BN/RSS/失败率分项 | 部分通过 |
| G4 外推 | 四拓扑 + 至少一种通信方式变化，family-disjoint | 部分通过 |
| G5 低误报预警 | 固定 FPR 下独立 lead-time 与及时召回 | 未通过 |
| G6 消融 | 双线、关系传播、模板、全图头的同协议消融 | 部分通过 |
| G7 复现发布 | 脱敏数据/生成器、配置、版本、失败记录和命令 | 基本通过 |
| G8 分区 | 学校采用版本的 CAS 大类/小类条目与查询凭据 | 未通过 |

## 5. 当前可写与不可写的结论

现在可以写：

- 我们提出一个 API-only、可审计的规范线 + 运行时关系线风险监测器；
- 在独立合成 MAS 轨迹的 family-disjoint OOF 上，双线融合优于匹配的 runtime/logistic/Bayesian 对照；
- 在两个 API 端点和公开外部轨迹上完成迁移诊断，并保留失败和不确定性边界。

现在不能写：

- “对真实任意 API Agent/MAS 有效”或“保证安全”；
- “超过 SentinelAgent/Node Contribution/AgentMonitor”等文献方法（未完成同信息复现）；
- “稳定提前预警”；
- “已确认中科院一区/二区投稿目标”；
- “白盒化了被保护的大模型”或“发现了真实因果图”。

## 6. 执行顺序

1. 冻结并执行 G1 独立确认集，优先于继续堆新模型；
2. 同一批数据完成 G2 原生 MAS baseline；
3. 接入语义抽取后做 G3 端到端成本；
4. 执行四拓扑/通信扰动和 G6 消融；
5. 最后按学校采用版本补 G8，并据结果选择 ESWA 或 TDSC；只有当方法确实是多源不完整证据融合时才考虑 Information Fusion。

## 证据入口

- 当前主结果：[paperAlpha/submission/FINAL_RESULTS.md](../../../../paperAlpha/submission/FINAL_RESULTS.md)
- 当前模型规格：[paperAlpha/results/independent_mas_journal_v1/MODEL_SPEC.md](../../../../paperAlpha/results/independent_mas_journal_v1/MODEL_SPEC.md)
- 独立确认协议：[paperAlpha/submission/EXTERNAL_CONFIRMATION_PLAN.md](../../../../paperAlpha/submission/EXTERNAL_CONFIRMATION_PLAN.md)
- 目标期刊与近邻核验：[target_comparison_20261005.md](target_comparison_20261005.md)

审计日期：2026-10-05。
