# 独立 MAS 确认集审计与冻结方案

日期：2026-10-05  
状态：**未满足独立确认门槛；仅冻结方案，不把现有结果冒充人工金标准**

## 结论

现有数据不能直接组成论文所需的独立确认集。原因不是样本数量不够，而是三个条件同时不成立：

1. 现有数据的标签大多来自原始 benchmark、执行器或本文自己的 oracle，不是与监测器输入完全隔离的双人安全裁决。
2. 现有数据没有一个同时覆盖四种真实工作流拓扑、并在两个独立 API/model 条件下运行的冻结队列。
3. 可以把不同来源的轨迹合并成更大的开发集，但不能把它们拼接后称为一个独立测试集；来源、标签语义、拓扑和可观测字段都不同。

## 现有来源逐项审计

| 来源 | 可见规模 | 拓扑/条件覆盖 | 标签来源 | 独立确认结论 |
|---|---:|---|---|---|
| A2ASecBench API | 80（4 个攻击族，40/40 attack-control） | 4 个威胁族，不等于 chain/fork/join/review 四种拓扑；主要是本地 API，按族使用 gpt-5.5 或 gpt-5.6-luna | benchmark 的 scenario role 与执行器字段 | **外部审计，不是双人安全金标准** |
| MAST/MAD full | 1,642 条轨迹、11 个 MAS×benchmark 组 | 日志没有统一的四拓扑字段 | MAST 事后失败注释；human-labelled 子集只有 19 条 | **开发/稳健性审计，不能满足规模与 API 条件** |
| AgentLeak | 4,258 条当前可用轨迹、937 request groups | 数据有多模型/多框架，但没有可核验的统一工作流 DAG 四拓扑 | benchmark 泄露判定 | **开发迁移集；不能与其他来源直接合并成独立测试集** |
| HarnessAudit | 初始 3 条 + expansion 20 条（其中 16 完成、4 失败） | 明确的串行阶段 DAG；没有四种统一拓扑，也只有本地 gpt-5.5 条件 | 原作者 actionwise checker/oracle；paired forecast 为离线回放 | **结构与观测审计，不是独立双标确认集** |
| 本地/LANYUN authority-payment API | 本地 65/80、LANYUN 64/80；另有 8 条四拓扑 smoke | 两端点，但使用同一手工双 agent 模拟器；四拓扑 smoke 只有 8 个 episode | 预定义 simulator/evaluator policy | **跨端点迁移诊断，不是独立人工标签** |

MAST 的 19 条 human-labelled 轨迹虽然含有三个 annotator 字段，但规模远低于门槛，且不是本项目预注册的 API 确认队列；不能用它替代新确认集。A2ASecBench 的四个攻击族也不能当成四个工作流拓扑。

## 最小可执行冻结方案

在任何模型、提示词、特征、模板或阈值再次根据确认集结果修改前，先生成并哈希一个不可变队列。建议采用 **160 个目标完成 episode**：四种真实 runner 拓扑（chain、fork、join、review）× 两个 API/model 条件 × 每格 20 条（10 benign、10 violating）。失败、拒答、超时和截断均保留在尝试分母；不得按结果替换样本。若因失败达不到 160 个完整 episode，直接报告短缺，不把成功样本补齐成“完成集”。

每条 episode 需要保存：

- 工作流定义、角色/工具权限和对象绑定；
- 监测器可见的完整前缀、动作序号、消息/工具参数及返回结果；
- 独立的执行状态或 sandbox 证据；
- API/model 条件、请求/响应 ID、失败记录和成本；
- 预注册的 policy/risk contract 与 topology ID。

两名标注者分别只看 monitor 可见材料和独立执行证据，不看 monitor 分数、隐藏 attack ID 或对方标签。他们分别记录：违规二值标签、首次违规事件、违规类别、证据 event ID 和简短理由。确认前报告 raw agreement 与 Cohen's κ；争议由第三人或预注册裁决规则处理。

评测拆成三部分：

1. 按 topology family 与 API/model condition 分组划分 fit/calibration/confirmation；确认集只在最后一次模型冻结后打开。
2. 主指标固定为 AUROC、AUPRC、group-weighted Brier、固定 FPR 下召回、首个告警 lead time、良性误报率、弃权覆盖率。
3. 另报语义抽取延迟、端到端 API token/费用、监测器 p50/p95 延迟、内存和失败率；不要把离线 BN 推理耗时写成端到端成本。

## 可执行顺序

1. 先实现四拓扑统一 runner，并为每种拓扑冻结 benign/violating policy contract。
2. 预注册 160 个 episode ID 与两个 API/model 条件，生成队列哈希。
3. 运行两条件 API 队列；所有失败原样保留。
4. 生成去标签、去隐藏字段的 A/B 标注包；标注者独立完成并计算 κ。
5. 固定模型、阈值和 baseline 后，才打开确认标签并运行一次最终评测。

在第 5 步完成前，主结果仍应写作“development package / external diagnostic”，不能写成“independent human-confirmed superiority”。审计脚本为 `scripts/audit_independent_confirmation_sources_v1.py`，默认只读现有文件并输出忽略的 JSON/Markdown 报告。
