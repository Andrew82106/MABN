# H-WD-BN：分层工作流约束动态风险监测器

## 1. 目标

在只能观察 API Agent 运行日志的条件下，识别“单个 Agent 局部信号不一定危险，但跨 Agent 组合后形成风险”的轨迹，并给出可审计证据。模型输出风险评估和预警，不自动阻断，也不声称完成因果发现。

## 2. 两张图的职责

### 规范工作流图 W

`W=(V,E_W)` 来自任务配置、角色定义、权限策略或人工审核后的 planner 结果。它描述允许的角色、能力和工作流边，不产生污点传播，也不被当作真实因果图。

W 只提供四类信息：

- 节点角色/能力模板；
- runtime 边是否 expected；
- 缺失前置步骤和规范外边偏离；
- 合规性先验和解释上下文。

### 运行时动态图 G

`G_t=(V,E_t)` 只由实际观测到的 `message_sent`、`delegation` 和父子任务事件构成。风险传播只能沿 `E_t` 进行；不存在的边不能由 W 补造。事件 `sequence/timestamp` 用于前缀预警和时间递推，不自动等价为因果方向。

## 3. 模型层次

### 局部模板层

每个节点提取可命名事实：

```text
untrusted_input
privileged_tool
same_agent_source_sink
permission_or_capability_mismatch
review_or_sanitization_state
```

模板参数用训练集估计的平滑 CPT 或 Beta-CPT 表示。episode 级标签不能复制成节点真值；若没有节点级回执，只能把节点分数称为弱监督局部证据，并通过 episode noisy-OR likelihood 训练。

### 动态传播层

每条运行时边使用来源、数据类别、置信度、expected/unexpected、delegation、时间间隔等特征：

\[
 g_e=\sigma(w^\top f_e),\qquad q_e=g_e\cdot carrier_e\cdot confidence_e.
\]

传播状态按事件发生顺序更新：

\[
 s_v(t^+)=1-(1-s_v(t))(1-s_u(t)q_e).
\]

unknown/redacted 只进入 upper/lower 观测不确定性，不默认为安全或危险。

### 全图策略/态势层

全图头可以使用 source/sink、路径距离、边缺失、unexpected 边比例、权限偏离、分支汇聚和观测覆盖率等统计量。但它必须作为独立分支或可解释融合分支，不能覆盖动态传播路径的解释。

最终输出至少包括：

```text
p_local, p_dynamic, p_workflow, p_final
node_scores, propagated_state
observed_paths, workflow_deviations
risk_lower, risk_point, risk_upper, abstain
first_alarm_sequence
```

## 4. 训练与公平性

1. 所有 CPT、门控、全图头和融合头只在训练组估计。
2. Fusion 使用 group-OOF 基础分数，不能用同一训练样本的 in-sample 基础输出拟合二级头。
3. 训练/测试按 scenario、topology 或 task family 分组；随机 episode split 只能做开发诊断。
4. 如果 `permission_mismatch` 是某个机制的强代理，必须同时报告：所有方法可用该字段的 observed setting，以及所有方法屏蔽该字段的 parity setting。
5. API-only 设定不能使用未知的规范 DAG；若 W 来自 planner，应单独标为 workflow-known，并报告 W 缺失/错误时的敏感性。

## 5. 必须报告的指标

- AUROC、AUPRC、F1、precision、recall；
- Brier、ECE 和 conformal selective risk；
- 首次报警位置/lead time；
- local-only 阴性而 full MAS 阳性的 cross-agent synergy recall；
- 按机制、路径深度、观测覆盖率、Agent 数分层；
- 规范图消融：runtime-only、W-only、W+runtime；
- 随机 group-CV 的均值、标准差和 cluster bootstrap 区间。

## 6. 当前证据边界

`independent_mas_v3` 已加入多种隐藏机制、运行时缺边和意外边。当前结果显示污点路径和协同机制较容易，而“无污点越权”和“规范外边”仍是盲区。因此下一版优先扩充授权/能力模板和偏离模板，再做时间递推与提前预警，而不是继续堆叠同一类污点路径特征。
