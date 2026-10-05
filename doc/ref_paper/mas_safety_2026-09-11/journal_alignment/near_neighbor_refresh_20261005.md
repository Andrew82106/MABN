# MAS 安全监测近邻刷新审计（2026-10-05）

目的：只纳入可由 DOI 与出版社正式页面核验的近邻论文，补充现有 `target_comparison_20261005.md`。分区不是由出版社页面给出；除非能取得学校采用版本的中科院分区条目，本文一律标为“未核验”。“已发表”按出版社卷期/文章号页面判断；仅有 DOI、Early Access 或未来卷期的条目标为 online-first/状态不确定。

## A. 四个重点方向的增量对照

| 方向 | 论文与 DOI（官方页） | 出版状态 | 可核验事实 | 对本文的直接启示 | 中科院分区 |
|---|---|---|---|---|---|
| MAS 安全/鲁棒监测 | [Enhancing robustness of LLM-driven multi-agent systems through randomized smoothing](https://doi.org/10.1016/j.cja.2025.103779)（Chinese Journal of Aeronautics） | 出版社页给出 vol.39 issue 7, July 2026, article 103779；已发表 | 在黑盒 MAS 共识中用 randomized smoothing 与自适应采样给出概率鲁棒性保证，目标是阻断恶意行为/幻觉传播；不是风险分类器，也不使用 API-only 运行日志 | 可作为“黑盒鲁棒性/传播抑制”参照；主实验应增加攻击传播、干预前后风险和计算开销，而不能把其 consensus accuracy 当监测 AUROC | 未核验（CJA 期刊页不提供中科院条目） |
| MAS 安全监测/日志图 | [Beyond the prompt: Log-based threat detection and attribution for multi-Agent LLMs](https://doi.org/10.1016/j.ipm.2026.104768)（Information Processing & Management） | 出版社摘要页给出 vol.63 issue 6, article 104768, 2026-09；已发表 | ALTEDA 只看同步应用/网络/主机日志，不看 prompt/response；800 traces、31.4% attack success；图级检测、SHAP 归因、约 40% 轨迹进度的早报 | 是 API-only 文本边界的“更丰富观测上界”；必须做同信息投影，或单列 richer-observation，不得与本文直接公平混比 | 未核验 |
| Bayesian/不确定性融合 | [A multi-agent LLM framework with Bayesian fusion and safety guardrails for ATC-pilot communication error detection](https://doi.org/10.1016/j.eswa.2026.132241)（Expert Systems with Applications） | 出版社/DOI 元数据给出 vol.321, article 132241, 2026-07；已发表 | 四角色（syntax/semantics/context/risk）+ 显式规则 + reliability-weighted Bayesian fusion；894 条专家标注真实交换（52 条错误）+400 压力样本；报告严格留出、5-fold OOF、召回/FPR、约 1.6 s 延迟 | 应复用低阳性率、固定 FPR、OOF 校准和端到端成本报告；ATC 错误标签不能当作本文 MAS 授权/信息流标签 | 未核验 |
| 白盒/可解释监测 | [Beyond the prompt: Log-based threat detection and attribution for multi-Agent LLMs](https://doi.org/10.1016/j.ipm.2026.104768)；方法补充参照 [ShapG: New feature importance method based on the Shapley value](https://doi.org/10.1016/j.engappai.2025.110409) | ALTEDA 已发表（见上）；ShapG 出版社页给出 vol.148, 15 May 2025, article 110409；已发表 | ALTEDA 用 SHAP + aggregator-aware redistribution 将系统告警归因到 agent/interaction；ShapG 是图结构约束的模型无关 Shapley 特征重要性方法，不是 MAS 安全数据集 | “解释”须落到风险节点、关系边、证据 ID，并做删除/替换证据的反事实稳定性；不能只展示自然语言 rationale | 未核验 |
| API-only Agent 运行时安全 | [Securing LLM agents: From prompt sanitization to autonomous red teaming and beyond](https://doi.org/10.1016/j.iotcps.2026.03.001)（Internet of Things and Cyber-Physical Systems） | 出版社页显示 vol.5 (2025), pp.185–209；文章 DOI 2026；已发表记录，但卷年与 DOI 年份跨年 | 综述将 prompt/decoding/runtime/backdoor、多 Agent 防护与 red teaming 分层，明确指出运行时监测和标准化 benchmark 的缺口；非实验 baseline | 可用作 API-only 运行时威胁覆盖清单：prompt injection、tool misuse、memory/跨 Agent 传播、运行时监测；不能引用其综述观点为性能数值 | 未核验 |
| API/系统运行时安全 | [Securing large language models: A quantitative assurance framework approach](https://doi.org/10.1016/j.jisa.2025.104351)（Journal of Information Security and Applications） | 出版社页给出 vol.97, March 2026, article 104351；已发表 | 把输入/输出验证、访问控制、第三方服务等要求映射为可测安全分数，并按 OWASP LLM 风险测试；不是 MAS 专用监测 | 可借鉴“安全控制覆盖率 + 残余风险 + 部署开销”报告，不把单一分类分数当作安全保证 | 未核验 |

## B. 状态与分区使用规则

1. `10.1016/j.cja.2025.103779`、`10.1016/j.ipm.2026.104768`、`10.1016/j.eswa.2026.132241`、`10.1016/j.engappai.2025.110409`、`10.1016/j.iotcps.2026.03.001`、`10.1016/j.jisa.2025.104351` 均能在 Elsevier/ScienceDirect 官方页面看到文章号或卷期；本审计将其记为“已发表记录”。
2. DOI 前缀年份不等于正式卷期年份：IoTCPS 文章页面显示 vol.5 (2025)，DOI 含 2026；不能据 DOI 年份推断卷期。
3. 本轮没有取得可审计的学校采用版中科院分区条目。因此 ESWA、IPM、Information Fusion、TDSC、CJA、JISA 等都只能写“投稿候选/分区未核验”，不能在论文中写成“中科院一区/二区已确认”。
4. ALTEDA 使用主机/网络日志；本文若只允许 API 可见消息、工具调用与结果，须把 ALTEDA 标为 richer-observation 上界，或重新抽取同信息特征后比较。

## C. 给本文实验的最小新增核对项

- 传播安全：按 chain/fork/join/review 分层报告攻击传播率、首次违规事件、固定 FPR 早报提前量和干预成本。
- Bayesian/不确定性：报告节点级与系统级 Brier/ECE、OOF 约束、低阳性率下固定 FPR 召回；补充无融合头、无关系边和 runtime-only 消融。
- 白盒解释：输出 agent/edge/evidence ID；加入删除或替换关键证据后的风险变化与归因稳定性。
- API-only 边界：明确 monitor 可见字段（工作流、消息、工具名/参数/结果、时间戳）；隐藏状态、违规标签和 evaluator 字段只给 evaluator。
- 状态声明：近邻论文分开写“已发表”“online-first/未来卷期”“arXiv 非 SCI”；不要用 Crossref 创建日期或二级分区网页替代官方出版/分区证据。

## 官方来源入口

- [CJA 官方文章页](https://www.sciencedirect.com/science/article/pii/S1000936125003851)
- [IPM 官方文章页](https://www.sciencedirect.com/science/article/abs/pii/S0306457326001597)
- [ESWA 官方 DOI/文章页](https://doi.org/10.1016/j.eswa.2026.132241)
- [Engineering Applications of AI 官方文章页](https://www.sciencedirect.com/science/article/pii/S0952197625004099)
- [IoTCPS 官方文章页](https://www.sciencedirect.com/science/article/pii/S2667345226000015)
- [JISA 官方文章页](https://www.sciencedirect.com/science/article/pii/S2214212625003874)
- [中科院分区官方入口](https://www.fenqubiao.com/)
