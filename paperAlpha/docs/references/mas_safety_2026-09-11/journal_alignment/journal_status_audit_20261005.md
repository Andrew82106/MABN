# 期刊近邻状态与分区证据审计（截至 2026-10-05）

本文件只审计论文的正式期刊身份、DOI、发表状态和分区证据；不修改代码、数据或主实验结果。检索优先采用出版社页面、Crossref 元数据和作者/DBLP记录。`CAS-未认证`表示没有取得中科院分区库登录后的具体条目，不能在论文中写成“中科院一区/二区已确认”。

## 结论

- 当前最稳妥的投稿对照期刊是 **Expert Systems with Applications (ESWA)**、**Information Fusion**、**IEEE Transactions on Dependable and Secure Computing (TDSC)** 和 **Information Processing & Management (IPM)**。它们的论文页面和 DOI 可以核验，但本次没有获得中科院官方分区库的登录后记录。
- ESWA、Information Fusion、TDSC、IPM 的 2025 分区只能写成“机构/二级来源报告的候选”，不能写成官方认证。`https://www.fenqubiao.com/` 本次返回 502，且需要登录后查询。
- 正式期刊论文可以作为“实验设计参照”；只有任务、输入信息、标签和指标一致，并且完成同信息量复现，才能进入数值 baseline 表。多数近邻论文只能做参照，不能直接混比 F1/AUROC。
- `HalluProp`、`Bayesian Uncertainty Propagation` 等 arXiv 工作不是 SCI 期刊论文；只能作为前沿方法参照，不能满足“SCI 一区/二区对照”的要求。

## 逐篇状态核验

| 论文 | 正式期刊/DOI状态 | 截至 2026-10-05 的发表判断 | 可否用于投稿对照 | CAS分区结论 |
|---|---|---|---|---|
| Sadak, *A multi-agent LLM framework with Bayesian fusion and safety guardrails for ATC-pilot communication error detection* | ESWA；[DOI 10.1016/j.eswa.2026.132241](https://doi.org/10.1016/j.eswa.2026.132241)；[出版社页](https://www.sciencedirect.com/science/article/pii/S0957417426011541) | Vol. 321，2026；出版社页已给完整摘要、卷和 article number，属于正式期刊文章 | **可以作为首要实验范式参照**；其 ATC 标签不能直接当作本文 MAS 安全 baseline | ESWA 的 2025 中科院计算机大类一区/Top 仅有机构/二级证据；**CAS-未认证** |
| Chahine, *Separating intent from execution* | ESWA；[DOI 10.1016/j.eswa.2026.133781](https://doi.org/10.1016/j.eswa.2026.133781)；[出版社页](https://www.sciencedirect.com/science/article/pii/S0957417426026898) | Crossref 标 Vol. 332、2027-01；出版社文章页可访问，属于已建立的出版记录，但截至本日期不能写成“2026 已刊卷期” | **可以作 MAS 安全架构/消融参照**；不直接比较通用风险监测 F1 | ESWA **CAS-未认证** |
| Yang et al., *Cracks in Collaboration* | IEEE TDSC；[DOI 10.1109/TDSC.2026.3670889](https://doi.org/10.1109/TDSC.2026.3670889)；[IEEE页](https://ieeexplore.ieee.org/document/11424974/)，[DBLP记录](https://dblp.org/rec/journals/tdsc/YangZLXZ26.html) | Vol. 23, Issue 3, May–June 2026, pp. 7191–7207；正式期刊卷期已核验 | **可以作拓扑/通信/攻击传播实验参照**；攻击成功率不能与本文风险检测率直接混比 | 2025 中科院大类二区来自机构/二级页；IEEE 只证明 SCIE，**CAS-未认证** |
| Rabieinejad et al., *Beyond the prompt (ALTEDA)* | IPM；[DOI 10.1016/j.ipm.2026.104768](https://doi.org/10.1016/j.ipm.2026.104768)；[出版社页](https://www.sciencedirect.com/science/article/abs/pii/S0306457326001597)，[DBLP记录](https://dblp.org/rec/journals/ipm/RabieinejadZD26.html) | 正式期刊文章；出版社摘要给 800 traces、31.4% ASR、日志图/部分轨迹早报/归因。卷期存在冲突：Crossref/J-GLOBAL 为 63(6)，DBLP 为 63(7)；正文只写 DOI、article 104768，待出版社卷期最终显示 | **最接近的日志图监测参照**；其输入含应用/网络/主机日志，本文 API-only 只能做输入投影或 richer-observation 上界 | 2025 中科院一区/Top 仅有机构/二级证据；**CAS-未认证** |
| Wang et al., *CogTrust* | ESWA；[DOI 10.1016/j.eswa.2026.131535](https://doi.org/10.1016/j.eswa.2026.131535)；[出版社页](https://www.sciencedirect.com/science/article/pii/S0957417426004483) | Vol. 313，1 June 2026；正式期刊文章 | **可以作 agent-level trust/reputation 对照方向**，但需同信息量实现或取得代码 | ESWA **CAS-未认证** |
| Ben Hassouna et al., *LLM-Agent-UMF* | Information Fusion；[DOI 10.1016/j.inffus.2025.103865](https://doi.org/10.1016/j.inffus.2025.103865)；[出版社页](https://www.sciencedirect.com/science/article/pii/S1566253525009273) | Vol. 127 Part C，March 2026；正式期刊文章 | **只能作工作流/组件建模参照**，不是风险检测 baseline | Information Fusion 2025 一区/Top 有机构引用分区表的证据；**CAS-未认证** |
| Wang et al., *PTFusion* | Information Fusion；[DOI 10.1016/j.inffus.2025.103731](https://doi.org/10.1016/j.inffus.2025.103731)；[出版社页](https://www.sciencedirect.com/science/article/pii/S1566253525007936) | Vol. 127 Part A，March 2026；正式期刊文章 | **只能作知识图、MCP工具编排和版本化知识参照**；其任务完成率不进入本文检测表 | Information Fusion **CAS-未认证** |
| Tang et al., *Security of LLM-based Agents* | Information Fusion；[DOI 10.1016/j.inffus.2025.103941](https://doi.org/10.1016/j.inffus.2025.103941)；[出版社页](https://www.sciencedirect.com/science/article/abs/pii/S1566253525010036) | Vol. 127 Part C，March 2026；正式综述文章 | **只能作威胁分类/覆盖矩阵来源**，综述不提供可运行检测器 | Information Fusion **CAS-未认证** |
| Li et al., *PenExpert* | ESWA；[DOI 10.1016/j.eswa.2026.133284](https://doi.org/10.1016/j.eswa.2026.133284)；[出版社页](https://www.sciencedirect.com/science/article/pii/S0957417426021937) | 出版社页标 15 Dec 2026、Vol. 331 Part B；按本审计日期仍是未来卷期，不能写成“已发表的 2026 对照论文” | **当前不进入正式已发表对照表**；可记录为已公开出版记录/未来卷期的预备参照 | ESWA **CAS-未认证** |
| Raafay, *Beyond Patterns* | Computers & Security；[DOI 10.1016/j.cose.2026.105092](https://doi.org/10.1016/j.cose.2026.105092)；[出版社页](https://www.sciencedirect.com/science/article/pii/S0167404826002683) | 31 July 2026，In Press / Journal Pre-proof；不是已排定卷期版本 | **可作 BN 行为意图方法参照**，但本项目当前投稿范围已因出版社 AI/ML 收稿声明排除该刊 | Computers & Security 当前收稿范围冲突；不作为投稿目标，亦不据第三方分区页声称 CAS |
| Cheimonidis & Rantos, *A novel proactive and dynamic cyber risk assessment methodology* | Computers & Security；[DOI 10.1016/j.cose.2025.104439](https://doi.org/10.1016/j.cose.2025.104439)；[出版社页](https://www.sciencedirect.com/science/article/pii/S0167404825001282) | Vol. 154, July 2025, article 104439；正式期刊文章 | **可作动态 BN 风险建模参照**，不能作 MAS 检测 baseline；期刊当前投稿范围不匹配 | Computers & Security 不作为投稿目标，CAS 不认证 |
| Badhon et al., *IRAF-BRB* | ESWA；[DOI 10.1016/j.eswa.2025.127979](https://doi.org/10.1016/j.eswa.2025.127979)；[出版社页](https://www.sciencedirect.com/science/article/pii/S0957417425015982) | Vol. 285, 1 Aug 2025, article 127979；正式期刊文章 | **可作可解释风险评估实验报告参照**，不是 MAS 运行时 baseline | ESWA **CAS-未认证** |

## 期刊分区证据等级

| 期刊 | 可核验的非 CAS 事实 | 现有 CAS 线索 | 本次最终表述 |
|---|---|---|---|
| ESWA | Elsevier 官方范围和论文页可核验；SCIE 当前条目未用登录后 MJL 重新认证 | 2025 机构/二级页面报告计算机科学大类一区、Top | `CAS候选（机构/二级证据）`，不能写 `已确认一区` |
| Information Fusion | Elsevier 官方论文页可核验；期刊范围与多源信息融合匹配 | 2025 机构页面称一区、Top | `CAS候选（机构证据）`，不能写 `已确认一区` |
| IEEE TDSC | IEEE 2025 title list/出版社记录可证明 SCIE 和正式论文身份 | 2025 二级/机构页报告计算机科学大类二区 | `CAS候选（机构/二级证据）`，不能写 `已确认二区` |
| IPM | Elsevier 官方论文页和 Crossref/DBLP可核验正式论文；当前分区未用官方库认证 | 2025 机构页面称计算机科学大类一区 Top，但不同机构学科口径有差异 | `CAS候选（机构证据）`，投稿前必须按学校口径复核 |

中科院分区官方入口：[fenqubiao.com](https://www.fenqubiao.com/)。本轮无法取得登录后期刊条目，因此没有任何“官方 CAS Q1/Q2”结论。JCR、SJR、CCF、SCIE 收录或机构宣传页都不能替代该条目。

## 对本文投稿实验的使用规则

1. **正式对照论文**：Sadak 用于低阳性率、Bayesian fusion、多后端、固定 FPR 和延迟报告；Yang 用于拓扑/通信/攻击传播分层；ALTEDA 用于日志图、部分轨迹早报、归因和资源成本；CogTrust 用于同信息量 trust/reputation 控制。
2. **架构/知识参照**：Chahine、LLM-Agent-UMF、PTFusion、PenExpert（未来卷期）只用于规范线、工作流节点、知识图和状态管理设计，不把其任务完成率混入风险检测指标。
3. **方法参照**：Computers & Security 两篇和 IRAF-BRB 可支撑 BN/可解释风险建模叙述，但不能声称解决 API MAS 运行时安全监测，也不属于当前投稿目标。
4. **统一比较要求**：只有相同可观测输入、相同 episode/family 划分、独立标签和相同预算下的实现才可进入主 baseline 表。其他论文数值仅在 related-work 表中列出，不能声称“我们的模型超过该论文”。

## 主要来源

- [中科院分区官方入口](https://www.fenqubiao.com/)
- [Clarivate Master Journal List](https://mjl.clarivate.com/)
- [ESWA 官方范围](https://shop.elsevier.com/journals/expert-systems-with-applications/0957-4174)
- [Information Fusion 官方范围](https://shop.elsevier.com/journals/information-fusion/1566-2535)
- [IPM 出版社论文页](https://www.sciencedirect.com/science/article/abs/pii/S0306457326001597)
- [IEEE TDSC 出版社条目](https://ieeexplore.ieee.org/document/11424974/)
- [Crossref API](https://api.crossref.org/)

