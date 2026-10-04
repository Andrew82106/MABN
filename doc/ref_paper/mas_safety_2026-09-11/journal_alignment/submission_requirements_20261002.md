# 投稿实验要求固定表（2026-10-04复核）

本次复核补入了 LANYUN 新凭证的实际连通性与 80 案例迁移证据；中科院分区仍未用官方数据库条目完成认证。

本文件只记录出版社/期刊官方页面能直接支持的要求，不把中科院分区、JCR 或网页二次信息混为一谈。

## ESWA：当前首选匹配目标

官方期刊介绍把 risk assessment、multi-agent systems、knowledge management、monitoring 和 intelligent systems applications 列为范围，并强调论文要有原创的 expert/intelligent-system 方法与测试。页面还特别提醒：仅重命名已有概念、用隐喻包装算法、缺少标准优化和比较的工作不利于科学比较。

对本项目的直接约束：

1. 必须把“双线 + 白盒”写成明确的数学对象和信息边界：规范 DAG/知识线、运行时事件线、未知证据处理、风险计算和可审计解释。
2. 必须有匹配信息条件下的强基线，不能只和弱规则或单层模型比。
3. 必须报告完整训练/验证/测试划分、拓扑族隔离、置信区间、固定误报预算、消融和失败样本。
4. 必须把 API 生成、语义抽取、监测器推理和网络传输的成本分开；不能只报一个 CPU 推理时间。
5. 仅有合成轨迹不能作为最终外部效度证据；需要真实 API MAS 或独立授权的 MAS 轨迹，并公开可复现的生成/标注协议。

## Elsevier 通用 Original Research 要求

Elsevier 的 Your Paper Your Way 页面要求原始研究清楚呈现完整研究、意义、原创性和严谨性，并把理解结果所需的关键实验程序放入正文。由此，当前主结果必须以一套冻结候选为准，旧版本只能留作可追溯证据，不能挑选各版本最好分数拼表。

Elsevier 的研究数据政策要求按目标期刊指南决定数据共享方式，并鼓励提供代码、模型、算法、协议及其他可复现实验材料；因此本项目必须提供公开投影、冻结配置、失败/缺失记录和可重跑命令，不能只给最终分数。[Your Paper Your Way](https://www.elsevier.com/en-gb/subject/next/guide-for-authors)；[Elsevier Research data guidelines](https://www.elsevier.com/researcher/author/tools-and-resources/research-data/data-guidelines)

## TDSC：安全论证参照

IEEE 的 TDSC 投稿入口要求遵循其 Information and Submission Guidelines。TDSC 更适合作为安全威胁模型、攻击传播、系统级误报/漏报和可复现性标准的参照；它不是本项目当前已确认的投稿目标，分区仍需按学校采用的中科院版本单独核验。

IEEE Author Center 也明确鼓励共享数据、代码和其他研究产物以便验证实验和结论；本项目的 TDSC 对照因此必须保留同信息输入、原生任务边界、依赖版本和失败覆盖表。[IEEE Research Reproducibility](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/research-reproducibility/)

## 当前证据对照

| 要求 | 当前状态 | 证据 |
|---|---|---|
| 双线、可审计监测器 | 已完成开发候选 | `paperAlpha/results/independent_mas_journal_v1/MODEL_SPEC.md` |
| family-disjoint 强基线 | 已完成开发评测 | `paperAlpha/results/independent_mas_journal_v1/REPORT.md` |
| 固定误报预算与聚类区间 | 已完成开发评测 | `metrics.json` 中 `fixed_fpr_5pct` 与 `hierarchical_fusion_bootstrap_95` |
| 动作前检查 | 已有合成前缀检查 | `paperAlpha/results/independent_mas_prefix_v1/REPORT.md` |
| 真实 API MAS 外部效度 | 有迁移证据，但仍未完成独立确认 | A2ASecBench 仍是参考运行器标签；Lanyun `glm-5.3-flash` 已有 80 案例迁移（64 完成），但标签来自同一手工模拟器；独立人工标注/授权 MAS 队列仍缺 |
| 完整 API 成本/延迟 | 部分完成 | Lanyun 传输成本、监测器 CPU 延迟和 DAG 前端 smoke 已记录；语义抽取的端到端成本、内存和扩展性仍缺 |

## 官方来源

- [Expert Systems with Applications — Elsevier 官方介绍与范围](https://shop.elsevier.com/journals/expert-systems-with-applications/0957-4174)
- [Elsevier Your Paper Your Way](https://www.elsevier.com/subject/next/guide-for-authors)
- [Elsevier Research data guidelines](https://www.elsevier.com/researcher/author/tools-and-resources/research-data/data-guidelines)
- [IEEE TDSC Information for authors](https://ieeexplore.ieee.org/document/5954685)
- [IEEE TDSC 投稿入口说明](https://www.computer.org/digital-library/journals/tq/cfp-dependable-secure-computing)
- [IEEE Research Reproducibility](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/research-reproducibility/)
