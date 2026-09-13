# 《情报杂志》项目参考论文

关联项目：[Intelligence Knowledge Boundary](../../../intelligence_knowledge_boundary/README.md)。

当前研究已转向弱监督事实错误监测，直接使用的 HaMI（NeurIPS 2025）和 RAGTruth（ACL 2024）论文、源码与结果见 [prelab](../../../prelab/README.md)。以下五篇保留为早期知识边界方向资料，表中的“优先复现”仅指当时方案。

以下为此前讨论中选定的五篇正式会议论文，保存自会议或论文集公开 PDF。文件来源、下载时间、大小和 SHA-256 见 [manifest.json](./manifest.json)。文献用途为初步阅读判断，不代表已复现结果或创新性查重完成。

| 顺序 | 论文与本地文件 | 发表信息 | 本项目用途 |
|---|---|---|---|
| 1 | [Estimating Knowledge in Large Language Models Without Generating a Single Token](./01_EMNLP2024_KEEN.pdf) | EMNLP 2024 主会 | KEEN：训练实体级知识探针，估计主体的总体知识覆盖 |
| 2 | [Query-Level Uncertainty in Large Language Models](./02_ICLR2026_Query_Level_Uncertainty.pdf) | ICLR 2026 | Internal Confidence：免训练的问题级内部置信度；优先复现 |
| 3 | [LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations](./03_ICLR2025_LLMs_Know_More_Than_They_Show.pdf) | ICLR 2025 | 回答可靠性探针、探测位置和跨数据集失效边界 |
| 4 | [Knowledge Boundary of Large Language Models: A Survey](./04_ACL2025_Knowledge_Boundary_Survey.pdf) | ACL 2025 主会 | 统一知识边界定义、检索相关研究 |
| 5 | [Trust Within? Seek Beyond? Knowledge Boundary Aware Policy Optimization for Agentic Search](./05_ACL2026_Knowledge_Boundary_Aware_Agentic_Search.pdf) | ACL 2026 主会 | 检索决策的相关工作；不将其等同于白盒探针方法 |

## 论文页面与官方代码

1. KEEN：[论文](https://aclanthology.org/2024.emnlp-main.232/)；[代码](https://github.com/dhgottesman/keen_estimating_knowledge_in_llms)。仓库已有脚本，但 README 仍写着 “Code is coming soon!”；完整性和可运行性待检查，不能视为开箱即用。
2. Internal Confidence：[论文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3a07c3a67cfe50d3236b71fb674c7f30-Abstract-Conference.html)；[代码](https://github.com/tigerchen52/query_level_uncertainty)。公开了运行示例与复现实验入口，当前未在本地运行。
3. LLMsKnow：[论文](https://proceedings.iclr.cc/paper_files/paper/2025/file/a712d461e57201efe35d429a6f1731c1-Paper-Conference.pdf)；[代码](https://github.com/technion-cs-nlp/LLMsKnow)。用于补充比较；生成后探针与生成前方法应分开评估计算成本和可用信息。
4. 综述：[论文](https://aclanthology.org/2025.acl-long.256/)。
5. KbPO：[论文](https://aclanthology.org/2026.acl-long.1276/)。

## 阅读时需核对的边界

- KEEN 的实体整体评分不能直接当作具体事件事实的已知/未知标签。
- Internal Confidence 是内部信号估计方法，并非额外训练的知识探针。
- 提供检索材料后的回答能力不能直接解释为参数中原有的知识。
- 模型事实知识、事实回答正确性、分析推断能力与未来预测可靠性分别处理。
- 方法能否迁移到中文事件、冷门事件和新近进展，需要本项目实测。

本目录只存文献和文献索引。后续源码、数据、权重和实验结果均存入关联项目目录。
