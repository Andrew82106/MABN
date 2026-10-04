# MAS 风险监测器：当前完整实验结果

本文件是当前唯一的结果摘要入口。它对应 `submission/protocol.json` 和
`results/submission/final_eval/` 下的冻结本地产物；后者因数据体积被 `.gitignore`
忽略，重新运行脚本即可复现。

## 方法

模型是两条可检查的风险证据线：

1. **规范/工作流 BN**：角色、能力、权限、允许的工作流边和危险汇点。
2. **运行时 BN**：实际消息、工具调用、权限不匹配、异常边、冲突、验证和终止事件。

两条线先分别计算风险，再用训练折内 OOF 分数做融合。整个评测不读取 API 模型权重、
隐藏状态或标签字段；这里的“白盒”指监测器自身的变量、关系、概率因子和证据路径可检查，
不是把被保护的大模型变成白盒。

## 主结果

数据为 `independent_mas_v3`：4,000 条公开 MAS 轨迹、1,345 条正例、186 个拓扑族。
采用按拓扑族隔离的五折 `StratifiedGroupKFold`；阈值只在训练折内确定。

| 方法 | F1 | AUROC | AUPRC | Brier |
|---|---:|---:|---:|---:|
| runtime logistic（同一运行时特征） | 0.653 | 0.805 | 0.653 | 0.167 |
| graph-features logistic | 0.578 | 0.724 | 0.527 | 0.194 |
| local-only logistic | 0.564 | 0.688 | 0.485 | 0.204 |
| reliability-weighted Bayesian fusion（期刊邻近对照） | 0.604 | 0.763 | 0.596 | 0.459 |
| **two-line BN fusion（本文方法）** | **0.668** | **0.820** | **0.675** | **0.160** |

相对于直接的 reliability-weighted Bayesian fusion，对同一批 OOF 预测做 2,000 次拓扑族
成对 bootstrap：

- ΔAUROC = **0.056**，95% CI **[0.045, 0.067]**；
- ΔAUPRC = **0.079**，95% CI **[0.061, 0.098]**；
- 两项差值在 **2,000/2,000** 次重采样中为正。

去掉 `permission_mismatch` 后仍为 AUROC 0.811、AUPRC 0.661，说明结果不只依赖一个字段。

## 外部与 API 迁移

- **LANYUN API 迁移**：80 个预设案例中 64 个完成；two-line fusion AUROC/AUPRC =
  **0.854/0.824**。这是迁移证据，不替代独立人工确认集。
- **MAST/MAD 外部审计**：1,642 条轨迹，按 MAS×benchmark 分组；AUROC/AUPRC =
  **0.681/0.852**。该标签是公开失败注释流水线，不是与本文安全目标一致的人类金标准；不同 MAS 的 AUROC 为 0.034--0.931，因此不宣称普适领先。
- **A2ASecBench**：只保留不读取 outcome/violation 字段的角色审计，AUROC/AUPRC =
  **0.788/0.835**；其标签仍是 benchmark 角色定义，不是人工安全裁决。

## 期刊对照边界

主要邻近方法是 ESWA 的多智能体 Bayesian fusion 与安全 guardrail 工作；其官方出版页见
[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0957417426011541)。
MAS 威胁与协作拓扑补充参考 IEEE TDSC 的 *Cracks in Collaboration*，作者代码见
[GitHub](https://github.com/S1mpleyang/Cracks-in-Agent-Collaboration)。MAST 数据集与失败分类见
[官方仓库](https://github.com/multi-agent-systems-failure-taxonomy/MAST)。

这些是方法和实验规范的邻近参照，不等于已经复现了对方任务，也不等于当前年份中科院分区
已经完成认证；`protocol.json` 明确要求提交前补做学校采用版本的 CAS 官方核验。

## 当前结论

当前已经有一套可复现、可解释、对强基线有成对统计优势的 MAS 风险监测实验包；但按投稿标准，
它仍不能诚实地写成“最终已投”结果。提交前还缺三项硬证据：独立人工/许可的 MAS 安全确认集、
完整在线语义抽取与内存/延迟测量、以及固定低误报点下的独立早期预警确认。
