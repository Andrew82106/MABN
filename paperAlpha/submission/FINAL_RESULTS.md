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

主模型的族聚类 95% CI 为 AUROC **[0.804, 0.835]**、AUPRC **[0.646, 0.707]**、
Brier **[0.154, 0.167]**；同一 bootstrap 下 runtime logistic 的 AUROC/AUPRC 为
**[0.791, 0.818]/[0.622, 0.683]**。

相对于直接的 reliability-weighted Bayesian fusion，对同一批 OOF 预测做 2,000 次拓扑族
成对 bootstrap：

- ΔAUROC = **0.056**，95% CI **[0.045, 0.067]**；
- ΔAUPRC = **0.079**，95% CI **[0.061, 0.098]**；
- 两项差值在 **2,000/2,000** 次重采样中为正。

去掉 `permission_mismatch` 后仍为 AUROC 0.811、AUPRC 0.661，说明结果不只依赖一个字段。

## 外部与 API 迁移

- **API 迁移（两种生成条件）**：同一预注册 authority-payment 队列各运行 80 案例。
  LANYUN `glm-5.3-flash` 完成 64/80，实际违规标签下 two-line fusion
  AUROC/AUPRC = **0.854/0.824**；本地 `gpt-5.6-luna` 完成 65/80，指标为
  **0.750/0.746**。LANYUN 队列记录 416 次请求/459,902 tokens，本地队列记录
  406 次请求/310,540 tokens。两者都是受控模拟器的跨端点迁移证据，不替代独立人工确认集，
  也不支持“对任意 API 模型均有效”的结论。
- **API 前执行策略目标**：以预先声明的 scope/revocation policy 为标签、保留失败样本时，
  透明知识规则在 LANYUN 上 AUROC/AUPRC = **0.950/0.950**，本地端为
  **0.913/0.913**；这是策略绑定诊断，不是通用安全准确率。
- **四拓扑 API 集成 smoke**：新增 8 个低成本 LANYUN episode，覆盖 chain/fork/join/review，
  共 22 次请求（21 成功、1 失败）；它验证统一调度、失败保留和 monitor/evaluator 分离，
  但任务文本显式给出授权/范围情境，因此不作为风险准确率或独立安全标签结果。
- **结构升级审计**：将 Layer-1 的局部 BN 风险作为节点状态沿运行时边传播，在 600 条开发轨迹
  的固定诊断划分上 AUROC/AUPRC 为 **0.772/0.618**，低于现有并行双线融合的 **0.775/0.635**；
  因此保留现有模型为主模型，把层间传播作为可解释结构审计，不用测试结果反向改主模型。
- **MAST/MAD 外部审计**：1,642 条轨迹，按 MAS×benchmark 分组；AUROC/AUPRC =
  **0.681/0.852**。但 flat logistic 为 0.640/0.831，max-line rule 为 0.676/0.858，
  所以这里是外部稳健性补充，不宣称 BN 在该数据上胜出。该标签是公开失败注释流水线，
  不是与本文安全目标一致的人类金标准；不同 MAS 的 AUROC 为 0.034--0.931。
- **A2ASecBench**：只保留不读取 outcome/violation 字段的角色审计，AUROC/AUPRC =
  **0.788/0.835**；其标签仍是 benchmark 角色定义，不是人工安全裁决。
- **AgentLeak**：4,258 条可用轨迹、937 个 request group；独立留组的 late fusion
  AUROC/AUPRC = **0.561/0.353**。它是重要的失败诊断：说明当前模型对真实泄露流的迁移仍弱，
  不能把主数据集上的优势外推成普适优势。

## 运行时审计

- 前缀预警在固定开发点的负例误报率为 **0.200**；正例及时召回率为 **0.145**（以全部正例为分母，
  在“存在可观测动作”的正例子集上为 **0.155**）。因此只能称为可提前的诊断信号，不能称为稳定的在线早期预警。
- 不确定性机制弃权 **6.1%**、覆盖率 **93.9%**，但覆盖样本错误率仍为 **0.350**，不是安全保证。
- 监测器本身的 p95 计算开销为 **0.55--1.96 ms**（按事件数增长），不含 API、语义抽取和训练；
  峰值 tracemalloc 约 **0.010--0.073 MB**，一次批量测量的进程 RSS 峰值增量不超过 **0.031 MB**
  （仅 Python 监测路径，不能解释为端到端内存上限）。API 端到端传输成本和语义抽取成本另测；
  进程级部署 RSS 和训练成本仍未纳入。
- **在线语义抽取成本（独立 smoke）**：从四拓扑 API 集成 smoke 的 21 条公开完成回复中，
  用低成本 `glm-5.3-flash` 做事件结构化；21/21 请求完成，JSON 严格解析 8/21，解析失败率
  **61.9%**；端到端抽取延迟 p50/p95 为 **6.53/11.76 s**，输入/输出 token 总量为
  **4,234/8,742**。这只是成本与可观测性测试，不是语义准确率、风险指标或独立安全验证；
  失败记录保留在 `results/submission/development/semantic_extraction_cost_20261005_lanyun_r1/`。

## 期刊对照边界

主要邻近方法是 ESWA 的多智能体 Bayesian fusion 与安全 guardrail 工作；其官方出版页见
[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0957417426011541)。
MAS 威胁与协作拓扑补充参考 IEEE TDSC 的 *Cracks in Collaboration*，作者代码见
[GitHub](https://github.com/S1mpleyang/Cracks-in-Agent-Collaboration)。MAST 数据集与失败分类见
[官方仓库](https://github.com/multi-agent-systems-failure-taxonomy/MAST)。

这些是方法和实验规范的邻近参照，不等于已经复现了对方任务，也不等于当前年份中科院分区
已经完成认证；`protocol.json` 明确要求提交前补做学校采用版本的 CAS 官方核验。
逐项的期刊/论文对照表见 `doc/ref_paper/mas_safety_2026-09-11/journal_alignment/target_comparison_20261005.md`。

## 当前结论

当前已经有一套可复现、可解释、对强基线有成对统计优势，并完成两种 API 生成条件迁移的
MAS 风险监测实验包；但按投稿标准，它仍不能诚实地写成“最终已投”结果。提交前还缺两项
硬证据：独立人工/许可的 MAS 安全确认集，以及端到端部署级内存/训练成本与固定低误报点下的
独立早期预警确认。

## 再现入口

使用同一数据、5 折和随机种子的完整重跑已在 `results/submission/development/journal_recheck_20261005/`
完成；双线融合 AUROC/AUPRC/Brier/F1 = **0.819671/0.674792/0.160353/0.667761**，
与本文件主结果一致。这是复现验收证据，不是新的测试集结果。

在仓库根目录执行以下命令可重建主公开轨迹和主评测；所有生成结果默认留在被忽略的
`paperAlpha/results/` 下，不会污染 Git：

```powershell
python paperAlpha/scripts/generate_independent_mas_benchmark_v3.py --n 4000 --seed 20260917 --out paperAlpha/results/independent_mas_v3
python paperAlpha/scripts/evaluate_independent_mas_journal_v1.py --input paperAlpha/results/independent_mas_v3/traces_public.jsonl --labels paperAlpha/results/independent_mas_v3/labels.jsonl --out paperAlpha/results/independent_mas_journal_v1 --folds 5 --seed 20261002
python paperAlpha/scripts/evaluate_reliability_bayes_baseline_v1.py --input paperAlpha/results/independent_mas_v3/traces_public.jsonl --labels paperAlpha/results/independent_mas_v3/labels.jsonl --out paperAlpha/results/submission/final_eval/reliability_bayes_baseline_v1
python paperAlpha/scripts/compare_primary_vs_reliability_v1.py
```

主模型、基线和结果摘要的协议入口是 `submission/protocol.json`；独立确认集的标注要求见
`submission/EXTERNAL_CONFIRMATION_PLAN.md`。
