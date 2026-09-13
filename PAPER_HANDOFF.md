# 论文协作交接说明

本仓库当前准备投稿《情报杂志》的方向是：在“问题＋检索资料＋大模型回答”的开源情报场景中，监测回答里的事实性错误。目标是先给出风险分数，再定位回答中需要人工复核的局部片段。

## 目前已经完成

- 固定了 RAGTruth QA 的资料组划分、人工事实错误 span、4-BPE 步长 1 窗口、整答 `max` 聚合和阈值规则。
- 在 3,680 条 fit 回答（615 个资料组）上训练，在 159 条 calibration 回答（154 个资料组）上做开发评测；官方旧 test 已封存退出，新的独立 holdout 协议已经写好但尚未运行。
- 候选检测器把三类白盒信号（注意力回看、生成概率/NLL、隐藏状态风险）与资料—问题—回答语义信号做固定融合，再以窗口最大风险得到整答风险。
- 已按同一考卷复现并评测 Lookback Lens、GHOST、LUMINA、ReDeEP、RAGognizer 等已完成的正式基线；基线的模型结构、特征和原生读出没有改动，只做无参数的坐标映射。

## 当前开发集结果

下表是 calibration 开发结果，F1 和 AUROC 均为越高越好。窗口是局部定位，整答是“这条回答是否含事实错误”。

| 方法 | 窗口 F1 | 整答 F1 | 窗口 AUROC | 整答 AUROC |
|---|---:|---:|---:|---:|
| 本项目候选（semantic + white-box + large） | **0.6903** | **0.8911** | **0.9134** | **0.9061** |
| Lookback Lens | 0.6009 | 0.8455 | 0.8724 | 0.8488 |
| RAGognizer | 0.5075 | 0.8069 | 0.8249 | 0.7069 |
| LUMINA | 0.3313 | 0.7850 | 0.6914 | 0.7022 |
| ReDeEP | 0.3296 | 0.7724 | 0.6693 | 0.5907 |
| GHOST | 0.3204 | 0.7722 | 0.6327 | 0.5890 |

这些数字支持“在当前开发口径下可行、点估计领先”的判断，不能直接写成独立测试或 SOTA 结论：候选和阈值仍在 calibration 上选择，配对 bootstrap 区间跨 0，最终 holdout 尚未打开。整答全报风险的简单基线 F1 约为 0.7722，因此论文需要同时报告 AUROC、AUPRC、混淆矩阵和局部窗口结果。

## 方法和论文切口

建议把贡献表述为“将白盒幻觉探针迁移到开源情报的证据约束场景，并做局部事实错误监测适配”。核心问题不是让模型重新生成，而是判断它是在使用给定资料，还是在资料没有覆盖时凭自身知识补写事实。窗口级结果回答“哪里值得复查”，整答级结果回答“这条回答是否需要复查”。

详细的数据口径、信号定义、基线边界、失败实验和写作建议见 [prelab/benchmark_ragtruth_qa/research/collaborator_paper_brief_v1/EXPERIMENT_OVERVIEW_CN.md](prelab/benchmark_ragtruth_qa/research/collaborator_paper_brief_v1/EXPERIMENT_OVERVIEW_CN.md)。固定协议和结果入口如下：

- [CURRENT_STATUS.md](prelab/benchmark_ragtruth_qa/CURRENT_STATUS.md)
- [BASELINE_PROTOCOL.md](prelab/benchmark_ragtruth_qa/BASELINE_PROTOCOL.md)
- [FORMAL_BASELINE_RESULTS.md](prelab/benchmark_ragtruth_qa/FORMAL_BASELINE_RESULTS.md)
- [独立基线审计](prelab/benchmark_ragtruth_qa/research/final_baseline_audit_v4/REPORT.md)
- [独立 holdout 协议](prelab/benchmark_ragtruth_qa/research/replacement_holdout_v1/BLIND_HOLDOUT_PROTOCOL.md)

## 协作者下一步

1. 先阅读交接说明和固定协议，确认研究问题、标签和指标。
2. 用新的资料组 holdout 运行一次完全冻结的候选与基线，报告独立结果。
3. 在同一 holdout 上报告窗口/片段定位、整答风险、校准曲线和不同错误类型的召回。
4. 论文中明确区分开发集结果、独立测试结果和历史失败尝试，不把训练内或人工 oracle 上限当作模型成绩。

## 仓库发布边界

Git 中保留源代码、协议、报告、论文索引和小型 manifest。模型权重、虚拟环境、缓存、原始/中间 JSONL、特征矩阵、checkpoint 和日志不提交；它们体积很大且可由 manifest 重新下载或生成。`paperAlpha/.env` 等凭据文件始终排除。这样合作者可以复核方法和数字，也不会把本地密钥或数百 GB 的运行产物带入公开仓库。
