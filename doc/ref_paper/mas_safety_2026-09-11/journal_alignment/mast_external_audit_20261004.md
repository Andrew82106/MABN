# MAST/MAD 外部数据审计（2026-10-04）

## 数据核验

- 来源：作者公开仓库与 Hugging Face 数据集 `mcemri/MAD`。
- 完整发布：1,642 条轨迹，来自 ChatDev、MetaGPT、Magentic、AG2、OpenManus、AppWorld、HyperAgent 七类 MAS，覆盖八类任务基准。
- 人工子集：19 条，三名标注者多数投票后 18 条为正；阳性率过高，不能承担主统计检验。
- 完整集标签：发布的 MAST annotation pipeline 结果，不是独立人工金标准。
- 本地文件：`paperAlpha/data/external/MAST/`。数据文件不写入仓库提交，避免把大型外部数据混入代码历史。

## 适配实验

`paperAlpha/scripts/evaluate_mast_external_v1.py` 只读取轨迹文本提取公开可见的角色、消息、工具、错误、验证、终止和重复等信号；不读取 `mast_annotation`、人工标注或标签字段。按 `mas_name::benchmark_name` 做五折分组留出，并使用内层 OOF 训练双线融合。

结果：双线 BN AUROC/AUPRC/F1/Brier = 0.680/0.851/0.858/0.178；匹配平面逻辑回归 AUROC/AUPRC = 0.640/0.831。按 MAS 系统拆分的 AUROC 为 0.069--0.931，说明迁移高度依赖系统和轨迹格式。该目标是“任意 MAST 失败”，与主实验的授权/污染/运行时风险不同，因此只能作为外部迁移证据，不能并入主表或宣称安全性能已被独立人工验证。

## 投稿边界

MAST 解决的是 MAS 失败分类与分类体系，不是 API Agent 的风险预警器。它适合证明我们的双线证据在另一种 MAS 轨迹形态上仍可运行，也暴露了外部标签语义不一致的问题；要关闭最终投稿门槛，仍需独立授权或人工标注、且目标与本文安全风险一致的 MAS 队列。

来源：[MAST 官方仓库](https://github.com/multi-agent-systems-failure-taxonomy/MAST)、[MAD 数据集说明](https://huggingface.co/datasets/mcemri/MAD)。
