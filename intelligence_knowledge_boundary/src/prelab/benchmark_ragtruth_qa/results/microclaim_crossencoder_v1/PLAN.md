# Microclaim full-evidence cross-encoder v1

主候选直接微调 ModernBERT-base-nli：输入为问题、三篇完整检索资料和一个原子微主张；输出该微主张的事实风险。五折按 source-connected group 严格隔离。训练损失在每个回答内部提高冻结 NLI 高风险但 gold 干净的微主张权重，并保持该回答的干净样本总权重不变，用来压低旧 NLI 的新增误报而不削弱正例总权重。

唯一控制是不训练同一 NLI 模型，直接用 1-P(entailment)。统一映射回 4-BPE 窗口；fit OOF 定阈值，calibration 只报告。正式 baseline 和 official test 不动。
