# Atomic microclaim retrieved-NLI v1

本实验只处理原始 QA fit634+cal159。确定性原子微主张在每个 passage 内独立 BM25 检索，冻结 ModernBERT-base-nli 对单句及 top-2 合并证据评分；未拆且文本完全一致的 pair 可复用冻结 v1 概率。

证据关系特征在微主张级汇总；fit 按 source group 五折 OOF 选读出与阈值，calibration 只报告严格迁移结果及明确标注的 cal-F1Opt 共同诊断。正式 baseline、标签、4-BPE 窗口与 official test 不变。扩展到 3680 条必须另建版本和冻结清单，不能混入本目录。
