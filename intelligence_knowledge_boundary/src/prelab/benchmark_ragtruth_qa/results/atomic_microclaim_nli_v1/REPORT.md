# Atomic microclaim retrieved-NLI v1

只用原始 QA fit634+cal159。原子微主张按每个 passage 独立检索；冻结 ModernBERT-base-nli 评分；读出和阈值只由 source-group fit OOF 决定。calibration 只报告。

| 结果 | fit OOF窗口F1 | cal严格窗口F1 | fit OOF整答F1 | cal严格整答F1 | cal-F1Opt窗口 | cal-F1Opt整答 |
|---|---:|---:|---:|---:|---:|---:|
| fusion__hist_leaf7 | 0.607222 | 0.632379 | 0.758527 | 0.861386 | 0.642598 | 0.861386 |
| raw_NLI | 0.316248 | 0.361109 | 0.694414 | 0.769841 | 0.373020 | 0.779528 |

NLI pair 共 98854，其中精确复用旧冻结 pair 37785，新推理 61069。

正式 baseline 未修改；official test 未打开。cal-F1Opt 是反复使用开发集上的共同诊断，不是独立测试。扩展到 fit3680 必须另建冻结版本。
