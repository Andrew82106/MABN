# Token source attribution v4 scoring

先在无标签阶段为全部 claim×source-sentence 冻结 NLI，并把白盒 shard 转成每 token、每层 8 维。随后直接以原 4-BPE 窗口为监督行。

- fit：按 source group 五折 OOF；候选、模型和双阈值只看 fit OOF。
- 融合：NLL 与 clean OOF Lookback/large 只在窗口评分层加入。
- calibration：冻结后只执行一次严格评测，不计算 cal-F1Opt。
- official test：无读取入口；正式 baseline 文件只校验哈希。
