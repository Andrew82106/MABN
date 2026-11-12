# Token source attribution v4 scoring

先在无标签阶段为全部 claim×source-sentence 冻结 NLI，并把白盒 shard 转成每 token、每层 8 维。随后直接以原 4-BPE 窗口为监督行。

- fit：按 source group 五折 OOF；候选、模型和双阈值只看 fit OOF。
- 四个固定候选：full 513/full+NLL 514、band4 65/band4+NLL 66；band4 对每个 token 的连续 8 层先求均值。
- 全部候选固定 C=0.01。现有 Lookback/large OOF 的 C/epoch 曾由同一 cal159 选择，故只登记为污染诊断来源，禁止进入正式候选。
- calibration：冻结后只执行一次严格评测，不计算 cal-F1Opt。
- official test：无读取入口；正式 baseline 文件只校验哈希。
