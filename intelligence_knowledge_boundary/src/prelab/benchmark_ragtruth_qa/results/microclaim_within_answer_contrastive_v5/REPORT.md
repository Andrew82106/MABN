# Within-answer contrastive ModernBERT v5

本阶段完成了 fit-only 配对、训练协议、成本核算和 CPU tiny；没有运行 GPU，也没有读取 calibration/test 标签。

## 配对结果

- 6,637 对；覆盖 3,375/3,474 个错误微主张（97.15%）。
- 覆盖 1,103/1,127 个含错误回答；未配对的是 24 个全错回答中的 99 个错误微主张。
- 选中 3,705 个不同安全微主张；每个可配对错误微主张取 1-2 个最相似安全负例。
- 同一原子父主张 388 对；只有 40 对不共享已选证据句。

## 训练接法

可直接加载每个对应的 expanded-v4 checkpoint 权重继续训练；v4 没保存优化器状态，所以 AdamW 状态会重新初始化。
模型和 risk logit 不变，不增加头。额外做一轮 pair endpoint pass：原 v4 endpoint BCE + 0.25 × RankNet。
OOF fold f 只用 held_fold != f 的 pair；full-fit 用全部 fit pair。最终仍按原 4-BPE 窗口和 fit 阈值评测。

## 成本与限制

六个模型额外 pair pass 约为原 v4 训练 token 的 41.5%。
同机 smoke 外推约 25.9 分钟，保守约 39.9 分钟；尚未实跑 GPU。
这里的‘困难’是同回答内的结构和表面相似，不是用 calibration 或模型分数挖掘。负例复用按 claim occurrence 校正，避免一个安全句反复出现而主导 BCE。
正式 baseline 未改，official test 未打开。
