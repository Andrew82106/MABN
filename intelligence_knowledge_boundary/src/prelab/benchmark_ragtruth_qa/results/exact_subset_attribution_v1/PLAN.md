# Exact subset attribution v1（我们的方法候选）

对每条已发布答案固定构造 8 个 Passage 子集，只重放同一答案的 selected-token log probability。由 8 个值精确计算三个来源的 Shapley 值、删源影响、交互和引用对应影响。

prepare/check/audit 只做无标签 CPU 工作。审核后才可单独运行 gpu-smoke、extract；score 最后才读 fit/cal 标签，按 source group 做五折 OOF，并映射到统一 4-BPE 窗口和整答最大值。正式 baseline 不作任何改动。
