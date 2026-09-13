# Exact subset attribution v3 fit-only pilot：CPU 审计

通过。固定 SHA-256 规则从 615 个 fit group 选择 256 组，每组一答；选择代码不读取标签、长度、风险类型或模型分数。五折也在无标签阶段冻结。

GPU 将复用 v2 WDDM 门禁，但 smoke、256 条 cache 和 lineage 全部写 v3。评分只允许读取 expanded-fit 的 answer/token/4-BPE-window 三份金标。A/B/C 是我们方法内部消融；论文 baseline 未运行、未修改。当前未使用 GPU 或标签。
