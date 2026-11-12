# Exact subset attribution v3 fit-only pilot：独立 CPU 审计

通过。独立脚本未导入生产 runner，重新从 615 个 fit group 计算固定哈希选择与五折，精确得到 256 个不同 group/source 的回答。46,481 个答案 BPE、904,496 个八视图输入 token 均重算一致。

CPU 入口未触达模型、GPU 或 fit 金标；GPU/cache/score 输出均不存在。A/B/C 是内部消融，论文 baseline 未运行或修改。
