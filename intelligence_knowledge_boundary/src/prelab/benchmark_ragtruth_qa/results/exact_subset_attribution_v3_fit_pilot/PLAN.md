# Exact subset attribution v3：256-group fit-only pilot

先在无标签 prepared inputs 中按固定 SHA-256 规则选 256 个不同 fit group，每组一答。CPU 阶段不看长度、分数或标签。审核后，以同一 Llama 和同一 8 个资料子集建立独立缓存。

冻结缓存后，`score` 只打开三份 expanded-fit 金标；按固定五折 GroupKFold 报统一 4-BPE 窗口和整答指标。A/B/C 是我们方法内部消融，不是论文 baseline。
