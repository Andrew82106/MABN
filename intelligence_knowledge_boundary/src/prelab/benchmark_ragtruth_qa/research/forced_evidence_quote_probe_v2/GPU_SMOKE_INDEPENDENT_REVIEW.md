# V2 GPU Smoke 独立复审

**结论：PASS；允许 full extraction。** 正式 `GPU_SMOKE.json` SHA-256 为 `4179ad72833440d9c5a68c7e2e0c7797b8c67ff5dd9785b9b58144c0570a04be`，绑定 runner SHA-256 `eb4a51353ba4edaff2cff877f576ff206f1fca1d6bd820928fb0db4acebfcca2` 与 numerical protocol SHA-256 `ad0a2198ba69e75c78852eab103e8a829b6f9fb21389f322d5879f016e989df1`。

## 硬门结果

- 固定索引顺序精确为 `[2826, 994, 1533, 1695, 1299, 1143, 3352, 3752]`；首条为 `tolerance_anchor`，其余 7 条为 `validation_smoke`，prompt 长度为 `[255, 354, 385, 435, 508, 568, 634, 799]`。
- 8 条均满足 teacher-forcing target IDs 与 cached IDs 相同、全部值有限、`failed_drift_gates=[]`。总 argmax mismatch 为 0，高置信 conflict 为 0。
- 全局最大 float64 绝对 drift：selected log probability `0.05869102478027344 <= 0.0625`；entropy `0.04814901947975159 <= 0.0625`；signed margin `0.5 <= 0.5`。索引 1299 的 margin 恰好命中包含边界并通过。
- 最短 2826 与最长 3752 的 repeat：generation IDs 精确相同，prompt-ID hash/positions/offsets/stop/parse 精确相同；cached statistics 与 replay hidden/attention 的最大 repeat 差值均为 `0.0 <= 1e-6`。
- `formal_claim_records_written=0`，正式 claim record 目录不存在；未读取 gold/cal/test，未运行 scoring。

## 事故与原子性

保留的失败收据 `GPU_RUNNER_FAILURE_gpu-smoke_1789222299176746400.json` SHA-256 为 `54c464a0b9c2905454ebc9ab780ae72578fc18fc209ea68018fce2fd7cb89102`。它记录了错误解释器在 `check_cpu_ready()` 内因缺少 `bitsandbytes` 元数据而退出：CUDA 未初始化、0/8 claim 执行、无模型/特征/标签/评分。失败收据使用独立文件名；当前无 claim cache、无 pending 文件，且正式 smoke 只有一个 `GPU_SMOKE.json`。runner 对正式 smoke 使用拒绝覆盖、flush+fsync、`.pending` 原子替换。

本次仅只读解析 label-free smoke、源码与哈希锁；未修改 runner、protocol、smoke、V1 或 baseline，未读取 fit gold/calibration/official test，未再次启用 GPU。

此 PASS 只允许 full extraction；正式评分仍须等待 full extraction、独立 finalizer review 和 feature freeze 完成。
