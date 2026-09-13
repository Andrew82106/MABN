# V2 Feature Finalizer 独立静态审查

**结论：PASS。** 被审 finalizer SHA-256 为 `25ac94cd67d61d09ee3440d8854d378f310be730d27509eeb51b24c22a5cfff9`；CPU synthetic selftest SHA-256 为 `8792c60bd4fd201810684c940bf9d3759c163adc71d0434bdd2974c214f5da15`。

## 核对结论

- V1 的 fail-closed、拒绝覆盖、临时目录整包原子发布、freeze 最后写入、完整哈希链、clean-row 顺序、逐记录 commit/metadata/NPZ 绑定、P1/P2/P3 顺序与 `21/549/549` 维、float32/finite、P1 等于 P2 前 21 列、递归 taint 拒绝均保持不变或加严。
- V2 的差异限于独立 V2 路径，以及将 runner、numerical protocol、GPU extraction completion、full-batch cache/replay QA 纳入 lineage、manifest 与 freeze。逐记录 record-only QA 会从保存数组重新计算并核对。
- 正式路径要求并绑定 CPU preparation review、runner CPU selfcheck、runner review、GPU smoke、GPU smoke review、evaluator selftest/review、本 finalizer selftest/review、full extraction receipt、runtime signature、3,776 条 record 顺序与完整文件集合；收集后再次验证静态与执行收据，防止中途漂移。
- selftest 使用 5 条 synthetic record 覆盖哈希、行序、orphan、nonfinite、P1/P2 prefix、原子拒绝覆盖与 V2 lineage；另以 3,776 条 label-free 行通过真实 evaluator frozen-feature loader contract。收据记录 gold/cal/test/scoring 均未运行，GPU/CUDA/model 均未启用。

`formal_finalize_allowed=true` 仅表示运行时所有门齐全且仍匹配时允许执行 finalize。审查时尚无 `GPU_EXTRACTION_COMPLETE.json`、frozen feature bundle 或评分结果，因此当前正式 finalize/scoring 仍关闭。

本次仅做源码 diff、AST、静态编译、哈希与既有 CPU receipt 解析；未运行 finalize/verify，未读取 fit gold/calibration/official test，未启用 GPU/CUDA，也未修改 finalizer、evaluator、runner、protocol、smoke、V1 或 baseline。
