# Evaluator 独立静态复审 V2

**结论：PASS。** 被审代码为 `src/evaluate_forced_evidence_quote_probe_v1.py`，SHA256 `d2f6fb4c3a863c6c84273448a6a6c5aec6202eeb8e2fb37e0e5240ea017a11cb`。旧版 BLOCK 文件保持原样，作为旧代码 SHA 的审计记录；本结论只适用于上述新版 SHA。

## 原阻断项复审

1. **B1 已解决。** `assert_runtime_contract`（第85–89行）要求 `sklearn.__version__` 严格等于 `1.6.1`。AST 确认 `evaluate_real` 的第一条可执行语句就是该检查（第826行）；随后才校验无标签冻结特征（第828行），再打开 fit gold/P0（第829行）。真实结果还会记录 evaluator SHA 和 sklearn 版本（第854–855行）。
2. **B2 已解决。** 对每个 claim，程序在派生 token 所属关系和标签之前，检查 `0 <= start < end <= len(original_response)`，并逐字验证 `original_response[start:end] == claim_text_raw`（第413–418行）。clean 与 answer（第402–408行）、token 与 answer（第409–412行）、window 与 token（第461–463行）均严格核对 `response_id/source_id/group_id`。

## 回归结论

V1 已通过的 P0–P3 特征合同、严格 5 外折 × 4 内折、折内 scaler/回答权重/类别权重、固定 C、阈值候选与 `>=`、claim→4-BPE 窗口 max、未覆盖窗口为0、answer max、pooled AP/F1，以及 baseline 只读约束，在本次修订中未被破坏。

因此，独立 evaluator 审查不再阻止真实评分。真实运行仍须使用上述代码 SHA，并由运行时冻结特征与哈希门禁全部通过。

本次只做源码、文件哈希和 AST 静态复审；未导入或执行 evaluator，未打开真实 feature arrays、fit gold、calibration 或 test，未运行评分或 GPU，也未修改 evaluator、baseline 或旧版审查文件。新 self-test receipt 仅核对 SHA256：`ba8d25592e57fcc16f5f9e3aaf8c8f6128cd4ffc36a4109d3ebfd804e31ad279`。
