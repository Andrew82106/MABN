# Evaluator 独立静态审查

**结论：BLOCK。** 被审代码为 `src/evaluate_forced_evidence_quote_probe_v1.py`，SHA256 `5f879d0714f7d0566659f24dabe2f0308db52fd1944d3e380bc624f65788549e`。真实评分应在下面两项修复并重新审查前保持关闭。

## 已正确实现

- `evaluate_real` 先完整校验并载入冻结、无标签的 P1–P3 特征，再调用唯一的 fit-gold/P0 入口；冻结缺失或哈希不符会在 gold 打开前失败。
- P0 是锁定 raw 1,024维窗口矩阵；P1/P2/P3 分别严格接收 21/549/549 维，且 P2 前21列必须逐值等于 P1。
- 每条件严格执行 5 个外折；每个外折内有 4 个 inner-valid 模型，每个 inner 模型只训另外 3 折；随后在 4 个 outer-train 折上重新拟合外层模型。
- scaler、每答总权重1、类质量和类别因子都在当前训练子集内重算。P0 的 C 固定 `1e-4`，P1–P3 固定 `1e-3`，liblinear/L2/seed/max_iter 均固定。
- 阈值候选是 `nextafter(max,+inf)` 加所有 distinct inner-OOF score；判正使用 `>=`，按 `(F1, precision, threshold)` 取最大。
- claim 风险以 max 映射至重叠的 4-BPE 窗；未被 claim 覆盖的 eligible window 固定为0；answer 对全部 eligible window 取 max，并复用本外折窗口阈值。
- pooled AP 使用 raw outer-OOF 概率；pooled F1 使用各外折阈值产生的二值结果。P0 文件只读，AST 中没有 baseline 写入，也没有 calibration/test 数据路径。

## 两个阻断项

1. **真实入口没有锁 sklearn 版本。** 协议固定 scikit-learn 1.6.1，但版本检查只在可跳过的 `synthetic_selftest` 第899行。用户可以直接运行 `evaluate`，此时不会检查版本；最终 `EVALUATION.json` 也不记录 sklearn 版本和 evaluator hash。应在打开 gold 前强制版本完全等于 1.6.1，并把版本与 evaluator SHA 写入真实结果。
2. **claim 坐标没有绑定回原回答文本。** 第407–417行直接用 `claim_start/claim_end` 生成 claim 标签，但没有检查边界，也没有断言 `original_response[start:end] == claim_text_raw`。若坐标发生偏移，程序可能给错误文本分配标签和窗口且仍继续。应在派生任何标签前逐 claim 做边界与逐字相等检查，并核对 clean row 与锁定 answer/token/window 的 response/source/group 身份。

非阻断建议：当前 feature manifest 锁住数组、plan 和无标签输入，但不含 extractor 代码/runtime 或有序列名摘要。最终 seal 应把独立 GPU extraction 审查和对应 runner hash连入结果链。

本次只读取 evaluator 源码、协议及文件哈希，并用 Python AST 检查调用/写入结构；没有导入或执行 evaluator，没有打开真实 feature arrays、fit gold、calibration 或 test，没有运行评分或 GPU，也没有修改 evaluator/baseline。
