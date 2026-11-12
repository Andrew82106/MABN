# 最终独立 CPU 复审

**结论：PASS（仅限 CPU 标签盲准备包）。** 我从 fit 的身份、问题、资料白名单与无标签 microclaim 独立重建了全部 3,776 行；它们与 `label_free_inputs.jsonl` 逐字段完全一致，覆盖 256 答、256 组，每答至少 2 个 claim。样本、prompt、机械 quote 和 claim 摘要均复现原锁。

- 清洁输入严格为 14 个字段，禁用字段及其内容未进入产物。
- `fit.jsonl` 的 JSON 行确实被完整解析，但代码只通过固定 allowlist 读取值；标签、回答正文、质量等字段没有被引用、复制或用于选择和特征。
- 原先 B1–B6 均已在最终协议或 CPU 产物层关闭。旧 2,423-claim cohort 已明确废止，当前唯一 cohort 是 3,776 claims。
- prompt 区域映射包含区域内标点；跨界 token 按字符重叠最大者归属，同量重叠直接失败，零长度 special offset 忽略。
- 白盒合同固定为：KV cache 只选生成 token；hidden/attention 必须在生成结束后以完整序列、`use_cache=False` 回放。
- 本次没有读取 calibration、official test 或 gold 值，没有加载模型、启用 GPU、训练或评分。

当前 runner 故意只提供 CPU skeleton，GPU 方法会直接报错。因此本结论不授权 GPU 或评分。真正的 GPU extractor 和 evaluator 落地后，仍须分别核对 replay、区域映射、唯一清洁输入、嵌套交叉验证和 P0 原始矩阵链。
