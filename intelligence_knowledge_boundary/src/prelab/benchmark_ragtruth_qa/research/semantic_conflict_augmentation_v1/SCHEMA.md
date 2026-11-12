# Pair schema

- `pair_id`: 稳定哈希 ID。
- `question`, `response_id`, `source_id`, `group_id`: 原 fit 身份。
- `passage_id`, `sentence_id`, `source_sentence`: 正确支持句及坐标。
- `nli_premise`: 冻结 NLI 使用的 premise；attribution 类型同时含正确 passage 与目标 passage 的对照句。
- `supported_claim`, `corrupted_claim`: 最小对两端。
- `corruption_type`: `entity|number|negation|temporal|attribution`。
- `edit`: 支持端字符范围、原值和替换值；替换端范围为 `[start,start+len(replacement))`。删除时替换端为空，应以边界邻域汇总。
- `labels`: 明确标记 `silver_not_human_gold`。
- `checks`: 机械一致性检查。
- `strict_rule_eligible`, `strict_exclusion_reasons`: 自动严格规则结果。
- `split_group_id`, `split_component_groups`: owner、entity donor、共享支持句形成的泄漏隔离组件。

