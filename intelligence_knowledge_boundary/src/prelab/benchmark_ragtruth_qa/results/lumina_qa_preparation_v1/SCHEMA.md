# 冻结输入接口

读取 `preparation_complete.json` 中 `artifacts_sha256` 的所有正式文件哈希，核验 `CPU_OUTPUT_CHECK.json` 完成。`source_sha256` 绑定原计划、tokenizer适配源码和R7公式源码。`feature_inputs.jsonl` 顺序为原生fit634＋新增fit3046＋原cal159，`response_order_sha256` 使用 `feature_qa.digest(response_id列表)`。

每行字段：

- `response_id/source_id/group_id/partition/official_split`：原开发身份；`old_plan_sha256` 是原完整无标签plan的规范JSON摘要。
- `original_prefix_ids/random_prefix_ids`：各自首次答案token之前的完整ID前缀；`answer_token_ids` 是两侧完全相同的原答案BPE。
- `original_input_ids/random_input_ids`：分别严格等于该侧 prefix＋answer；无答案后尾部token。两侧input摘要为同名前缀的 `_sha256` 字段。
- `original_answer_positions/random_answer_positions`：各自完整输入中原答案位置；`original_predictor_positions/random_predictor_positions` 均为前者减1。最后一个预测位置是完整长度−2，用于预测保留的最后一个答案token。
- `response_token_offsets/response_token_offsets_raw`：原回答字符半开区间；前者裁至回答内，后者允许首区间起点−1。重复/重叠offset不去重。
- `original_prompt/random_prompt/original_response` 及其摘要：可见文本；没有donor的问题或答案。`original_reference_range/random_reference_range` 是各自未加聊天wrapper的prompt字符范围。
- `donor_source_id/donor_group_id/material_sha256/donor_material_sha256`：同源多生成器共享选择；`donor_assignments.jsonl` 每目标材料1条，全部 donor 只来自fit池。
- `position_shift`：随机前缀token数减原前缀token数；`suffix_tokens_after_answer=0`；`labels_used=false`，`exact_original_generation_trace=false`。

`source_materials.jsonl` 只用于审查可见资料和前后模板，包含已开放fit/cal材料。真正的donor池仍严格限其中634份fit材料；不得将cal材料当donor。重放必须使用完整已冻IDs，不能重新分词答案、改加空格、换chat模板、截断或省略末token。

`protocol_attempt01.json`、`*_attempt02.*` 与 `PREPARATION_FAILURE_*.json` 只保留历史失败证据，不是正式输入。提取程序只读取complete绑定的正式文件。
