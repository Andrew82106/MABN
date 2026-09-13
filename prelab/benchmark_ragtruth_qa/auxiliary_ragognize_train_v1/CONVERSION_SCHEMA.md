# RAGognize 统一训练格式

每个 cohort 均包含以下文件，所有文件只来自官方 train 的 Llama-2 回答。

## `answers.jsonl`

每个回答一行。核心字段：

- 身份：`response_id`、`train_row_index`、`official_user_prompt_index`、`source_id`、`group_id`。
- 输入：`question`、`retrieved_passages`、`released_prompt`、`original_response`。
- 原标签：`released_hallucinations`，完整保留官方 `{start,end,text,valid}` 列表；`answer_risk` 是是否存在有效 span。
- 审计元数据：`answerable`、hint 设置、prompt 中是否实际出现明确拒答指令、原记录/文本/标签哈希、token/window 数量。

`golden_answer`、其他生成器输出和注释解释不进入转换后的检测器数据。

## `sequences.jsonl`

每个回答一行，保存完整 Llama tokenizer 输入和回答轴：

- `full_input_ids`、`attention_mask`、`answer_token_positions`；
- `response_token_ids`；
- `response_token_offsets` 为截到原回答 `[0,len(response))` 的字符区间；
- `response_token_offsets_raw` 保留跨越 prompt/answer 分隔空格的负起点；
- `lexical_mask`、`risk_mask`、`answer_risk`。

这里没有模型前向结果，也不是作者历史生成轨迹。

## `tokens.jsonl`

每个回答 BPE 一行：`answer_token_index`、`full_token_position`、`token_id`、截断/原始字符区间、`lexical`、`risk`。它是 `sequences.jsonl` 回答轴的展开形式。

## `windows.jsonl`

每个滑动窗口一行：`window_index`、回答 token 的 `[token_start,token_end)`、四个 `token_ids`、字符包络、`lexical_token_count`、`risk_token_count`、`eligible_lexical`、`risk`。窗口固定宽 4、步长 1；本数据没有少于 4 BPE 的回答。

## `groups.jsonl`

每个材料连通组一行，列出该 cohort 内的 `response_ids`、官方问题 ID 和 source IDs。后续交叉验证或内部留出必须整组移动。

## 标签含义

官方字符 span 是自动标注。未标记位置在训练时只能视为发布数据中的银标负例，不能解释为人工核实正确。token/window 标签是字符 span 的确定性派生值，不是新增标注。
