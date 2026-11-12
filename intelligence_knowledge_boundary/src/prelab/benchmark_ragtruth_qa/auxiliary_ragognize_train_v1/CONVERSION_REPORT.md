# RAGognize train-only 统一转换结果

## 结论

RAGognize 的官方 **train** 分片已完整转换成当前实验可直接读取的回答级、词元级和 4-BPE 窗口级结构。转换及独立复查均通过；未下载或读取 test/validation，未运行模型、GPU、训练或基线。

建议把 `no_unanswerability_hint` 作为最贴近本课题的主训练候选，因为它排除了模板中明确设置“无法回答提示”的样本；`all_train` 用于样本量消融。两者是包含关系，不能互相充当独立测试集。

## 冻结来源

- 数据集：`F4biian/RAGognize`
- 官方修订：`aab54518c2a7c0d25fff8bffbf5337d0321de142`
- 许可：CC-BY-SA-4.0
- 本地仅有的内容分片：`train-00000-of-00001.parquet`
- train SHA-256：`6ad56f84a06863b4dea28e86014cc16e37a64fefe8965c53902b370f51839a44`
- 转换设计 SHA-256：`bc16e304d76b04fcc1184485d6e231c9019f21c3ac01ed1bedf790301d5c71c7`
- intake MANIFEST SHA-256：`abd676f3bf62213df8a00d2a71177b89245b491b70ca571c70bf85685e86f73b`
- conversion MANIFEST SHA-256：`91bdbb7adee2461a8d04e13ebfbecf5539874c3685405e7ea85c54f162828ca1`
- conversion AUDIT SHA-256：`09b433d3d99f80356218558d9db550930b88cb044d4eb7669350a210f1a4e4a0`

## 转换规模

| 子集 | 回答 | 连通组 | 幻觉回答 | BPE | 可评词元 | 风险词元 | 可评 4-BPE 窗口 | 风险窗口 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all train | 1,842 | 917 | 1,007（54.67%） | 203,286 | 174,928 | 45,664（26.10%） | 197,452 | 53,841（27.27%） |
| no hint | 971 | 491 | 542（55.82%） | 102,778 | 88,133 | 23,571（26.74%） | 99,649 | 27,816（27.91%） |

`no hint` 完全按 `template_details.settings.unanswerability_hint == false` 预先冻结，没有看标签或分数。971 条实际提示词中也未发现明确允许或要求回答“资料不足”的文字。

## 输出

每个子集均包含：

- `answers.jsonl`：问题、检索资料、原提示词、原回答和官方字符 span；
- `sequences.jsonl`：完整输入 ID、回答词元位置、字符偏移和风险掩码；
- `tokens.jsonl`：逐 BPE 标签；
- `windows.jsonl`：宽度 4、步长 1 的窗口标签；
- `groups.jsonl`：按问题和来源连通去重的分组；
- `manifest.json`：数量、tokenizer 和文件哈希。

总清单为 `conversion_v1/MANIFEST.json`，独立复查为 `conversion_v1/AUDIT.json`。复查确认原问题、资料、回答和 span 没有变化，词元与窗口索引连续，所有样本均属于且仅属于一个冻结组。

## 使用限制

1. RAGognize 的 span 是自动标注，不是人工金标；未标位置只能视为银标负例。适合补训练，不能单独支撑最终效果结论。
2. 原回答由 Llama-2-7B-chat 生成，但数据没有保存当时的激活和 log probability。后续要使用白盒信号，需用同一检查点按冻结文本做因果重放提取，并单独说明这是重放轨迹。
3. 官方 `token_starts` 与本地 tokenizer 有 29 条边界差异；转换统一从字符 span 映射到本地 BPE，没有改官方 span。另有 1 条回答含重叠 span，原 span 原样保留，二值标签只取并集。
4. 训练/验证切分必须以 `group_id` 为单位，不能按回答随机切分。最终外部测试仍需使用与此训练集不重叠的人工标注数据。
