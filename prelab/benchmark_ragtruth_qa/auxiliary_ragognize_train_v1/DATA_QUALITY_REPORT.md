# RAGognize train-only 数据质量报告

## 结论

**可以接入训练准备，但本轮没有训练。** 官方仓库把 train 与 test 放在两个独立 Parquet 中，因此只下载了 train。Llama-2-7B-chat 的 1,842 条回答都具备问题、检索资料、回答和字符 span；全部 1,483 个有效 span 坐标合法且能映射到本项目的 4-BPE 窗口。

需保留三项限制：标签是自动标注，不是人工金标；871 条 prompt 的模板元数据开启了“资料不足时拒答”提示；官方 `token_starts` 含哨兵并有 29 条与本地 tokenizer 的字符起点不完全一致。

## 下载边界与版本冻结

- 官方数据集：[F4biian/RAGognize](https://huggingface.co/datasets/F4biian/RAGognize)
- 固定 revision：`aab54518c2a7c0d25fff8bffbf5337d0321de142`
- 许可：`CC-BY-SA-4.0`
- 官方树中只有一个 train shard 和一个 test shard，物理分离。
- 已下载：官方数据卡、HF API 元数据、HF 文件树元数据、`data/train-00000-of-00001.parquet`。
- 未下载、未打开：test、任何 validation 内容、`RAGognize-with-samples-test`。
- train 文件：30,129,198 bytes；SHA-256 `6ad56f84a06863b4dea28e86014cc16e37a64fefe8965c53902b370f51839a44`，与官方 LFS OID 完全一致。

## Llama-2-7B-chat 子集

| 项目 | 数量 |
|---|---:|
| train 问题-资料行 / Llama 回答 | 1,842 / 1,842 |
| answerable / unanswerable | 924 / 918 |
| 有至少一个有效幻觉 span 的回答 | 1,007 |
| 无 span 的回答 | 835 |
| 有效字符 span | 1,483 |
| 问题、资料、回答或 prompt 缺失 | 0 |
| span 越界或 `answer[start:end] != text` | 0 |
| 外层与内部最终 span 列表不一致 | 0 |

共 5,322 段文档；5,098 段原文直接出现在 `documents_str`，另 224 段去掉首尾空白后完全出现，没有真正缺失。1 条回答有两个已发布 span 重叠；不改标签，二元词元/窗口标签取两者并集。

数据卡说明这些回答由 `Llama-2-7b-chat-hf` 在 temperature 0.0 下自然生成，幻觉不是后期注入。它同时提供完整生成 prompt 和回答，但没有历史 logits、token IDs 或激活，因此以后模型前向仍应表述为固定检查点上的可复现重放，不能声称拿到了原始运行轨迹。

## “资料不足”提示审计

| 模板设置 | 行数 |
|---|---:|
| `unanswerability_hint=false` | **971** |
| `unanswerability_hint=true`，`in_system=false` | 668 |
| `unanswerability_hint=true`，`in_system=true` | 203 |

在实际 Llama `full_prompt` 中，418 条明确出现“不能回答、不能提供答案、承认资料局限”等拒答指令；971 条关闭提示的 prompt 均未出现这类文字。另有 453 条元数据标为开启，但该明确指令没有实际渲染进 Llama prompt。普通的“仅依据资料、不要推测”没有算作明确允许说资料不足。

已按模板设置、完全不看回答或标签表现冻结两套索引：

- `ALL_TRAIN_INDEX.jsonl`：1,842 行、943 个问题、917 个推荐材料组。
- `NO_UNANSWERABILITY_HINT_INDEX.jsonl`：971 行、498 个问题、491 个推荐材料组，仅使用 `settings.unanswerability_hint == false`。

同一个官方问题组没有混用 hint 设置，因此 no-hint 子集不会拆开同题的 answerable/unanswerable 配对。以后应把 all-train 与 no-hint 当作预先规定的两组对照，不能看验证分数后挑其中较高者。

索引冻结后再做的标签描述显示：no-hint 中 542/971（55.82%）回答有 span，27,816/99,649（27.91%）可评 4-BPE 窗口为风险；all-train 对应 1,007/1,842（54.67%）和 53,841/197,452（27.27%）。两者风险比例接近。这只是样本组成说明，不是模型成绩，也没有参与子集选择。

## `full_chat` 与 `token_starts`

`full_chat` **不等于**无分隔符的 `full_prompt + output`，官方格式在两者之间固定插入一个空格：

- 1,840 条精确等于 `full_prompt + " " + output + "</s>"`；
- 2 条精确等于 `full_prompt + " " + output`，缺少末尾 EOS；
- 其他格式 0 条。

官方 `token_starts` 也不能原样当回答 token 起点：全部 1,842 条开头都有两个 0，分别包含 BOS 哨兵与第一个回答 token；1,840 条末项等于回答字符长度，代表终止边界。去掉首个哨兵和末尾终止边界后：

- 1,813 条与本地 `LlamaTokenizerFast` 起点逐项完全一致；
- 1,840 条 token 数量一致；
- 29 条存在字符边界差异，其中仅 2 条 token 数量相差 1。

字符 span 本身是可靠主标签：1,483 条全部精确切中原回答，且全部含词面字符并能映射到本地 BPE。只有 349 条 span 的首尾都恰好落在本地完整 token 边界，所以不能把它理解为“天然整 token 标签”；应继续使用字符交集映射。

## 4-BPE 映射结果

映射使用本项目现有本地 `Llama-2-7b-chat-hf` fast tokenizer：`完整官方 full_prompt + 一个空格 + 原回答`，`add_special_tokens=false`、不截断；只对回答轴形成宽度 4、步长 1 的 raw BPE 窗口。

| 项目 | 数量 |
|---|---:|
| 回答 raw BPE | 203,286 |
| 可评词面 BPE | 174,928 |
| 风险 BPE | 45,664 |
| 全部 4-BPE 窗口 | 197,760 |
| 可评词面窗口 | 197,452 |
| 风险窗口 | 53,841 |
| span 映射失败 / 非空白字符丢失 | 0 / 0 |
| 最长完整输入 | 1,568 BPE |

因此，当前 4-BPE 管线可以直接处理全部 1,842 条回答。正式准备时应以本地 tokenizer 的完整字符串 offsets 为准，把官方 `token_starts` 留作审计字段；若以后要求与作者历史 token 边界逐项一致，应单独隔离那 29 条，而不是静默修正。

## 来源与问题分组

- 1,842 行对应 943 个官方问题/来源主体：899 个问题各有两行，44 个只有一行。
- 规范化问题、官方 `user_prompt_index` 和源文章身份均得到同样的 943 组。
- 精确检索上下文有 1,816 种；22 个上下文哈希重复，涉及 48 行。
- 按“相同官方问题、规范化问题、源文章或精确检索上下文”做传递合并，得到 **917 个推荐组**；最大组 6 行。

`GROUP_INDEX.jsonl` 只保存行号、哈希和组 ID，不复制正文。以后切分必须按 `recommended_group_id`，不能按回答随机切分。由于严格没有读取 test，本报告无法验证 train 与官方 test 之间是否还有同源或近重复；这一点保持未知，不用测试内容补查。

## 使用边界

本批数据适合后续训练准备，优先比较 all-train 与预先冻结的 no-hint 子集。它不能代替人工标注外测，也不能单独证明模型在用户要求的“未提示资料不足”场景中有效。本轮没有模型前向、训练、GPU 操作或 baseline 修改。
