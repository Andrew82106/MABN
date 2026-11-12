# PsiloQA CPU 词元准备

**15,847 / 15,847 答全部完成，异常 0，未截断。** 实际 CPU 会话 `27112` exit 0；词元处理耗时 32.27 秒。没有模型前向、GPU或训练。

| 数量 | 结果 |
|---|---:|
| ModernBERT 完整输入词元 | 3,372,272 |
| 最长完整输入 | **2,761**（限制 8,192） |
| 输入 95 分位 | 523.7 |
| 辅助 Llama raw BPE | 1,277,020 |
| 其中可评 lexical 词元 | 1,101,613 |
| 自动风险 lexical 词元 | 606,721 |
| 4 raw BPE 可评窗口 | 1,228,839 |
| 其中自动风险窗口 | 744,745 |
| 保留无标记答案 | **635** |
| 映射异常 / 超长 / 丢弃答案 | **0 / 0 / 0** |

完整输入严格由原 `wiki_passage + SEP + question + SEP + llm_answer` 构造。实际分词入口只读取三字段白名单与标签候选文件，不读取 provenance。标准答案、生成器身份、HAL 标注文本没有被额外拼入输入或特征；自然出现于原资料的相同文字不被删除。4 条空 complexity 不参与输入、过滤或分词。

辅助 raw BPE 使用已存在的本地 Llama tokenizer 和完整聊天包装，回答字符边界、标点占位、重叠字节词元都保留。它仅用于和现有 QA 损失、4 BPE 计数兼容，**不是这些原生成器的历史 token 轨迹**，没有运行 Llama 前向。标签仍由原 span 与 `isalnum` 字符交集得到，635 个无标记答案没有被正例条件拒绝。

`psilo_train_6697 / 7004 / 7334` 的发布风险 span 各自只是句号。按既定规则保留整答自动 risk=1，而 lexical 风险词元为 0；未扩展标签或编造风险词元，也没有为此删除答案。这是原标签与计数粒度的差异，单列记录。

只有 `psilo_train_14944` 涉及严格 NFC 映射补齐：原回答字符 179、200、210 的 `U+0301` 分别与前面的 `a` 合成为 `á`，实际 tokenizer 正规化及 source anchor 均给出局部证明。新增字符质量分配给已有 encoder token，原字符、raw offsets、labels 完全不改。其余 15,846 答走原有字符映射路径；未放宽任何不明字符缺口。完整证明在 `nfc_repairs.jsonl`。

所有行运行了独立区间标签 oracle、映射下标与每个非空白 raw token 的质量和检查。原 `IndexedAuxiliary` 实际读取首/中/末三答通过，返回结构仅有 ID/group/input_ids/raw_count/mapping。冻结记录绑定候选、源码、两个本地 tokenizer 文件及包版本；没有改旧模块全局变量。

主要产物为 `token_inputs.jsonl`、`token_byte_offsets.npy`、`answer_index.jsonl`、`attempts.jsonl`、`exceptions.jsonl`、`nfc_repairs.jsonl`、`token_input_preparation.json`、`TOKEN_CHECK.json`、`complete.json`。token_inputs SHA256：`adab09994d14e43760bbbdc3587a3b7c0fc2203e5634c276aae0a835c95124c9`。

这是自动标签辅助数据的准备结果，不是人标数据或模型成绩；无标记位置仅作银标负例。当前训练、校准、封存测试与三份总状态文档均未修改，未安排 GPU 或启动拟合。
