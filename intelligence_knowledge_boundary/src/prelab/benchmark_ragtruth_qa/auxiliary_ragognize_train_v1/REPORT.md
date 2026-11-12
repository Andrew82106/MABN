# RAGognize train-only 接入报告

## 可用结论

RAGognize 可以安全接入当前项目的**训练侧**。官方 train/test 是物理分离文件，本轮只取得固定 revision 下的 train。`Llama-2-7b-chat-hf` 有完整 1,842 条回答，问题、检索资料、回答和字符 span 均齐全；1,483 个有效 span 全部通过坐标核验并能映射到本项目 4-BPE。

已冻结两套不看标签表现的训练索引：

| 子集 | 回答 | 问题 | 推荐材料组 | 有 span 回答 | 风险 4-BPE 窗口 |
|---|---:|---:|---:|---:|---:|
| all-train | 1,842 | 943 | 917 | 1,007（54.67%） | 53,841 / 197,452（27.27%） |
| no-unanswerability-hint | 971 | 498 | 491 | 542（55.82%） | 27,816 / 99,649（27.91%） |

no-hint 只按 `template_details.settings.unanswerability_hint=false` 选取，索引先冻结、风险比例后统计。两组以后都应报告，不能根据验证结果择优。

## 主要限制

- 标签由模型自动标注，不是人工金标；适合补训练，不能代替人工独立测试。
- 871 条模板元数据开启了“资料不足时拒答”的 hint，实际有 418 条 Llama prompt 明确出现这类指令。no-hint 子集更贴近用户目标场景。
- 官方 `token_starts` 含 BOS/EOS 哨兵；清理哨兵后仍有 29/1,842 条与本地 tokenizer 起点不完全相同。当前字符交集映射全部成功，因此 4-BPE 应以固定本地 tokenizer offsets 为准，不直接使用官方起点。
- 1 条回答有两个发布 span 重叠；保留原标签，二元风险取并集。
- 没有读取 test，所以不能验证官方 train/test 是否存在同源问题或资料；不得反向打开 test 来补做这个检查。

## 下一步转换建议

1. 只从当前 train parquet 导出 Llama-2 记录：`question=user_prompt`、`retrieved_passages=documents_str`、`released_prompt=details.full_prompt`、`original_response=response.text`、`labels=response.hallucinations(valid=true)`。
2. 保留 `ALL_TRAIN_INDEX.jsonl` 和 `NO_UNANSWERABILITY_HINT_INDEX.jsonl` 两个入口，按 `GROUP_INDEX.jsonl` 的 `recommended_group_id` 做任何内部切分。
3. 用 `full_prompt + 一个空格 + response` 完整字符串一次分词；字符 span 按词面字符交集映射到回答 BPE，再形成宽 4、步长 1 的窗口。
4. 将 29 条 token-start 差异和 1 条 span 重叠保留为审计标志，不静默修改或按标签删样本。
5. 后续如做 Llama-2 白盒重放，单独固定模型 checkpoint、tokenizer、prompt、精度和生成/教师强制口径；本轮没有模型前向、训练或 GPU 操作。

详细计数见 `DATA_QUALITY_REPORT.md` 和 `AUDIT.json`；固定字段见 `SCHEMA.json`，下载边界见 `PROTOCOL.json`。
