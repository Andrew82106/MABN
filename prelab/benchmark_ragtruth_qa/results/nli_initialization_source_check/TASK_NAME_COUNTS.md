# 固定NLI训练文件：任务名单补查

这是对既有来源核查的增量事实记录，不覆写旧结论。仅读取固定revision `1e009645b2943106614107b06107b1ee85ac1161` 的Parquet元数据和 `task_name` 单列；未物化文本、假设或标签列，未下载整套数据、模型，未读本地封存答案。

共 **1,018,733行、34个任务名**。按已审Tasksource加载器规则，剔除5项 **885,242行**，保留 **29项、133,491行**。这是固定数据版本套用该加载规则的结果，不是已经证明模型训练时使用了相同代码/数据快照。

没有名称为RAGTruth的项目。剩余项目中 `mixtral_small_zeroshot` 占33,478行，`trueteacher`占2,000行；仅凭这些集合名仍不能确定其逐样本来源和原始划分。本次不继续无限追溯其他133项，也不把“名称未出现”改写成“无污染”。

| task_name | 行数 | 该加载器处理 |
|---|---:|---|
| agnews | 4,000 | 保留 |
| amazonpolarity | 2,000 | 保留 |
| anli | 162,865 | 剔除 |
| appreviews | 2,000 | 保留 |
| banking77 | 9,508 | 保留 |
| biasframes_intent | 2,000 | 保留 |
| biasframes_offensive | 2,000 | 保留 |
| biasframes_sex | 2,000 | 保留 |
| capsotu | 4,648 | 保留 |
| emocontext | 4,000 | 保留 |
| emotiondair | 5,036 | 保留 |
| empathetic | 4,226 | 保留 |
| fevernli | 196,805 | 剔除 |
| financialphrasebank | 2,524 | 保留 |
| hateoffensive | 2,152 | 保留 |
| hatexplain | 2,958 | 保留 |
| imdb | 2,000 | 保留 |
| lingnli | 29,985 | 剔除 |
| manifesto | 10,000 | 保留 |
| massive | 9,794 | 保留 |
| mixtral_small_zeroshot | 33,478 | 保留 |
| mnli | 392,702 | 剔除 |
| rottentomatoes | 2,000 | 保留 |
| spam | 1,865 | 保留 |
| trueteacher | 2,000 | 保留 |
| wanli | 102,885 | 剔除 |
| wellformedquery | 2,000 | 保留 |
| wikitoxic_identityhate | 2,000 | 保留 |
| wikitoxic_insult | 2,000 | 保留 |
| wikitoxic_obscene | 2,000 | 保留 |
| wikitoxic_threat | 1,760 | 保留 |
| wikitoxic_toxicaggregated | 2,000 | 保留 |
| yahootopics | 10,000 | 保留 |
| yelpreviews | 1,542 | 保留 |

官方查看接口统计为全量，但字符串列没有给出任务名频数，故使用HTTP206范围读取。整文件206,032,209字节，实际传输 **2,357,577字节**（含footer，任务列本身1,166,665字节），共1019行组；会话89621实际exit0，耗时101.11秒。每个范围要求准确Content-Range，传输预算4MB；服务器若返回全文会在读取body前停止。数据文件LFS SHA来自官方元数据，不声称本次对未下载的完整文件进行了本地SHA复算。

可复核产物：[完整机器计数与范围SHA](TASK_NAME_COUNTS.json)、[只读单列脚本](task_name_counts_probe.py)。

来源：[固定数据文件及revision](https://huggingface.co/datasets/MoritzLaurer/dataset_train_nli/blob/1e009645b2943106614107b06107b1ee85ac1161/data/train-00000-of-00001.parquet)、[官方统计接口](https://datasets-server.huggingface.co/statistics?dataset=MoritzLaurer%2Fdataset_train_nli&config=default&split=train)、[Tasksource固定加载器1105–1108行](https://github.com/sileod/tasksource/blob/ef6535aebaed3f6b9c72a833e63106313fdadac0/src/tasksource/tasks.py#L1105-L1108)。
