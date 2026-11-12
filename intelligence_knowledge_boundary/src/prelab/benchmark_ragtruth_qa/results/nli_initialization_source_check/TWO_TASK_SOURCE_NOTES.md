# 两个保留任务的来源补查

本附录只跟随每个任务的一条作者官方来源链，不改变先前报告。固定 `dataset_train_nli` revision 为 `1e009645b2943106614107b06107b1ee85ac1161`；计数见 [TASK_NAME_COUNTS.md](TASK_NAME_COUNTS.md)。未下载文本列、模型或本项目封存答案，也未训练。

## trueteacher：2,000 行

Moritz 作者的[数据清单](https://github.com/MoritzLaurer/zeroshot-classifier/blob/e680edcb6d92d8ca09f3d39b70358ccac8ccbd2f/v1_human_data/datasets_overview.csv)明确指向 Google TrueTeacher，并注明 Google、2023、摘要事实一致性、FLAN-PaLM 540B 银标及 CC-BY-NC-4.0。

其直接链接的 [Google 官方说明](https://github.com/google-research/google-research/blob/master/true_teacher/README.md)明确：底层文章来自 **CNN/DailyMail train**；在 XSum 上微调的五种规模 T5 生成摘要；FLAN-PaLM 540B 判断是否符合文章。提供文章 ID、摘要和二分类标签。这是模型生成和标注的数据，不是人标错误片段。

因此，官方描述的原始 TrueTeacher 构造不以 RAGTruth 为任务来源。本次没有追溯 Moritz 文件中具体 2,000 行的选样身份，也没有检验共享新闻原文，不能进一步宣称这些行与所有评测材料独立。

## mixtral_small_zeroshot：33,478 行

作者固定版本的[官方训练 notebook](https://github.com/MoritzLaurer/zeroshot-classifier/blob/e680edcb6d92d8ca09f3d39b70358ccac8ccbd2f/v2_synthetic_data/synth_train_eval.ipynb)包含这个精确名称：读取 `dataset_train_nli` 时明确删除它，再加入 `mixtral_written_texts_for_tasks_v4` 和 `mixtral_refinedweb_nli`。后两者来自作者的[合成数据仓库](https://huggingface.co/datasets/MoritzLaurer/synthetic_zeroshot_mixtral_v0.1)；所查 README 只有配置元数据，没有解释旧名称与新子集的逐样本对应关系。

这证明旧别名确实出现在作者代码中，但**没有闭合其 33,478 行的原始材料、构造版本和划分来源**。不能直接把它认作后来 v4 子集，也不能把集合名称当作它使用或未使用 RAGTruth 的证据。

## 对下一步的影响

三种结论须分开：①明确含 RAGTruth 任务——本次未找到此证据；②没有直接证据但不能保证——**当前整体结论属于此项**；③已证明不含/材料独立——本次未达到。

可将 ModernBERT-base-nli 作为明确披露来源限制的**探索性初始化对照**，无需因无法绝对证明独立性而无限搁置；但不能标为已证明未见评测的干净独立基线。本次不启动下载或训练，也不继续扩展任务链追溯。
