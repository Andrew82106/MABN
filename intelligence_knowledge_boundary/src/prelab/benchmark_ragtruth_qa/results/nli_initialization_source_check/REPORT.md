# ModernBERT NLI 初始化来源核查

**结论：可作为明确披露来源不确定性的探索初始化，但目前不建议把它作为干净、独立的下一版基线。** 未发现显式使用 RAGTruth 的证据，也没有足够证据证明未见本项目评测材料或标签。没有下载模型、数据正文或封存答案，也未训练。

- **固定版本：** 官方 API 当前返回 `tasksource/ModernBERT-base-nli` revision **`de4ab7e77845098b7fab7f6ab9d370ddff27b19c`**，最后修改 2025-01-06。模型卡称多任务训练 200k 步；该版本 config 列出 **133 个不同任务 ID**，没有名称包含 RAGTruth 的项目。模型为 22 层、768 维的 ModernBERT-base 序列分类器，三类为蕴含/中立/矛盾，`max_position_embeddings=2048`；不能直接按普通 base 的 8192 长度使用。[固定模型卡](https://huggingface.co/tasksource/ModernBERT-base-nli/blob/de4ab7e77845098b7fab7f6ab9d370ddff27b19c/README.md)、[完整任务配置](https://huggingface.co/tasksource/ModernBERT-base-nli/blob/de4ab7e77845098b7fab7f6ab9d370ddff27b19c/config.json)、[官方版本 API](https://huggingface.co/api/models/tasksource/ModernBERT-base-nli)

- **`dataset_train_nli` 的真实入口：** Tasksource 的 `moritz_zs_nli` 从 **`MoritzLaurer/dataset_train_nli`** 读取 `text`、`hypothesis`、`labels`，并过滤 `task_name` 为 mnli、anli、fevernli、wanli、lingnli 的行；并非只取这些经典 NLI 数据。所查代码 commit 为 `ef6535aebaed3f6b9c72a833e63106313fdadac0`。这是可核验的当前代码映射，模型卡没有把该代码 commit 与实际训练运行绑定，不能冒充训练时快照。[Tasksource 源码第1105–1108行](https://github.com/sileod/tasksource/blob/ef6535aebaed3f6b9c72a833e63106313fdadac0/src/tasksource/tasks.py#L1105-L1108)

- **无法闭合的来源链：** 该数据集 revision 为 `1e009645b2943106614107b06107b1ee85ac1161`。官方卡片只有结构元数据：一个 train 划分、**1,018,733 条记录**、蕴含/非蕴含两类及 `task_name` 字段；没有各 task_name 的完整来源清单、原数据划分或逐样本来源记录。因此不能证明这个集合及其被保留子集未包含评测来源。模型任务表还包括 doc-nli、seahorse 等复合任务，名称检索不构成逐样本隔离证明。[数据集固定卡片](https://huggingface.co/datasets/MoritzLaurer/dataset_train_nli/blob/1e009645b2943106614107b06107b1ee85ac1161/README.md)、[官方数据元数据 API](https://huggingface.co/api/datasets/MoritzLaurer/dataset_train_nli)

8GB 下，该 base 规模用于当前最长约979 token 的 QA 路线在资源上有可行性；但资源可行不等于来源干净，也不保证冲突定位提升。当前人工辅助训练与已完成 CPU 融合保持原样，不因本次核查改数据、切换模型或增加训练。

补查公开时间：该 NLI 仓库 `createdAt=2024-01-21T14:33:35Z`、`lastModified=2024-01-21T14:33:43Z`；唯一数据文件 `data/train-00000-of-00001.parquet` 也在当前 revision `1e009645b2943106614107b06107b1ee85ac1161`、同日14:33:43Z提交，公开提交历史此后没有更新。该文件 LFS SHA256 为 `e4d948bb5c64f799a9fc4f908d5421c6e0560f718df15b45387b9f037bd21e25`。这些来自官方元数据，不需要下载数据正文。[提交历史](https://huggingface.co/datasets/MoritzLaurer/dataset_train_nli/commits/main)、[文件提交元数据](https://huggingface.co/api/datasets/MoritzLaurer/dataset_train_nli/tree/main?recursive=true&expand=true)

但是 RAGTruth 论文 v1 提交于 **2023-12-31**；其官方仓库明确的 **“release dataset”** 提交 `87081c533b0019674c46cdad1b8de746c9902db5` 为 **2024-01-15T08:21:10Z**。因此 NLI 数据的公开文件比 RAGTruth 数据发布晚约6天，时间先后不能排除它已包含 RAGTruth。这里仍是“无法据时序排除”，不是“已查实污染”；不因为缺子来源说明无限延长检查，但也不能据目前日期将其改称干净独立基线。[RAGTruth 数据发布提交](https://github.com/ParticleMedia/RAGTruth/commit/87081c533b0019674c46cdad1b8de746c9902db5)、[论文提交历史](https://arxiv.org/abs/2401.00396)
