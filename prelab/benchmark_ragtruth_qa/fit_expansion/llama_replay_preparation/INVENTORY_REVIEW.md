核验通过，无准备阶段阻断。新增 3,046 条均为 fit/train，复用原 634 个来源、615 个组；新回答 ID 与旧 634 fit、159 calibration 均无交集。对更宽的 162 个 calibration 来源及 181 个封存测试/关联来源，回答、来源、组交集也均为零。

完整输入最长 1,207 token，均低于 4,096；答案共 495,347 个 raw token。3,044 条首 raw offset 为 −1，另两条为 0；全部保留首词元，clipped offset 从 0 开始。10 个重复/重叠 offset 原样保留，无答案字符遗漏。计划 schema 无标签内容，文本、token ID、来源绑定哈希及本地 tokenizer 文件哈希均通过。

原生成器（去重后 canonical 回答）：

| 原生成器 | 条数 |
|---|---:|
| gpt-3.5-turbo-0613 | 578 |
| gpt-4-0613 | 614 |
| llama-2-13b-chat | 631 |
| llama-2-70b-chat | 629 |
| mistral-7B-instruct | 594 |

另保留 32 条重复回答的来源记录。生成器只作来源说明，不进入特征计划。

5 × 1,024 维 Lookback、4,096 维 hidden_last 和 1 维 NLL 的 float32 原始载荷为 **18,262,453,196 bytes（17.008 GiB）**。加 token IDs/positions 的 int64 及两套 offsets 的 int32，约 **17.023 GiB**；若坐标全为 int64，约 17.030 GiB。不假定压缩率，未计 JSON、临时文件等开销。

本次只能称为**固定 Llama NF4 对公开答案的教师强制重建**，不能称为五个原生成器的 native trace。仅做 CPU 元数据核验；未加载模型、重分词、前向、训练或读取测试答案/标签。提取源码及数值门禁由后续独立核验。
