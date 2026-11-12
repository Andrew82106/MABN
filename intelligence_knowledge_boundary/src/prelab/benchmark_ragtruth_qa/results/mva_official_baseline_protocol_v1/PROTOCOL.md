# MVA 正式基线：作者代码范围与接线协议

**状态：只完成来源核查；尚未选定可执行的正式配置，未提取特征、训练或读取封存测试集。** 本文只说明如何保持作者算法。现有三类注意力统计是 MVA 的输入，不能单凭这些输入就称完整复现。

来源是 [Hallucinated Span Detection with Multi-View Attention Features（*SEM 2025）](https://aclanthology.org/2025.starsem-1.31/)，锁定作者提交 `f8b871a06b6c18dabe5881bd02a68854b2940b81`。代码文件及逐字节 SHA-256 见同目录 `protocol.json`。本次读取了作者源码和 notebook 的代码单元，没有读取作者数据文件或利用 notebook 输出。

## 1. 应保持的完整分类器

每个完整回答对应一个序列，不按句子或固定小窗拆分。三个信号按 **key_avg → query_entropy → key_entropy** 排列，每类包含全部层、全部头；当前 Llama 32 层×32 头形成 **3072 维**。我们已有提取器输出顺序是 key_avg → key_entropy → query_entropy，接线时仅交换后两个 1024 维块，不能降维、平均层头或增加其他信号。[作者特征顺序](https://github.com/Ogamon958/mva_hal_det/blob/f8b871a06b6c18dabe5881bd02a68854b2940b81/features/optimize.py#L107)

原网络为：

`Linear(3072,d) → dropout → 固定正弦位置编码 → n 层双向 Transformer → Linear(d,2) → 二状态 CRF`。

每层使用多头自注意力、残差、**post-LayerNorm**；前馈层固定为 `d→2048→d`，GELU，带 dropout。只有 padding mask，没有因果 mask；作者还保留每层每头注意力输出。CRF 初始转移矩阵为 `[[1,-1],[-1,1]]`；起止转移沿库默认初始化。分类器从随机初始化开始。[完整结构](https://github.com/Ogamon958/mva_hal_det/blob/f8b871a06b6c18dabe5881bd02a68854b2940b81/features/optimize.py#L657)

**此前提议的 256 维、两层小 Transformer、FFN512、partial-label CRF 均属于我们的适配方法。** 不能代替本表中的正式 MVA 结构。构造函数的 1024 维/4 头/4 层/dropout .3 也只是缺省值，不是作者已发表的最佳配置。

## 2. 序列、标签与标准化

作者先把标签为 -1 的位置连同特征**删除并压紧**，再把完整回答右侧补齐；不是在原序列上保留未知位置的 partial-label CRF。padding 特征为 0，标签为 -1，同时提供标签 mask 和注意力 padding mask。作者以 train、validation、test 三部分的最长有效长度作为统一长度。[预处理源码](https://github.com/Ogamon958/mva_hal_det/blob/f8b871a06b6c18dabe5881bd02a68854b2940b81/features/optimize.py#L259)

三类信号分别按每层每头标准化。**每个分区分别计算均值和样本标准差，而且把补零位置算进去**；标准差为零改为 1。因此它使用评估分区自身的特征分布，不能悄悄替换成我们常用的 fit-only scaler，再称作者代码复现。论文仅写标准化，没有明确这项分区细节。[实现](https://github.com/Ogamon958/mva_hal_det/blob/f8b871a06b6c18dabe5881bd02a68854b2940b81/features/optimize.py#L340)

作者 notebook 的标签转换通过去空格后的错误文字在完整拼接文本中 `find` 首次出现位置，并用模型特定的 `output + 7` 等规则截答案；没有利用原标注 start/end，重复文字可能错位。**允许的数据接口适配**是使用现有准确的回答字符边界、原生 BPE 坐标和原人标字符跨度，替代这些模型专属解析规则，不改变标注事实。[官方预处理 notebook](https://github.com/Ogamon958/mva_hal_det/blob/f8b871a06b6c18dabe5881bd02a68854b2940b81/features/2_ragtruth_preprocessing.ipynb)

原生主评测应覆盖完整回答词元：字符跨度覆盖的标点同样可为阳性，不能把标点一律补成安全。原 QA 的 lexical 标签和 4-BPE 窗口另行保留、绝不覆盖。原代码跳过无标签/全 -1 行；接线必须报告对应数量，不能借此静默删除当前回答。

## 3. 损失、优化和选择

| 项目 | 作者 span＋CRF 路径 |
| --- | --- |
| 损失 | CRF 负对数似然默认求和，再除以本批有效词元总数 |
| 样本权重 | 没有类别平衡、材料组权重或整答辅助损失 |
| 优化器 | AdamW；学习率与 weight decay 来自搜索；其余默认 |
| 批次 | QA batch 64，训练打乱，不丢末批；没有梯度累积或裁剪 |
| 精度 | CUDA autocast＋GradScaler；初始参数 FP32，没有明确改成 BF16 |
| 训练预算 | 论文/训练 notebook 最多 150 epochs、200 trials；CLI 缺省仅 50/50，不能混称同一预算 |
| 搜索时调度 | validation CRF loss 的 ReduceLROnPlateau，factor .5、patience 5 |
| 给定 top 参数重训 | 原代码这一分支**没有上述 scheduler**，不能随意拼接两条路径 |
| 早停及恢复 | validation Viterbi token micro-F1；patience 10、delta .01，恢复最后一次满足此增量条件的模型 |
| 随机性 | 主程序 seed 42；每 trial 不重新初始化随机种子；Optuna sampler 未显式固定 seed |

损失及训练见 [optimize.py](https://github.com/Ogamon958/mva_hal_det/blob/f8b871a06b6c18dabe5881bd02a68854b2940b81/features/optimize.py#L1118)，给定参数重训入口见 [原分支](https://github.com/Ogamon958/mva_hal_det/blob/f8b871a06b6c18dabe5881bd02a68854b2940b81/features/optimize.py#L1363)。早停不是保存每个微小上升的数值最大值。原依赖也并非全部锁版本：Docker 固定 torch 1.12.1＋cu113、Optuna 3.1.0，但 pytorch-crf 等未固定。

代码搜索空间为 d∈{256,512,1024}，heads∈{4,8,16,32}，**span 深度∈{4,6,8}**，dropout∈[.1,.5]，LR∈[1e-5,1e-3]、weight decay∈[1e-6,1e-2]（后两者 log）。论文 Table 2 的深度则为 {2,4,6,8,10,12,14,16}；不能把两个版本混在一起。[作者 span 搜索](https://github.com/Ogamon958/mva_hal_det/blob/f8b871a06b6c18dabe5881bd02a68854b2940b81/features/optimize.py#L1118)

**最大的未补齐项是超参来源。** 论文 §4.3 先在 Data2Text 选择参数，再把相同参数迁移到 QA/摘要；锁定仓库没有提交已选 top5 JSON、分类器权重或 study 数据库。训练 notebook 示例也不是公开的 QA 最佳配置。优先恢复作者 Data2Text 已选参数，再沿原“给定参数重训”路径；若恢复不到，需另行重现该选择过程。直接在我们的 QA 上跑作者搜索，只能明确称“作者代码 QA 搜索迁移”，不能说重现了论文的 Data2Text→QA 选型。

代码保存按 validation F1 排序的 top5，逐个重训/评分，**不是五模型集成**。作者在每个 trial 后会记 test 分数，但目标函数返回 validation F1；我们必须推迟 test 回调，不能因照跑原脚本提前打开官方 test150。

## 4. 后处理与评测口径

正式输出是 **CRF Viterbi 的二元词元序列**，把连续阳性合并为半开区间。没有阈值搜索、后验平滑、迟滞解码或来源传播。[解码与计数](https://github.com/Ogamon958/mva_hal_det/blob/f8b871a06b6c18dabe5881bd02a68854b2940b81/features/optimize.py#L808)

作者名为 span-F1 的实现把跨度展开为阳性词元集合，累计所有回答的交集、预测阳性数和真实阳性数。因此主指标是 **token micro precision/recall/F1**，不是严格跨度完全匹配 F1，也不是我们的 4-BPE 窗口 F1。保存的两个 logits 是 CRF 发射分数，不是 CRF 边缘概率，不能直接包装成官方后验/AUROC。

若以后完整方法身份先冻结并运行，统一任务只允许下面的只读映射：原 Viterbi 二元词元结果按固定坐标投到项目 4-BPE 窗口，窗分数取四个冻结词元位的 max，整答分数再取全部合格项目窗的 max。该映射无参数、无标签且确定性，不重新解码、不加平滑；统一阈值与 F1/AUROC/AP 只在映射落盘后计算。作者 token micro-F1 另列原生复核，不能替代统一主比较。当前因 Data2Text 已选参数未公开，窗口与整答都保持 N/A。

## 5. 我们这里可做的最小接口与尚缺条件

1. 使用原 3839 答（3680 fit/159 cal）及既有材料组隔离，载入锁定的三类信号，作列顺序转换和精确标签映射。所有层头保留。当前是 Llama2 NF4 的离线重放；论文用 Llama3/Qwen，须明确是跨模型、跨数据接口迁移。
2. 原结构、损失、批次、超参选择路径、早停及 Viterbi 全部保持。不得加入我们原来的组平衡、双级 maxmin 选型、语义融合、256 维轻量化或 partial-label CRF。
3. 先补齐 Data2Text 参数身份或明确采用哪条作者代码选型路径。**本文件没有擅自选择一个配置。**
4. 当前只允许开发分区，不能读取 test 以求全局长度或跑作者 callback。开发版仅由 fit/cal 求长度时，须披露它与原三分区预处理的范围差异；最终测试的 padding/分区标准化方式须在开封前明确，不能临时借测试成绩修改。
5. 原 batch64、全序列双向注意力及保留注意力矩阵未在 8GB 上实测。作者使用两张 A6000 48GB；当前 CPU 统计自检不证明完整分类器能跑。若原结构显存不够，保留失败或采用经等价验证的工程执行方式；不能只留下能跑的小模型，再声称完成了原搜索。

**结论：结构可复现，公开的论文最佳参数仍缺失。** 当前正确名称是“作者代码 MVA 输入统计准备完成；正式分类器基线待参数/预处理范围固定”。本轮仅写协议，不新增训练器或实际实验。
