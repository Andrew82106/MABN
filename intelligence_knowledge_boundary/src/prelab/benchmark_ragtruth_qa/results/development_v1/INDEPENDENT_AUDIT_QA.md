当前 QA 基线的均值公式、标签坐标和 12 个 LR 数值审计通过，未发现阻断错误。但应称为 **“只统计检索资料、读取当前词元后计算的 Lookback 适配版”**，不能说精确复现了原版实验。尚无消融证据证明这些适配解释了当前性能。

| 检查 | 当前实现 | 官方定义／实现 | 判断 |
|---|---|---|---|
| 比值 | 资料注意力均值 ÷（资料均值＋回答均值） | 同样使用两个均值 | 公式一致；改成两个总和反而改变定义 |
| 查询位置 | 当前回答词元读入后的位置 i | 生成当前词元前的最后前缀／上一回答位置 | 明确适配，相差一步 |
| 当前词元 | 回答池包含当前词元，也保留其自注意力 | 待预测词元尚未加入；查询位置自身注意力仍保留 | 两者都保留查询点的自注意力，但查询点不同 |
| 资料池 | 仅与 released retrieved_passages 字符区间重叠的词元 | 官方 NQ 代码的整个输入前缀，含文档、问题和任务说明；减去固定回答头 | 明确适配；当前问题／说明仍影响模型，只未进入比值的两组均值 |
| 回答池 | 从回答首词元到当前词元，按实际已读数量求均值 | 固定 `#Answer#:` 回答头＋先前生成词元 | 明确适配；没有用最终回答长度作当前均值分母 |
| 数值与模型 | NF4、BF16 矩阵乘法、float32 softmax | 官方生成代码设置为未量化 FP16 | 重建状态，不能称原始生成轨迹 |

依据为 [EMNLP 2024 论文第 2.1 节](https://aclanthology.org/2024.emnlp-main.84/)及[固定官方代码](https://github.com/voidism/Lookback-Lens/blob/e0a1fa3a898fbf6512af7be5567dea8ffe7a6620/step01_extract_attns.py)。官方生成循环先保存注意力，再追加待预测词元，已核对本地固定版本。问题和任务说明虽然不进入当前比值池，仍参与模型计算；共同 softmax 归一化因子在比值中代数消去，不意味着模型忽略了它们。

NLL 用位置 i−1 的分布计算实际词元 i 的负对数概率；hidden 用位置 i 的最终 RMSNorm 后状态。这与当前 LB 的读后定义一致披露，但 NLL 与其不同步。因果遮罩不允许读取后续内容；本审计未证明整段回放与每种长度的在线前缀数值完全相同。

全部 793 条回答、213,159 个原始词元均重新绑定了当前 NPZ/sidecar 哈希、计划、发布原回答、人工 span 与坐标。793 个跨回答边界的首词元都保留；846 段原人工标注未修改。没有零长度／纯非字母数字风险段，也没有“整答有风险而风险词元为零”的异常。既有独立字符级金标审查和两例 GPU 数值自检的哈希仍匹配；后者是两例抽样数值证明，不冒称 793 条另用独立模型实现重算。原生 BF16 softmax 回转与保留 float32 的 LB 最大差约 0.00102，已单列；float32 oracle 比较为零。

| 分区 | 回答／含风险回答 | 来源关联组 | 可评 4 词元窗口／风险窗口 |
|---|---:|---:|---:|
| fit | 634／328 | 615 | 168,123／21,477 |
| calibration | 159／100 | 154 | 42,241／5,984 |

两分区已知关联组无交集；额外 416 个完全没有字母数字的窗口排除，全部优质回答保留。这里未标风险的拒答仍作为负例窗口，与 R16 排除正常拒答定位窗口的规则不同。四个原生 span 类型全部保留，包括 `implicit_true` 和 `due_to_null`。

12 个 LR 共 2,524,368 个窗口分数、9,516 个整答分数按保存系数手工回放，最大差 2.22e−16。fit 窗口身份、组权重、标准化范围正确；4 个标准化矩阵逐值一致。PCA 只用 634 个 fit 回答的 20,288 个固定采样词元，权重／均值复核通过，固定投影得到的全部 210,364 个窗口矩阵逐值一致，未重估 PCA。24 个校准阈值、4 组选 C（均为 0.001）、整答取全部可评窗口最大值及 48 个指标块复算通过。scaler 的累计样本权重显示为 168123.0143，已确认是 sklearn 分块 float32 累加的舍入，不是增加／漏掉样本。

这些是开发集一致性与代码审计。校准数据同时用于选 C、阈值，结果有选择偏乐观；官方 test 仍未打开。原文 AUROC 也不能直接当作当前 F1 目标。重新拟合、模型运行、旧文件修改均未发生。

机器记录：[特征与坐标](D:/Projects/Multi_Agent_Graph_Analysis/prelab/benchmark_ragtruth_qa/results/development_v1/FEATURE_DEFINITION_AUDIT_QA.json)、[系数与 PCA](D:/Projects/Multi_Agent_Graph_Analysis/prelab/benchmark_ragtruth_qa/results/development_v1/COEFFICIENT_AUDIT_QA.json)、[校准与分母](D:/Projects/Multi_Agent_Graph_Analysis/prelab/benchmark_ragtruth_qa/results/development_v1/CALIBRATION_AUDIT_QA.json)、[合并审计与哈希](D:/Projects/Multi_Agent_Graph_Analysis/prelab/benchmark_ragtruth_qa/results/development_v1/INDEPENDENT_AUDIT_QA.json)。
