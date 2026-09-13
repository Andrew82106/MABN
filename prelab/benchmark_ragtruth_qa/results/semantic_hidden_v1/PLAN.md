这是增加一个冻结的 MiniCheck 核查模型后的融合探针，不是单靠原生成模型的白盒信号。MiniCheck 已读取完整陈述和一个资料块；对应到早期词元的状态也含后文，因此只能解释为离线核查。

固定使用原 QA fit 634 答、cal 159 答，test 不打开。资料、回答、人工标注、4 个原始 BPE 的窗口几何和评测分母全部保持原样。

1. 每条自动分句选官方支持分数最高的资料块，同分取首块；选择不使用人工标签。读取该次前向的最后层 1024 维 claim 词元状态。
2. 用原回答中的半开字符坐标映回 Llama 原始 BPE。每个非空白字符被多个 MiniCheck 词元覆盖时先平均，再对 Llama 词元所含字符平均。保留标点的实际状态；纯空白词元为零。任何非空白字符或 lexical 词元缺覆盖即停止，不删样本。
3. PCA64 只在 fit 上拟合。逐答最多 32 个确定原始词元、来源组/回答/样本词元均权，直接沿用已有 PCA 的采样身份与权重。投影后每个原始 4-BPE 窗取均值。
4. 同步固定三个 LR 族：MiniCheck 隐藏64；LB1024＋NLL＋该64；前者再加官方句风险 logit。每族 C=0.001/0.01/0.1，共9次。权重、归一化、阈值和选择规则与冻结 LR 基线一致。
5. 风险标量保持官方映射：词元取与实际字母/数字字符相交的 claim 最大风险；窗取 lexical 词元最大风险，再裁到 [1e-6,1-1e-6] 取 logit。标点仍占窗口宽度，风险标签与可评性不变。

准备脚本为 `src/run_semantic_hidden.py`。`initialize` 只冻结协议并运行合成坐标检查；必须等上游 `claim_feature_manifest.json` 与 `inference_complete.json` 同时完整，才能显式运行 `prepare`。此阶段不加载 GPU，也不会等待未完成的数据后自动启动拟合。

```powershell
prelab/.venv/Scripts/python.exe -X utf8 prelab/benchmark_ragtruth_qa/src/run_semantic_hidden.py initialize
prelab/.venv/Scripts/python.exe -X utf8 prelab/benchmark_ragtruth_qa/src/run_semantic_hidden.py prepare
prelab/.venv/Scripts/python.exe -X utf8 prelab/benchmark_ragtruth_qa/src/run_semantic_hidden.py fit
```

准备输出映射后的原词元隐藏状态、风险/风险 logit、fit-only PCA、逐词元及逐窗投影、覆盖审计和所有输入 SHA；每个 LR 保存模型、scaler、训练身份/权重引用、全部预测及指标。准备约需1.1 GB磁盘；三个 LR 标准化 fit 矩阵约再需1.5 GB。CPU 时间须按实际记录，当前不预报成绩。

LR 完成后才另行冻结一个宽32、感受野7原始词元的 TCN，输入固定为1090维，不依据本阶段 cal 成绩增加结构。本文件不启动该下一阶段。PCA64失败不能说明完整状态没有信息，校准集反复选型后的成绩也不是最终泛化成绩。
