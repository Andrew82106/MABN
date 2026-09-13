# NLI 编码器初始化对照：CPU 准备完成，尚未训练

仅把完整资料检测器的编码器初始化替换为 [tasksource/ModernBERT-base-nli](https://huggingface.co/tasksource/ModernBERT-base-nli)，固定 revision `de4ab7e77845098b7fab7f6ab9d370ddff27b19c`。这属于额外语义核查模型，不是原生成模型的内部探针；完整回答可见，因此是离线检测。

## 已核验

- 下载文件 598,442,860 字节；模型 SHA256 `86c32c52ce38b8f26e028ca959b06daee3a5f3f6947c63258bc8695dde88a465` 与官方 LFS 一致。小文件也核了官方 Git blob 身份。
- 134 个 `model.*` 编码器张量严格载入并与固定 NLI 权重逐值一致。它们都不同于通用预训练编码器。
- 继承原通用模型的 `head.dense.weight`、`head.norm.weight`；二分类 `classifier.weight/bias` 按原 seed 新建。四个非编码器张量及初始化后的 CPU 随机状态均与原 base v2 完全一致。
- 丢弃 NLI 的预测变换和三分类输出，共四个张量；没有把已有的三分类输出映射成风险标签。最终模型 149,606,402 参数，均为可训练 FP32。
- 3,680 训练答和 159 校准答的完整输入 IDs、全输入 offsets、回答坐标、原始 Llama BPE 映射逐值一致。随后才逐字节复制原输入、权重和六轮顺序。总计 1,857,538 编码器词元、708,506 原始 BPE。
- RoPE、局部/全局注意力、激活、归一化和 dropout 等前向配置一致。只有长度上限由 8,192 变为 2,048；全部实际输入最长 979。NLI tokenizer 的巨大哨兵值不作为长度许可。两者均强制关闭 reference_compile 并使用原 SDPA。
- 实际 CPU 模型前向有限且重复一致；原训练前向/映射在 tiny 合成例子上通过梯度和 FP32 检查。没有真实 QA 训练，没有 CUDA 初始化，没有读取官方 test。

## 固定后续预算

完全复用原 base v2 训练循环：六轮、seed 20261005、AdamW lr 1e-5 / wd .01、八答梯度累积、clip 1；BF16 前向，参数、梯度、Adam 状态、映射和损失均 FP32。原来源组/回答/词元权重及总质量 560,300 不变。

每轮保存全部预测、检查点和两级指标。仅校准集选择阈值及轮次；epoch 0 仅诊断，epoch 1–6 按原 min 两项 F1、窗口 F1、窗口精确率、较早轮次规则选择。原四个 BPE 窗口和整答 max 口径不变。

GPU 自检和训练尚未执行，需要根代理另行排程。参数量和实际输入规模相同，原 base 六轮用时约 42.6 分钟可作预算参考，不能当成本次实测。

```powershell
prelab/.venv/Scripts/python.exe -X utf8 prelab/benchmark_ragtruth_qa/src/run_full_context_nli_initialization_v2.py check
# 下两条仅在根代理安排 GPU 后执行：
prelab/.venv/Scripts/python.exe -X utf8 prelab/benchmark_ragtruth_qa/src/run_full_context_nli_initialization_v2.py gpu-smoke
prelab/.venv/Scripts/python.exe -X utf8 prelab/benchmark_ragtruth_qa/src/run_full_context_nli_initialization_v2.py train
```

## 来源限制与版本记录

这是**来源可追溯范围有限的探索性 NLI 初始化**。已查任务清单未发现直接 RAGTruth 专项任务证据，但没有证明全部上游文本与评测材料独立。部分上游使用合成文本或银标；TrueTeacher 的原始说明涉及 CNN/DM train、T5 摘要及 FLAN-PaLM 判断，具体保留的 2,000 行未逐一追溯。旧 `mixtral_small_zeroshot` 也不能直接等同后来的版本。详见相邻 `nli_initialization_source_check/TASK_NAME_COUNTS.md` 与 `TWO_TASK_SOURCE_NOTES.md`。不称“已证明无污染的干净基线”。

v1 在 CPU 接线时发现 Transformers 的显式 config 与重复 `reference_compile` 参数不兼容，未发生任何训练。v1 源、准备结果和失败报告完整保留；v2 仅移除重复关键字，保留 config 中的 False，再重新完成 CPU 核验。没有改变训练参数或查看成绩后选配置。

证据文件：`protocol.json`、`INPUT_AGREEMENT.json`、`CONFIG_AGREEMENT.json`、`CPU_SELFCHECK.json`、`source_snapshot.json`、`preparation_complete.json`。
