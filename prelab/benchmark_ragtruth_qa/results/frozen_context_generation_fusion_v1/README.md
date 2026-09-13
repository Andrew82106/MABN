# 固定语义模型，仅训练融合头

状态：固定骨干缓存及两个 CPU 模式的六轮训练均已完成，进程均退出0。结果见 `REPORT.md`，真实会话与进程记录见 `RUN_RECORD.json`。以下保留已执行的固定流程说明；完成的任务禁止隐式覆盖或重跑。

1. `cache_frozen_context_fusion.py` 从上游最终选中权重，以原完整输入、batch1、BF16 前向和原字符映射缓存 3,839 答的 708,506 个原始 BPE 状态：768 维末层状态、实际原始 logit 和 GPU 概率。数组约 2.18 GB。保存到独立 `results/full_context_frozen_cache_v1`，不改原模型或旧产物。核对原逐词元概率和两级分数，在原阈值下记录计数差异；逐词元概率差超过预定 2e-6 则保留失败记录，不完成缓存。
2. `train_frozen_context_fusion.py` 在 CPU 上只训练 52,944 参数的融合头，主模型不加载、不更新。复用原 LB/NLL、权重和六轮次序；`fusion` 与零生成输入的 `semantic_only` 使用同结构、同六轮预算。AdamW lr=1e-4、wd=.01、clip=1，累积八答，损失总质量 560,300。

CPU 零残差要求 logit 与缓存完全相同。CPU sigmoid 与 GPU 概率的微小差异、原阈值下计数单独记录；没有概率替换或梯度特判。两个模式各保存 0–6 轮，0 轮仅诊断；仅从 1–6 轮按既定校准双 F1 规则选头及阈值。

后续由根代理调度，顺序命令为：

```powershell
# 上游六轮结束并获得独占GPU后：
prelab/.venv/Scripts/python.exe prelab/benchmark_ragtruth_qa/src/cache_frozen_context_fusion.py infer
# 缓存进程成功退出、GPU交还其他任务后，仅CPU：
prelab/.venv/Scripts/python.exe prelab/benchmark_ragtruth_qa/src/train_frozen_context_fusion.py train --mode semantic_only
prelab/.venv/Scripts/python.exe prelab/benchmark_ragtruth_qa/src/train_frozen_context_fusion.py train --mode fusion
```

这是额外语义核查模型与生成白盒信号的两阶段检测，不是纯生成探针。上游已用同一校准集选型，训练缓存也来自已拟合上游的训练样本，不能称交叉拟合或独立验证。低容量只限制本阶段可更新参数，并不保证优于既有模型。
