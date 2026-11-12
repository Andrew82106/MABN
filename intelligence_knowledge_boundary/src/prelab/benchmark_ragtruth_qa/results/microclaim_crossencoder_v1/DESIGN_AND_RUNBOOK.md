# Microclaim cross-encoder v1：设计与运行命令

## 固定主候选

每个样本是一条原子微主张。NLI 的 premise 包含问题与三篇完整检索资料，hypothesis 是该微主张。使用原始 `tasksource/ModernBERT-base-nli` 三分类结构，不增加新头：

`risk_logit = logsumexp(neutral, contradiction) - entailment`

全部参数以 1e-5 学习率微调 1 epoch。fit 按 source-connected group 做 5 折 OOF；另训 1 个 full-fit 模型预测 calibration。模型和阈值都不使用 calibration 标签。

## 针对旧 NLI 误报的处理

旧 NLI 把“没有明确蕴含”普遍当作风险，新增了 1,524 个 calibration 误报窗口。这里用两步处理：

1. 输入全部检索资料，避免 BM25 只取局部句子造成“其实有依据，但模型没看到”。
2. 对 fit 中 gold 干净、但冻结 NLI 风险高的微主张增加训练关注度。倍率固定为 `1 + 2*risk²`，并在每个回答内部重新归一化，保持该回答的干净样本总权重不变；所有正例权重保持原值。

独立 CPU 审计显示，6 套训练权重里正例只有 float32 存储舍入差（最大 `4.62e-7`），每个回答的干净样本总质量差小于 `5.30e-7`。

## 唯一控制

同一冻结 NLI 模型、同一完整输入直接输出 `1-P(entailment)`。它只做一次前向，不训练第二个骨干。

## 数据与资源

- fit：9,055 条微主张，1,291 正例，615 个组。
- calibration：2,267 条微主张，363 正例，154 个组。
- 输入长度：中位 327、P95 528、最大 697；固定上限 768，零截断。
- 每折 train/held 的 group、source、response、answer hash 和 passage hash 交集均为 0。
- RTX 3070 smoke 峰值 reserved 3.21 GB；总流程点估 37 分钟、保守 55 分钟；6 个 FP32 checkpoint 约 3.59 GB。

## 精确命令

```powershell
$py = 'prelab/.venv/Scripts/python.exe'
$runner = 'prelab/benchmark_ragtruth_qa/src/run_microclaim_crossencoder_v1.py'

& $py $runner prepare
& $py $runner check
& $py $runner cpu-tiny
& $py $runner gpu-smoke

foreach ($fold in 0..4) {
  & $py $runner train-fold --fold $fold
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
& $py $runner train-full
& $py $runner control
& $py $runner finalize
& $py $runner verify-final
```

正式 baseline、统一 4-BPE stride-1 评测和 official test 均不改动。
