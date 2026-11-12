# 绝对 Lookback 统计量旁路：待训练候选

状态：CPU 检查及准备完成，未使用 GPU、未训练。旧融合 v1 与正在运行的纯语义 ModernBERT v2 均未修改。

保留原 LayerNorm 分支，在融合向量末尾追加同一原始词元 1024 个 Lookback 值的均值及两倍总体标准差。两值天然处于 [0,1]，不拟合缩放器、不再对其做 LayerNorm。融合维度 65→67，参数 52,940→52,944；残差仍零初始化。`semantic_only` 的两个旁路值始终为零。

CPU 小型 ModernBERT 检查通过：初始输出与原基线完全相同，正常反传且新增残差权重取得有限非零梯度；整体平移 0.25 后均值旁路正确变化，标准差不变；固定合成读出可区分该平移。合成读出没有训练，也不是性能结果。

67 维 gate 的默认初始化尺度随输入维度改变，因此不声称旧 gate 各权重与 v1 相同；零残差保证初始最终输出相同。融合头初始化仍隔离 RNG，不改变主模型随机状态。

六份旧准备文件全部按原 SHA256 复制并核验：输入、原始生成信号、边界、信号清单、损失权重与六轮次序。共 3,839 答、708,506 个原始 BPE 词元；没有重提取、重分词或改标。详见 `reuse_identity.json`、`preparation_complete.json`、`CPU_SELFCHECK.json`。

两个模式均保留原六轮、优化参数、权重、次序及校准选型规则。当前只完成 CPU 命令；日后须单独调度 GPU，显式运行：

```powershell
prelab/.venv/Scripts/python.exe prelab/benchmark_ragtruth_qa/src/run_full_context_fusion_v2.py gpu-smoke
prelab/.venv/Scripts/python.exe prelab/benchmark_ragtruth_qa/src/run_full_context_fusion_v2.py train --mode semantic_only
prelab/.venv/Scripts/python.exe prelab/benchmark_ragtruth_qa/src/run_full_context_fusion_v2.py train --mode fusion
```

本修改只补回已证实丢失的两项信息。原训练集均值单信号 AUROC 约 0.522，现有证据不能认定它是主要性能瓶颈，也不能保证 F1 提升。校准仍是反复开发数据，正式测试保持封存。
