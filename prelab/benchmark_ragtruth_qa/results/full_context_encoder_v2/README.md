固定六轮已实际完成，CPU 保存分数复核通过。选中第 3 轮：窗口 F1 0.641796、整答 F1 0.829493；全部曲线、限制与耗时见 RUN_REPORT.md。官方测试集仍封存。

以下保留训练前精度选择和准备说明（不是当前执行状态）：

CPU 准备、tiny ModernBERT 检查和真实 GPU 小测试已通过，正式六轮训练已排队，尚未开始。v1 源码和全部旧产物保持不变。

实际GPU小测试：最长979词元、全零合成目标，重复差0，单步0.2836秒，峰值3,108,717,056字节；参数、梯度、Adam状态均为FP32，检查点反向重算确认BF16。v1相同输入的单步为0.4404秒。据此选择v2，决定不使用真实标签或校准成绩，见PRECISION_SELECTION.json；单步速度不保证完整训练同样加速。必须等当前Llama特征提取进程明确退出释放GPU后才训练。

唯一训练精度变化是 CUDA 模型前向使用 BF16 autocast；原生非重入检查点会在反向重算时恢复同一 autocast 上下文。参数、累积梯度、AdamW 状态仍是 FP32。两个输出 logit 各自先转 FP32，再相减、映射到原 Llama BPE 并计算 BCE；CPU 不启用 autocast。

3,680 份训练回答、159 份校准回答的完整输入与 v1 逐字节相同，训练权重数组、六轮样本顺序也精确一致。通用 ModernBERT 权重、初始化 seed、所有超参数及按校准集选轮次的规则不变。相关哈希见 `preparation_complete.json`、`source_snapshot.json` 与 `V1_REUSE_AGREEMENT.json`；数值检查见 `CPU_SELFCHECK.json`。

BF16 可能改变预测、梯度和最终结果，不能假定与 FP32 训练数值等价，也尚未证明更快。当前没有读取新训练成绩来选择精度。后续只在主代理分配 GPU 后运行：

```powershell
prelab/.venv/Scripts/python.exe prelab/benchmark_ragtruth_qa/src/run_full_context_encoder_v2.py gpu-smoke
```

小测试固定使用 v1 的最长 979-token 输入和全零合成目标，核重复推理、有限非零梯度、FP32 参数/梯度/优化器状态，以及检查点重算期间的 BF16 autocast；记录单步时间和训练显存峰值。FP32 与 BF16 未训练输出的差只作描述，不作调精度依据。小测试通过后仍由主代理决定是否执行 `train`；不会自动开始训练。
