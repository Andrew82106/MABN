已完成协议冻结、可运行入口和 CPU 接线检查，**尚未完整准备数据、GPU 自检或训练**。

`src/run_full_context_fusion.py` 在完整资料、问题和原回答上运行 ModernBERT；其原风险输出保持独立，再叠加一个初始化为零的修正。修正读取对齐至相同原 Llama 词元的语义隐藏状态、原 LB1024 和 NLL。生成信号来自统一 Llama NF4 重放，不能称作五个其他生成器的原生内部轨迹。

固定两种模式：`fusion` 读取实际生成信号；`semantic_only` 用同样网络，把 LB/NLL 全部置零，控制额外网络容量。两者均沿用原 v2 的 3,680 训练回答、159 校准回答、权重、六轮顺序、骨干初始化、原始词元标签和四词元窗口指标。新增 9,678 辅助回答不混入。骨干学习率 1e-5，修正网络 1e-4，其他训练/选轮次预算保持一致。

CPU 检查已验证：额外网络初始化不推进骨干随机状态；相同 dropout 随机路径下，零修正与原风险输出精确相等；骨干和修正输出层均有梯度；旧 fit/cal 四个锚点的原 token IDs/位置/字符坐标与 LB/NLL 缓存一致。见 `CPU_SELFCHECK.json`。

执行顺序如下；GPU 步骤必须由主代理另行排程，不能挤占已排定的 v2 基线：

```powershell
prelab/.venv/Scripts/python.exe -X utf8 prelab/benchmark_ragtruth_qa/src/run_full_context_fusion.py prepare
prelab/.venv/Scripts/python.exe -X utf8 prelab/benchmark_ragtruth_qa/src/run_full_context_fusion.py gpu-smoke
prelab/.venv/Scripts/python.exe -X utf8 prelab/benchmark_ragtruth_qa/src/run_full_context_fusion.py train --mode fusion
prelab/.venv/Scripts/python.exe -X utf8 prelab/benchmark_ragtruth_qa/src/run_full_context_fusion.py train --mode semantic_only
```

`prepare` 在新增 3,046 条缓存完整前只报告 WAIT，不使用子集拟合。完整时机械核对全部 3,839 条原词元坐标、缓存身份和文件哈希，导出约 2.91 GB 的 FP32 生成信号矩阵。两个模式各保留初始化诊断及六轮全部检查点、分数、阈值和指标。GPU 自检只使用最长输入和全零合成目标，不读取真实标签决定结构或精度。

这是带额外语义核查模型的离线检测：完整回答中的未来词元可见，不是实时探针。最终收益须分别对比原 v2 与同容量零生成信号对照；校准成绩仍不能冒充封存测试成绩。
