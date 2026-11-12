# 第二轮预实验

第二轮已完成：扩大到 7B、完成五组主实验和 110 条新来源新闻复核，审计基线实现，区分整段监督与局部监督。第一轮结果保留在上级 `results/`。

主要发现：整段短答可检测；新闻中位置监督改善局部排序，但可靠的低误报自动报警仍未解决。三事实自动匹配误标签严重，仅作数据质量诊断。

- [基线实现范围](BASELINES.md)
- [协议与固定参数](configs/protocol.json)
- [结果报告](results/REPORT.md)
- [完整性核验](results/final_audit.json)、[报警指标独立核验](results/alert_metric_audit.json)
- [小模型状态核验](results/feature_audit_small.json)、[大模型状态核验](results/feature_audit_large.json)
- [额外新闻复核](results/confirmation/confirmation_summary.json)
- [自动标签抽查及修订](results/label_rule_audit.json)

已完成主实验：0.5B 与 7B 的新闻重放、两模型各 2,700 个独立问答，以及 7B 的 1,050 个三事实 RAG 回答。数量指生成总量，过滤后的训练／验证／测试可用量见报告。0.5B 三事实任务只进行了可用性试跑，未列为完成的基准，原因见 [记录](results/rag_small_pilot_decision.json)。

解释器沿用已授权的 `../.venv/Scripts/python.exe`。在仓库根目录依次运行：

```powershell
& prelab/.venv/Scripts/python.exe prelab/round2/src/download_assets.py
& prelab/.venv/Scripts/python.exe prelab/round2/src/prepare_benchmarks.py
& prelab/.venv/Scripts/python.exe prelab/round2/src/run_inference.py --model large --task news
& prelab/.venv/Scripts/python.exe prelab/round2/src/analyze_run.py --model large --task news
```

将 `--task` 换为 `trivia`、`rag` 分别执行；小模型使用 `--model small`。不要对未完成的生成运行分析。缓存可续跑；改变数据、模型或特征协议应另建实验目录。已有检查点会被复用，标签变更后不得不经处理直接重用旧检查点。

```powershell
& prelab/.venv/Scripts/python.exe prelab/round2/src/audit_features.py
& prelab/.venv/Scripts/python.exe prelab/round2/src/audit_generation.py --model large
& prelab/.venv/Scripts/python.exe prelab/round2/src/verify_alerts.py
& prelab/.venv/Scripts/python.exe prelab/round2/src/verify_round2.py
& prelab/.venv/Scripts/python.exe prelab/round2/src/build_report.py
```

额外新闻复核按 `prepare_confirmation.py` → `alarm_control.py` → `cache_confirmation.py` → `evaluate_confirmation.py` → `confirmation_intervals.py` 运行。`alarm_control.py` 只读旧训练／验证；新新闻从不用于拟合和阈值校准。大模型特征提取与探针 GPU 训练顺序执行，避免 8GB 显存溢出；CPU 评估可用 `evaluate.py --cpu`。

模型、数据、特征和权重不进入 Git；记录、代码与报告可追溯。本轮 7B 为第三方量化权重，已通过固定上游 SHA-256 校验。原始数据和模型来源记录在 `results/` 的 manifest 中；实际依赖版本见 [runtime.json](results/runtime.json)，补充依赖见 [requirements.txt](requirements.txt)。
