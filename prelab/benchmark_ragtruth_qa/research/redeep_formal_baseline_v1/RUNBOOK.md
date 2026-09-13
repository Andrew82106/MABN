# ReDeEP(Token) 正式迁移运行手册

以下命令只接受已经冻结的 fit/cal 无标签输入。GPU 抽取器内部使用全局 `O_EXCL` 锁，不会自动删除别人的锁。

```powershell
.\prelab\.venv\Scripts\python.exe prelab\benchmark_ragtruth_qa\src\run_redeep_formal_baseline_v1.py --partition all --resume
```

这是唯一正式 GPU 抽取命令。不要传 `--limit`，不要改 `--logit-chunk-tokens` 的冻结默认值 1024。它同时落盘论文公式所需信号和作者源码诊断信号，不需跑两次模型。

完整抽取后先运行身份与逐值复核；它会短暂再次占用 GPU，并且只读 4 条固定 fit/cal 输入：

```powershell
.\prelab\.venv\Scripts\python.exe prelab\benchmark_ragtruth_qa\src\verify_redeep_extraction_identity_v1.py
```

然后按顺序执行完整性审计与三个 CPU 阶段：

```powershell
.\prelab\.venv\Scripts\python.exe prelab\benchmark_ragtruth_qa\src\audit_redeep_raw_features_v1.py
.\prelab\.venv\Scripts\python.exe prelab\benchmark_ragtruth_qa\src\score_redeep_formal_baseline_v1.py check
.\prelab\.venv\Scripts\python.exe prelab\benchmark_ragtruth_qa\src\score_redeep_formal_baseline_v1.py freeze
.\prelab\.venv\Scripts\python.exe prelab\benchmark_ragtruth_qa\src\score_redeep_formal_baseline_v1.py evaluate
.\prelab\.venv\Scripts\python.exe prelab\benchmark_ragtruth_qa\src\verify_redeep_scoring_v1.py
```

`freeze` 只读取共享 fit 标签，先选择具体头/层并冻结 fit/cal 分数；`evaluate` 才读取 calibration 标签。任何阶段都不接受 test 路径。

本轮上述命令均已通过。重跑前必须先核对 `METHOD_FREEZE_PRE_EXTRACTION.json` 及身份验证器中的哈希；不能在 A/B 之间按分数择优。
