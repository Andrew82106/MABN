# 人工选点诊断

用户授权：用人工标注选准待核查位置，但不向模型提供正确/错误标签，区分选点与判断的瓶颈。

保持现有Qwen2.5-7B、探针权重和回答级阈值不变，比较自动选点、人工指定错误句、再圈准完整错误片段。复用上一轮50条回答作诊断，并匹配20错误＋20正确单位检验给定位置后的区分能力。不是新的独立测试，不能作为自动定位成绩。

结果见[报告](results/REPORT.md)和[全部样例](results/examples.html)。原预实验结果完整保留。

已完成并通过复核：原自动选点的回答级F1为0.588；人工指定错误句子后为0.737，检出10/20→14/20、误报维持4/30；再圈准完整错误片段没有进一步提高回答级检出数。

在20错误＋20正确的给定片段上，冻结内部探针的单位F1为0.780、AUROC为0.847，说明有可区分的信号。这个单位F1不是逐token F1，且全部判错的单位基线已有0.667。直接自检分数的片段AUROC为0.856，尚未证明白盒探针具有额外优势。自动选句是当前瓶颈之一，后续判断、汇总及阈值仍会漏检。原自动定位成绩没有被本诊断替换。

仓库根目录按顺序执行（任一步失败先检查日志）：

```powershell
& prelab/.venv/Scripts/python.exe prelab/round5_oracle/src/prepare_oracle.py
& prelab/.venv/Scripts/python.exe prelab/round5_oracle/src/run_oracle.py
& prelab/.venv/Scripts/python.exe prelab/round5_oracle/src/analyze_oracle.py
& prelab/.venv/Scripts/python.exe prelab/round5_oracle/src/audit_oracle.py
& prelab/.venv/Scripts/python.exe prelab/round5_oracle/src/report_oracle.py
```

环境：`D:\Projects\Multi_Agent_Graph_Analysis\prelab\.venv\Scripts\python.exe`，Python3.11.15。本轮不新增依赖；调用同一本地Qwen，无外部大模型。依赖上一轮已有数据、内部状态缓存、探针和PCA，哈希记录在`protocol.json`。更换模型、提示或数据需新建实验目录。
