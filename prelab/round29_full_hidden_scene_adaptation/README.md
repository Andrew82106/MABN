# 原 R16 场景：完整语义隐状态对照

已完成 CPU 五折对照及独立分数复核。只将 R28 的 PCA32+logit 改为完整 hidden768+logit，其他设置和校准预算一致；旧控制零重训。

结果见 `results/REPORT.md`。原 R26 融合后的定位 F1 为 0.722039、整答 F1 为 0.804054。这是原 R16 的开发结果，非新独立测试；标签仍非人标，不替代公共 QA 结果。

入口：`src/run29.py`；冻结协议：`protocol.json`；复核：`results/audit29.py` 和 `results/AUDIT.json`。没有使用 GPU，也未修改任何旧实验文件。
