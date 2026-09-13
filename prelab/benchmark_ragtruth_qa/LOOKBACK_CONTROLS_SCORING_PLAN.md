五种 Lookback 定义共享同一数据、模型和评分口径。每种只使用 LB1024，固定比较 C=`0.001/0.01/0.1`，不混入 NLL、hidden、PCA 或平滑。

共有 15 个候选：四种新定义执行 12 次新 LR；旧版参照直接重放原 `lookback_mean` 的三个冻结模型，不重复拟合同一数据。旧参照的窗口特征、标准化矩阵、分数、校准阈值、指标和最终所选 C 必须与原结果精确一致，否则不开始四种新定义的拟合。

训练仍用 634 个 fit 回答的全部 168,123 个可评窗口。组→回答→窗口的权重、fit-only 类别权重和总损失质量、16,384 行分块 scaler、LR 求解器和随机种子均沿用原 QA 定义。每个窗口仍是四个原始 BPE，标点占位置，特征取四个实际词元向量的均值。整答分数取该回答全部可评窗口分数的最大值。

159 个 calibration 回答、42,241 个窗口只用于原规则的阈值与 C 选择，不参与 scaler 或 LR 拟合。窗口／整答分别选择风险 F1 最好的阈值；阈值并列时依次比较精确率、高阈值。C 按两层 F1 的较小值、窗口 F1、窗口精确率、较小 C 依次选择。所有 15 个候选都保存结果；不是只报告最好的定义。增加对照后的校准结果仍有选择偏乐观，官方测试集保持封存。

正式入口先要求新特征 manifest 完成全部 793 条，并检查原身份、计划、布局、词元坐标及每份文件哈希；随后逐窗口构建五份完整矩阵并与旧参照矩阵逐值核对。没有“部分完成就先拟合／报分”的路径。原回答、人工四类 span、质量筛选、未标风险拒答的负例窗口规则均不变。

已完成的 CPU 验证只涉及合成几何和原冻结结果：三个 C 值各 210,364 个窗口、793 个整答的回放全部精确一致，没有新拟合。新特征实际生成后，还需再次通过其与旧特征／矩阵的门禁；现在的回放通过不能代替那一步。

源码：[run_lookback_controls.py](D:/Projects/Multi_Agent_Graph_Analysis/prelab/benchmark_ragtruth_qa/src/run_lookback_controls.py)。协议：[lookback_controls_scoring_protocol.json](D:/Projects/Multi_Agent_Graph_Analysis/prelab/benchmark_ragtruth_qa/lookback_controls_scoring_protocol.json)。验证：[baseline_cpu_check.json](D:/Projects/Multi_Agent_Graph_Analysis/prelab/benchmark_ragtruth_qa/results/lookback_controls_v1/baseline_cpu_check.json)。

新特征全部完成后，运行 `prelab\.venv\Scripts\python.exe -X utf8 prelab/benchmark_ragtruth_qa/src/run_lookback_controls.py fit`。这一步仅使用 CPU，预计 12–30 分钟为预算，尚未实测；五份原始窗口矩阵约 4.01 GiB，四份标准化训练矩阵约 2.57 GiB，正式拟合前要求额外空盘 9 GiB。结果仅写 `results/lookback_controls_v1`，不改旧结果。
