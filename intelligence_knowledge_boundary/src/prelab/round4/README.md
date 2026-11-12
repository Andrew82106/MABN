# 第四轮：事实点核查与F1优化

本轮升级、两批测试和复算审计均已完成，**尚未证明能稳定达到F1>0.7**。指标是回答级、事实错误为正类的二分类F1，逐词定位单独报告。

第一批100条新来源：验证集预选主方法F1为0.645；两个预设对照分别达到0.709（三句直接自检）和0.703（句子＋事实点内部状态探针）。冻结后两种方案和原探针对照，再测50个完全不同的新来源：内部状态探针为0.588，直接自检为0.562。不能只用第一批最高分宣称目标完成。详见[第一批报告](results/REPORT.md)和[独立复核报告](confirmation/results/REPORT.md)。

复核的内部状态探针检出10/20条错误，误报4/30条正常；逐词F1仅0.130。10条漏检中，5条没有选到错误句子，另外5条虽然选中了错误句子仍未识别。继续改进需要同时处理核查范围与事实关系判断，单纯提高网络复杂度或降低阈值没有获得可靠证据。

此前看过的584条新闻全部列为开发数据，按生成器和标签分层重新划分464训练、120验证。另冻结100个全新来源作为测试（40错误、60正常）；所有参数、方法、汇总方式和阈值仅在验证集选择。测试集不参与追逐目标分数。

同一本地Qwen2.5-7B NF4重读新闻摘要，读取A/B/C输出分数与提示末端内部状态，不调用额外大模型。比较单句、最多三句、最多三个事实点、整篇及组合核查。事实点通过固定规则和折外探针风险选择，核查分数回填原摘要的对应字符范围；训练仍使用原人工标注。

旧低误报阈值并非为F1设计，本轮另在验证集平衡精确率和召回率。除F1外保留混淆矩阵、误报、始终报错基线、逐词定位、覆盖率和成本。真实联网、自行生成时的效果不在本预实验中冒称已验证。

解释器沿用 `../.venv/Scripts/python.exe`，没有安装新依赖。原始数据与Qwen权重沿用前轮缓存，迁移机器需更新冻结数据中的绝对缓存路径。

仓库根目录依次执行；每步成功后再执行下一步：

```powershell
& prelab/.venv/Scripts/python.exe prelab/round4/src/prepare.py
& prelab/.venv/Scripts/python.exe prelab/round4/src/cache_test.py
& prelab/.venv/Scripts/python.exe prelab/round4/src/make_queries.py
& prelab/.venv/Scripts/python.exe prelab/round4/src/readouts.py
& prelab/.venv/Scripts/python.exe prelab/round4/src/fit.py
& prelab/.venv/Scripts/python.exe prelab/round4/src/evaluate.py
& prelab/.venv/Scripts/python.exe prelab/round4/src/audit_readouts.py
& prelab/.venv/Scripts/python.exe prelab/round4/src/audit.py
& prelab/.venv/Scripts/python.exe prelab/round4/src/report.py
```

本轮首次结果已知后，按顺序运行 `prepare_confirmation.py`、`run_confirmation.py`、`report_confirmation.py` 完成独立复核。模型权重与两种阈值全部沿用第一批的冻结配置。复核新样本50条含20错误、30正常，排除此前全部684个来源；剩余Llama错误来源不足，故在冻结样本前扩大至原数据中的五种生成器，每种均保持40%错误。具体两次不可行样本规模的准备尝试及最后抽样规则保存在[复核协议](confirmation/protocol.json)，均未查看新预测。

`prepare.py`、`make_queries.py`保留已有冻结数据；`readouts.py`按查询ID续传，完整缓存不重复推理。`fit.py`写出验证选择后，`evaluate.py`才做最终测试。更改数据、提示或方法应新建轮次，不能把旧缓存和新配置混用。

结果入口：[报告](results/REPORT.md)、[所有测试样例](results/all_test_examples.html)、[固定实验协议](configs/protocol.json)。
