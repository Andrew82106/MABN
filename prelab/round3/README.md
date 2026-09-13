# 第三轮：探针结构与针对性补充信息

状态：两条路线的实验及独立核对已完成。18个网络配置、30次网络训练、10种补充信息融合对照，完整新测试共51组指标（含种子及集成）。见[结果报告](results/REPORT.md)和[全部结构参数记录](results/STRUCTURE_SEARCH.md)。

主要发现：复杂网络未稳定胜过组合特征线性探针；当前长解释融合未改善检测。三选一短核查更有希望改善回答级风险筛选，但逐词定位增量未证实，主阈值只检出7/40条错误摘要。全部80条新测试输出与人工错误区间保存在[风险可视化](results/all_test_examples.html)。

证明什么：在固定人工新闻标签下，改进检测器及追加核查是否能改善事实冲突定位和低误报检出。

怎么证明：训练／验证沿用第二轮 240／65 条 Mistral 新闻，旧测试和复核集只作历史结果；另冻结未用来源的 Llama-2-7B 新闻作为跨生成器测试。均由本地 Qwen2.5-7B NF4 重放或追加回答；不调用其他大模型 API。仅在训练／验证上选参数，最终新测试一次性评估。

结构实验：逻辑回归、不同宽深 MLP、因果卷积与 GRU；注意力和状态组合特征；学习率、正则、位置损失对照。所有候选使用同样的位置标注，按验证全局 token AUROC 选定，再报告三种子结果。报警阈值在验证集固定，新测试同时报告误报与检出。

补充信息实验：固定第一阶段线性注意力探针，训练样本用按来源三折的折外预测挑选可疑句子，另选随机句子作等调用次数对照。对每个选中句子比较核查提示的输出前状态、要求摘证据后判断的追加回答、普通扩写。补充阶段以同一个 Qwen 完成，标签仍取原新闻人工标注，不把生成内容作为金标。未选句子也参与评估，补充特征置零，原分数经同一融合器校准；报告整个摘要与选中句子覆盖率，不能只统计挑中的错误。

产出：可复跑代码、冻结数据与配置、完整预测和成本、结果报告；无论结果正负都不将简单调参或已有提示方法称为原创算法。旧测试均已被查看，新测试生成器变化需单独披露。

训练／验证阶段的选句检查已完成：验证集25条错误摘要，主动选句覆盖14条，随机选句覆盖8条。此结果仅用于解释选句能力，不当作新测试成绩，见 `results/selection_train_val_audit.json`。

解释器：`../.venv/Scripts/python.exe`，沿用用户授权的本地环境。GPU 作业串行。详情见 `configs/protocol.json`。

## 复跑顺序

在仓库根目录运行以下命令。沿用第二轮已冻结的数据、训练／验证特征、7B权重和HaMI检查点；路径保存在数据文件中，迁移机器时需调整根路径。脚本会复用已有缓存；更改样本、模型或提示时应另建实验目录。

```powershell
$round3Python = 'D:\Projects\Multi_Agent_Graph_Analysis\prelab\.venv\Scripts\python.exe'
& $round3Python prelab/round3/src/prepare.py
& $round3Python prelab/round3/src/cache_test.py
& $round3Python prelab/round3/src/prepare_inputs.py
& $round3Python prelab/round3/src/tune_structures.py
& $round3Python prelab/round3/src/prepare_queries.py
& $round3Python prelab/round3/src/supplement.py --batch-size 1
& $round3Python prelab/round3/src/fusion.py
& $round3Python prelab/round3/src/evaluate_round3.py
& $round3Python prelab/round3/src/audit_supplement_states.py
& $round3Python prelab/round3/src/explore_direct.py
& $round3Python prelab/round3/src/verify.py
& $round3Python prelab/round3/src/report.py
```

每步成功后再运行下一步。本轮已保存的补充回答来自批次4、2、1的运行，运行协议记录了变更。单批重跑可能有量化数值差异，不能承诺逐字相同；原始token和逐批记录均保留。

`explore_direct.py`是在查看测试结果后补做的不确定性描述，复用固定预测，不重训或改阈值，不能冒称预先指定的验证。主比较及其结果仍保留原样。
