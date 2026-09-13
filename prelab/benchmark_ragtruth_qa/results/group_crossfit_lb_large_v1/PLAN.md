# 按材料组交叉拟合：Lookback＋ModernBERT-large

目的是检查：组合器此前看到的是上游模型“已经学过这条答案”的分数，是否因此过度信任语义模型。只改变组合器训练时的两列分数；原159条校准资料、标签和两列输入预测保持不变。

当前状态：`prepare`、`check`、`cpu-test` 均已实际退出0。随后根任务授权的3个固定Lookback已完成，session16074实际退出0，均6次迭代；整折模型重放session42546实际退出0，保存概率逐值相等。large与组合器尚未训练，本分支GPU未初始化。固定协议在 `protocol.json`，数据身份及代码哈希在 `source_snapshot.json`。

## 固定数据与训练

原3680条训练回答来自615个材料连通组。按组ID排序后，用seed20261012随机打乱并轮流分为三折；每折205组。一个组里的原Llama回答和其他生成器回答一起留出，不能只排除其中一条。

| 折 | large训练答 | 留出答 | 原Llama留出答 | 原Llama留出窗口 | 每轮更新 / 末批答数 |
|---|---:|---:|---:|---:|---:|
| 0 | 2474 | 1206 | 208 | 54815 | 310 / 2 |
| 1 | 2456 | 1224 | 210 | 54884 | 307 / 8 |
| 2 | 2430 | 1250 | 216 | 58424 | 304 / 6 |

每折large从同一通用权重、同一seed初始化，固定训练3轮；不使用校准或留出标签选轮。顺序是原large前三轮全局顺序删去留出回答，保留剩下的相对次序。只保存每轮训练日志及最终一套模型、优化器和随机状态。

large折内重新计算“组→原/辅各半→回答→词元”的基础权重，再估计类别平衡并重新平衡组；总质量固定560300。每答损失为加权词元BCE总和乘 `折内训练答数/(本批实际答数×560300)`，累积8答，最后一批按真实数量。优化器仍AdamW、lr1e-5、wd.01、clip1、foreach=False；前向BF16，参数、梯度、映射和Adam状态FP32。

Lookback每折只使用折内原Llama回答的全部窗口，重算scaler和原组/回答/窗口权重；固定C=.0001、liblinear、seed20260924、最多2000迭代，总损失质量168123。没有C搜索；不收敛即记录失败。

三折留出分数拼回原634答/168123窗，每行恰好覆盖一次。另保留此前的训练内两列分数。分别训练两个完全同配置的二输入单调树：100轮、深2、最多4叶、lr.05、L2=1，原634答权重完全相同。两者都只在训练窗口拟合，不用159条校准回答训练树。校准只选择两级阈值；同一候选同时报告窗口和整答F1。

## 已完成的CPU检查

- 原全部3680答权重重算与冻结large文件逐值一致；原634答窗口权重与冻结规则逐值一致。
- 原large逐词元概率重新聚合，与原窗口分数逐值一致；两列输入顺序绑定全部210364个窗口ID。
- 615组不跨折；3680答及168123个原Llama窗口各留出一次；三轮顺序均由旧顺序过滤得到。
- 一个tiny ModernBERT实际调用新训练循环，完成三轮、每轮两次AdamW更新；末批2答、连续优化器状态、FP32映射和有效梯度通过。它不使用正式样本拟合。

## 审核后才可执行的命令

从项目根目录运行；`fit-lb` 已执行完成，不能重复运行。其余命令尚未执行。GPU由root另行排程，每个进程实际退出后才进入下一项。

```powershell
prelab/.venv/Scripts/python.exe prelab/benchmark_ragtruth_qa/src/run_group_crossfit_lb_large.py fit-lb
prelab/.venv/Scripts/python.exe prelab/benchmark_ragtruth_qa/src/run_group_crossfit_lb_large.py train-large --fold 0
prelab/.venv/Scripts/python.exe prelab/benchmark_ragtruth_qa/src/run_group_crossfit_lb_large.py train-large --fold 1
prelab/.venv/Scripts/python.exe prelab/benchmark_ragtruth_qa/src/run_group_crossfit_lb_large.py train-large --fold 2
prelab/.venv/Scripts/python.exe prelab/benchmark_ragtruth_qa/src/run_group_crossfit_lb_large.py combine
```

`combine` 要求全部三折large和LB完整且哈希匹配，才开始两次组合器拟合和校准阈值计算。旧模型和旧成绩文件全部保留，没有修改旧模块全局变量。

## 时间与结论边界

三折large共2763次更新，接近原六轮的2760次；预计GPU约75–95分钟，含最后留出推理及保存，实际以日志为准。三份最终模型/Adam状态约14.3GB；CPU的3个LR和2个小树预计数分钟。原large峰值约7.2GB，仅作为同精度、同架构的历史资源参考。

这能检验上游训练内分数造成的偏差，但不能完全消除分布差异：留出分数来自410组训练模型，校准分数来自615组训练模型。固定3轮和LB的C也继承了历史开发选择。159条校准已多次使用，不能当作新的独立测试；official test150本轮仍不读取。large属于额外语义核查模型；不称纯生成模型内部探针，也不预先承诺改善。
