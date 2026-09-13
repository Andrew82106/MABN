# Round 7 正式评测前检查

范围：只审核代码、预定协议和运行清单；未运行正式 fit/test，未读取测试分数。按根任务追加授权，仅修改报告耗时措辞并新增测试后计时脚本。测试标签的最终裁决由根任务负责。

## 已覆盖

- 12 类方法全部接入：线性探针、三种固定种子的 HalluRAG 风格 MLP、Lookback Lens、ReDeEP、LUMINA、NLL、熵、表面特征、自评、直接核查及两个常数对照，共 14 个独立预测器。
- `fit` 只读取 train/validation 标签。标准化、类别权重、ReDeEP 排序和缩放仅拟合训练集；参数和阈值由验证集选择。MLP 共同参数按三种子平均验证 F1 选择，保留每种子阈值与测试指标，不挑最佳测试种子、不平均概率另造方法。
- 两个留出集由 `split=test/external_test` 区分，须同时齐备。模型、参数、开发标签、全部输入/特征及评测代码哈希在测试前冻结；标签精确文本、范围和生成文件哈希逐项核对。测试开始后保留标签哈希，统一完成后才写两套结果。
- 回答项 F1/P/R、混淆计数、误报率、AP/AUROC、条件分表、风险子类、同回答内排序、全部预定项的前 20% 复查均已实现。缺预测的已知风险计漏检；AP/AUROC 可评分分母另报。旧的金标过滤复查指标明确是次要诊断。
- 主测试和外部测试分别报告，组 bootstrap 共同抽取同题的两个条件及各回答项。区间以冻结模型为条件，不包括重新训练或重新标注的不确定性。

## 具体缺口及处理

1. `supported_in_both_conditions` 选择“两份实际输出均被标为有依据”的项，不能等同于“来源删除设计中未受影响的主体”。根任务已另派 `supplement7.py` 按删除主体定义补充分析。正式报告应保留这两个不同名称，并核对补充输出。
2. 自动 `TABLES.md` 尚不展开各方法的条件表现、风险子类、多事实数量、表面特征解释和成本。JSON 已有前几类统计；根任务的最终 REPORT 需要实际写出这些分析，不能仅用 JSON 链接替代结论。
3. `fit_seconds` 是本次 fit 调用墙钟，含载入、校验、参数选择和验证；恢复缓存时不含此前调用。已修改 `report7.py` 的“累计训练”措辞，不改变任何评测计算。
4. 新增 `benchmark7.py`，只有存在有效 `test_complete.json` 才可运行。读取冻结模型及两个留出集的特征；不读标签或测试指标，不训练，不改阈值。载入、矩阵准备单独记录；每方法一次预热、五次 CPU 全批评分，保存各次时间、中位数、项数、有限分数数及每项摊销时间。脚本尚未执行，仅通过 AST 语法检查。

测试后执行：

```powershell
prelab\.venv\Scripts\python.exe prelab\round7_evidence_grounding\src\benchmark7.py
```

该脚本输出 `results/cpu_scoring_benchmark.json`。纯 CPU 评分排除生成、重放、读取磁盘、组装矩阵、阈值判定与评测；原始信号对照的 CPU 操作主要是读取已算好的数值，不能据此宣称自评或 LUMINA 几乎零成本。五次全批时间除以项数是摊销成本，不是单项在线延迟。当前评测器在每次 MLP 评分时重建小网络并复制已载入的 state_dict；基准保留并显式记录这项实际评分开销，不能叫纯前向时间。

## 实际运行成本从哪里取

| 环节 | 记录来源 | 解释 |
|---|---|---|
| 原始生成 | `data/generation_records/*.json` 的 seconds、peak_allocated_gib、token 数 | 按 split/condition 汇总；generation_manifest 本身没有时间总和 |
| 隐状态/NLL/熵、注意力、LUMINA | 各阶段逐行 JSON 及 manifest | stage seconds 求和；峰值取最大。注意力另存两次前向时间，LUMINA 另存原资料与替换资料时间 |
| 自评与直接核查 | baselines 逐行 items 的 self_confidence_seconds、direct_seconds、输入/输出 token 数 | 两种独立请求分别计成本；外层 stage 时间还含循环与准备开销，不能与内部时间重复相加 |
| CPU 拟合 | freeze.fit_seconds；MLP 候选缓存训练信息中的 seconds | 整次调用时间和单候选训练时间分开解释，不能将二者相加 |
| CPU 评分 | 新增测试后 benchmark 输出 | 模型/特征载入与已备好矩阵的评分分开，不读标签 |

源码检查时的 manifest 快照：features 500/500、113.25 秒；attention 500/500、829.02 秒；LUMINA 与 baselines 清单仍为 100/500，尚未完整。清单可能只在阶段结束更新，因此这不是 GPU 当前逐行进度，也不是最终总成本。正式 fit 的清单完整性检查会拒绝不齐备方法。

GPU 峰值字段来自 PyTorch allocated memory，不能叫整卡占用或 reserved memory；模型载入发生在逐行计时之外。CPU prepared_matrix_bytes 也不是进程峰值内存。所有阶段之和排除模型初始化、部分磁盘写入和任务等待，须称“已记录的计算阶段耗时”，不能当完整运行墙钟。

## REPORT 必须披露的适配和范围

- 目标是模型可见来源是否支持实际断言，与答案是否符合完整参考分开。无依据但碰巧正确仍可为风险；明确虚假的“主体未出现”等附加资料断言也在本轮标签范围内。
- 本轮是固定检索资料的回答项监测，英文百科事实为主；尚未复现自主联网搜集、动态来源真伪判断、多轮研判或逐词定位。外部每份回答仅一项，无法估计同回答内部排序。
- 标签是助手辅助标注与助手复核，不能称独立研究者人工金标准。未决、拒答、缺项、混合多事实和解析失败须公开；不能为了达到 0.70 删除困难项。
- Qwen2.5-7B-Instruct 使用 NF4 量化及固定生成设置；不能外推至所有开源模型。内部状态为回答项最后一个内容 token 被读入后的状态，第 28 层归一化位置不同于前几层。
- HalluRAG 是隐藏状态分类器适配，未复现其全部 FFN 信号；Lookback 是 Qwen 重新训练、按项聚合、使用本轮资料边界和后读位置的适配。
- ReDeEP 改用标准正向 JSD，Qwen/GQA 信号、训练集 Pearson 排序与 MinMax、288 组验证候选及按项聚合，均须链接 `references/redeep_adaptation.md`，不称原论文完全复现。
- LUMINA 固定官方代码版本和 λ=0.5、top-k=100；保持自身生成 token，在固定无关资料上重放。层范围及最终归一化依固定代码，和论文公式并非完全一致；本轮没有官方额外空格、重新分词或字符截断，详见逐行 metadata。
- 注意力两次重放和 LUMINA 资料替换前向有实际成本；同一 Qwen 自评/直接核查不是独立裁判。所有风险分数均不能自动解释为校准后的真实概率。
- 只有最终冻结结果能判断是否达到 0.70；需要与熵等简单强对照及实际复查收益一并比较，不能只挑某一 F1 或某一种子。

## 审核版本

`evaluate7.py`: `ed2461799e9fdf0e0d5bc2c6313bd512540b8c2645122328939061b893aa4c2e`  
`report7.py`: `ec39dd0d9ddbe17aac735250dd2134c6efcf9cea8b9abee8a63a43b825c9849f`  
`benchmark7.py`: `9e2f49148238f5162378740ebb72975d3f67ff161bb78cab734d5aeedc2c7654`  
`protocol.json`: `3b9c54cf5ee0f67f1cce7b96832e1453b49f631b12f9975d8df703ed3e1ff20a`  
`PLAN.md`: `4d1d5adb32122caf2a200b32a3ca839ae98f49bbb5ecfb7fe5fb31931db9133b`
