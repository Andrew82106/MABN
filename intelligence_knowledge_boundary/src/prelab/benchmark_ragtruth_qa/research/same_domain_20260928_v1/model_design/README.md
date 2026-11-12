# RAGognize 同域完整上下文分支

入口为 `train_full_context.py`，通用 ModernBERT-base 初始化，149,606,402 个参数。全部资料、问题、回答保留独立分隔符并联合编码；风险 logit 按非空白字符重合映回原始 Llama BPE。训练只用 fit，组、组内回答、回答内 lexical token 逐级等权 BCE。

当前统一划分来自 `../data`：fit 1292 条/617 组，validation 271 条/133 组；本目录不读取 test。完整输入最长 1201 个 ModernBERT token，无截断。CUDA BF16 前向和梯度检查点重算；参数、映射、损失、梯度和 AdamW 状态为 FP32。CPU 检查与最长输入 GPU 试跑均已通过。

共享 `../metrics.py` 用 validation no_hint 队列选两级阈值，报告 all/no_hint；最多6轮，按 no_hint 两级 F1 最小值、窗口 F1、窗口精确率、较早轮次选最佳。窗口为4个原始 BPE、步长1，最大 lexical 风险；整答为窗口最大值。

```powershell
$python = 'D:\Projects\Multi_Agent_Graph_Analysis\intelligence_knowledge_boundary\src\prelab\.venv\Scripts\python.exe'
$runner = 'D:\Projects\Multi_Agent_Graph_Analysis\intelligence_knowledge_boundary\src\prelab\benchmark_ragtruth_qa\research\same_domain_20260928_v1\model_design\train_full_context.py'
& $python -u -B -X utf8 $runner prepare
& $python -u -B -X utf8 $runner cpu-test
& $python -u -B -X utf8 $runner gpu-smoke
& $python -u -B -X utf8 $runner train
& $python -u -B -X utf8 $runner predict --split fit
```

以上为分开执行的入口，现有训练不可重复启动。输出在 `modernbert_base_v1`：`best.pt`、`best.json`、每轮验证预测/指标及完整训练日志。预测包含原始 token、窗口和整答分数。

6轮已完成，训练进程正常退出。按预定验证规则选择第6轮：no_hint 回答F1 0.834225、窗口F1 0.608540；全validation为0.788889、0.522565。训练及逐轮验证共690.64秒。这不是测试成绩。

诚实融合使用新入口 `train_crossfit.py`。三个目录 `modernbert_oof_fold_0/1/2` 已按根任务的折名单完成CPU准备与检查：训练/留出回答分别859/433、844/448、881/411。每折从通用权重重新初始化，固定训练6轮，仅末轮输出 `held_fit_predictions.jsonl`；不读取validation、不使用held目标，不按held成绩选模型。只在GPU重新排程后执行：

```powershell
& $python -u -B -X utf8 <model_design/train_crossfit.py> train --out <model_design/modernbert_oof_fold_0>
```

测试推理单独由 `predict_full_context.py` 处理。`modernbert_base_test_v1` 已完成279条无标签测试输入的CPU编码与检查，最长1097词元；其prepare仅读取test的inputs/geometry与split_assignments，不接触sealed标签。待GPU排程后运行 `predict_full_context.py predict`；此入口只生成预测，不评分。

## 现有资源盘点

- `auxiliary_ragognize_train_v1/conversion_v1/all_train` 已有1842条官方train的文本、Llama分词与字符/窗口标签；本次逐文件SHA均与原manifest吻合。917个旧组经本轮资料连接合并为883个新组，以新 `../data` 为准。
- 此前没有已完成的1842条 RAGognize Lookback/NLL/hidden 或 MiniCheck 特征缓存。旧no-hint971回放仅有输入计划，未提取特征。
- `full_context_aux_transfer_v1` 是其他人工辅助9678条→RAGTruth3680条，不是RAGognize域内训练。其旧checkpoint未加载。
- 旧 ModernBERT-large、MiniCheck tail2、aligned/atomic/router 结果均来自RAGTruth。可复用实现，但不能把旧模型或缓存当本轮同划分结果。atomic/router未显示稳定收益，本轮优先完整上下文分支与新白盒分支，不先扩展复杂路由。

需要新增的有监督融合应使用共享fit的折外预测，或另留互不重叠的融合训练组；不要在基模型训练内分数上拟合融合树。问题作为条件输入，不作为支持回答的证据。
