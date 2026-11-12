# 六基线共同评测报告

本次比较 6 种独立基线与此前固定的 HGB 融合方法。Lookback 的默认版单独列为补充，不增加基线数量。

回答级最强基线为 RAGognizer（同域重训，NF4）（F1=0.8000）；本文方法 F1=0.8439，高于该基线，差值 +0.0439。 配对 95% 区间：[-0.0121, +0.1107]（区间含 0）。

4 词元窗口级最强基线为 RAGognizer（同域重训，NF4）（F1=0.6240）；本文方法 F1=0.6458，高于该基线，差值 +0.0218。 配对 95% 区间：[-0.0322, +0.0773]（区间含 0）。

差值均为“本文方法减基线”。区间含 0 时，只报告本次分数差，不认定提升已经得到明确统计支持。

## 数据与评测口径

数据来自 F4biian/RAGognize 官方 train 的 Llama-2-7B-chat 输出，按来源组划分为训练 1292 条、验证 271 条、测试 279 条；使用发布的有效自动错误范围。

主测试为原提示没有资料不足暗示的 155 条回答、75 个来源组，共 15696 个有效窗口；全量 279 条、133 组、28977 个窗口作补充。

共同窗口为原始 Llama 分词后的连续 4 个词元，步长 1；排除全无字母数字的窗口，窗口内任一有效错误词元即为正例。回答风险取窗口最高分。P 为精确率，R 为召回率，F1 为两者的综合指标，AUROC 衡量风险分数的排序能力。

Lookback、RAGognizer 复用本轮既有同域训练的冻结预测并重评分；ReDeEP、LUMINA、LettuceDetect、MiniCheck 在本次扩充中实际适配运行。HGB 的检查点、预测和阈值保持固定。测试此前已经解封，本轮属于同一固定划分上的补充比较。

## 主测试：155 条

| 方法 | 回答 P | 回答 R | 回答 F1 | 回答 AUROC | 窗口 P | 窗口 R | 窗口 F1 | 窗口 AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Lookback Lens（验证调参） | 0.8182 | 0.6585 | 0.7297 | 0.8050 | 0.4218 | 0.6833 | 0.5216 | 0.7878 |
| RAGognizer（同域重训，NF4） | 0.7727 | 0.8293 | 0.8000 | 0.8543 | 0.5767 | 0.6798 | 0.6240 | 0.8491 |
| ReDeEP（论文公式，FP16） | 0.5390 | 0.9268 | 0.6816 | 0.5356 | 0.2183 | 0.9988 | 0.3583 | 0.5052 |
| LUMINA（NF4） | 0.5727 | 0.7683 | 0.6562 | 0.6183 | 0.2318 | 0.8817 | 0.3671 | 0.5613 |
| LettuceDetect v1（同域重训） | 0.7396 | 0.8659 | 0.7978 | 0.8567 | 0.5906 | 0.6085 | 0.5994 | 0.8493 |
| MiniCheck-FlanT5（原生句级） | 0.6327 | 0.7561 | 0.6889 | 0.6824 | 0.3642 | 0.5919 | 0.4509 | 0.7407 |
| 本文方法（固定 HGB） | 0.8022 | 0.8902 | 0.8439 | 0.8939 | 0.6113 | 0.6845 | 0.6458 | 0.8636 |

## 全量补充：279 条

| 方法 | 回答 P | 回答 R | 回答 F1 | 回答 AUROC | 窗口 P | 窗口 R | 窗口 F1 | 窗口 AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Lookback Lens（验证调参） | 0.7969 | 0.7183 | 0.7556 | 0.8179 | 0.4186 | 0.6809 | 0.5185 | 0.7802 |
| RAGognizer（同域重训，NF4） | 0.7267 | 0.8239 | 0.7723 | 0.8500 | 0.5861 | 0.6876 | 0.6328 | 0.8501 |
| ReDeEP（论文公式，FP16） | 0.5336 | 0.9507 | 0.6835 | 0.5865 | 0.2211 | 0.9984 | 0.3620 | 0.5045 |
| LUMINA（NF4） | 0.5500 | 0.7746 | 0.6433 | 0.6304 | 0.2301 | 0.8745 | 0.3644 | 0.5378 |
| LettuceDetect v1（同域重训） | 0.7039 | 0.8873 | 0.7850 | 0.8584 | 0.6170 | 0.6172 | 0.6171 | 0.8542 |
| MiniCheck-FlanT5（原生句级） | 0.6331 | 0.7535 | 0.6881 | 0.7049 | 0.3763 | 0.6304 | 0.4713 | 0.7484 |
| 本文方法（固定 HGB） | 0.7651 | 0.8944 | 0.8247 | 0.8860 | 0.6200 | 0.7085 | 0.6613 | 0.8762 |

回答级最强基线为 LettuceDetect v1（同域重训）（F1=0.7850）；本文方法 F1=0.8247，高于该基线，差值 +0.0396。

4 词元窗口级最强基线为 RAGognizer（同域重训，NF4）（F1=0.6328）；本文方法 F1=0.6613，高于该基线，差值 +0.0285。

## 验证选择与阈值

新增方法的 mean4／lexical max4 适配仅在无暗示验证集 135 条上选择，按“两级 F1 的较小值、窗口 F1、窗口精确率”依次比较，并列优先 mean。回答与窗口阈值分别最大化验证 F1，并列依次取精确率、高阈值。表内分数均为风险方向，越高越可疑。

| 方法 | 窗口分数 | 回答阈值 | 窗口阈值 | 验证回答 F1 | 验证窗口 F1 |
|---|---|---:|---:|---:|---:|
| Lookback Lens（验证调参） | 有效词元最大值 | 0.4868948232 | 0.3133082853 | 0.8675 | 0.6536 |
| RAGognizer（同域重训，NF4） | 原始4词元均值 | 0.6340426803 | 0.5685654283 | 0.8772 | 0.7333 |
| ReDeEP（论文公式，FP16） | 有效词元最大值 | 0.3083648077 | 0.0421981065 | 0.7850 | 0.5119 |
| LUMINA（NF4） | 原始4词元均值 | 0.4696516208 | -0.0834896984 | 0.8172 | 0.5162 |
| LettuceDetect v1（同域重训） | 有效词元最大值 | 0.1135795936 | 0.2233770639 | 0.8736 | 0.7368 |
| MiniCheck-FlanT5（原生句级） | 有效词元最大值 | 0.9456357881 | 0.8685122579 | 0.8475 | 0.6452 |
| 本文方法（固定 HGB） | 直接预测共同窗口 | 0.3174249622 | 0.3633343616 | 0.9080 | 0.7575 |

## 与各基线的配对比较

按主测试的 75 个来源组进行 2000 次有放回重采样；同一次抽样对所有方法一致，随机种子为 20260928。下列为 F1 差值及其 95% 百分位区间。

| 基线 | 回答 F1 差 | 回答 95% 区间 | 窗口 F1 差 | 窗口 95% 区间 |
|---|---:|---|---:|---|
| Lookback Lens（验证调参） | +0.1142 | [+0.0446, +0.1859]（区间全高于 0） | +0.1242 | [+0.0508, +0.2004]（区间全高于 0） |
| RAGognizer（同域重训，NF4） | +0.0439 | [-0.0121, +0.1107]（区间含 0） | +0.0218 | [-0.0322, +0.0773]（区间含 0） |
| ReDeEP（论文公式，FP16） | +0.1623 | [+0.0963, +0.2316]（区间全高于 0） | +0.2875 | [+0.2116, +0.3726]（区间全高于 0） |
| LUMINA（NF4） | +0.1877 | [+0.1199, +0.2707]（区间全高于 0） | +0.2787 | [+0.2067, +0.3586]（区间全高于 0） |
| LettuceDetect v1（同域重训） | +0.0462 | [-0.0083, +0.1011]（区间含 0） | +0.0464 | [+0.0052, +0.0958]（区间全高于 0） |
| MiniCheck-FlanT5（原生句级） | +0.1550 | [+0.0890, +0.2274]（区间全高于 0） | +0.1949 | [+0.1190, +0.2794]（区间全高于 0） |

## 方法身份与运行方式

| 方法 | 保留的方法与训练来源 | 精度及原生粒度 |
|---|---|---|
| Lookback Lens | 注意力比率、原生8词元平均、逻辑回归；本轮1292条重新拟合，C只由验证选择 | NF4 Llama回放；原生8词元窗口 |
| RAGognizer | 原LoRA、中间层MLP、语言模型损失与风险损失联合训练；通用Llama初始化，本轮1292条训练5轮 | NF4基座、BF16计算；原生词元 |
| ReDeEP | 使用论文ECS/JSD公式；原始训练词元Pearson选择头/层，MinMax仅fit拟合；固定1头、10层、α=1、β=0.2 | FP16及CPU卸载；原生词元 |
| LUMINA | 保留官方IPR与top100余弦MMD组合；不训练检测器，替换资料仅来自训练集并固定抽取 | NF4、BF16计算、FP32归约；原生词元 |
| LettuceDetect v1 | 保留官方v0.1.6成对输入、两类词元交叉熵及AdamW；通用ModernBERT-large初始化，本轮1292条训练6轮，验证选轮次 | 参数/梯度/优化器FP32，前向BF16；原生ModernBERT词元 |
| MiniCheck | 官方冻结FlanT5-large检查点；保留doc+claim模板、500词资料分块与最大支持概率；仅验证校准阈值 | FP32；原生句级 |
| 本文HGB | 固定既有完整资料分支、白盒分支与17项窗口特征；融合器使用训练集来源组折外预测拟合 | 原结果冻结；共同4词元窗口 |

MiniCheck 的 doc 仅为检索资料，claim 仅为原回答中的一句话；问题不作为证据或待核查断言输入。句级风险为 1−最大支持概率，再按字符对应到共同词元坐标，因此该项仍是句级方法的窗口适配。

ReDeEP 的论文公式与其发布代码有差异，本轮固定采用论文公式；LUMINA 保留已锁定官方代码公式。二者使用共同原始输入，不额外改写问题或截断回答。LettuceDetect/MiniCheck 的新坐标转换保留空白词元的原生重叠或固定最近邻分数，未将其硬设为零。

## 实际耗时

下表逐项注明计时范围，不把共享特征、分支训练和单次打分当作同一种成本。时间来自实际完成记录和逐样本 receipt。

| 方法 | 计时范围 | 耗时 | 说明 | 来源 |
|---|---|---:|---|---|
| Lookback Lens | 既有选中 C 的拟合 | 9.65 秒（0.16 分） | 不含共享 Llama 特征提取及其他 C 候选 | [CONVERGENCE.json](<D:/Projects/Multi_Agent_Graph_Analysis/intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/baseline_audit/lookback_run_v2/tuned/CONVERGENCE.json>) |
| Lookback Lens | 既有 279 条测试打分 | 4.88 秒（0.08 分） | 使用冻结特征缓存 | [PREDICTION_RECEIPT.json](<D:/Projects/Multi_Agent_Graph_Analysis/intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/baseline_audit/predictions/lookback_tuned_test/PREDICTION_RECEIPT.json>) |
| RAGognizer | 既有 5 轮训练流程 | 10713.80 秒（178.56 分） | 使用其训练完成记录 | [COMPLETE.json](<D:/Projects/Multi_Agent_Graph_Analysis/intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/baseline_audit/ragognizer_run_v1/COMPLETE.json>) |
| RAGognizer | 既有 279 条测试预测 | 153.40 秒（2.56 分） |  | [PREDICTION_RECEIPT.json](<D:/Projects/Multi_Agent_Graph_Analysis/intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/baseline_audit/ragognizer_test_predictions_v1/PREDICTION_RECEIPT.json>) |
| ReDeEP | 1842 条特征提取进程记录 | 3788.90 秒（63.15 分） | 该进程记录范围；含载入开销，断点续跑时不等同全程 | [EXTRACTION_COMPLETE.json](<D:/Projects/Multi_Agent_Graph_Analysis/intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/baseline_extension_20260928/whitebox/redeep/EXTRACTION_COMPLETE.json>) |
| ReDeEP | 1842 条逐样本计算合计 | 3746.44 秒（62.44 分） | 逐条 receipt 累加；不含模型载入/样本间调度 | [features](<D:/Projects/Multi_Agent_Graph_Analysis/intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/baseline_extension_20260928/whitebox/redeep/features>) |
| LUMINA | 550 条特征提取进程记录 | 588.57 秒（9.81 分） | 该进程记录范围；含载入开销，断点续跑时不等同全程 | [EXTRACTION_COMPLETE.json](<D:/Projects/Multi_Agent_Graph_Analysis/intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/baseline_extension_20260928/whitebox/lumina/EXTRACTION_COMPLETE.json>) |
| LUMINA | 550 条逐样本计算合计 | 566.98 秒（9.45 分） | 逐条 receipt 累加；不含模型载入/样本间调度 | [features](<D:/Projects/Multi_Agent_Graph_Analysis/intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/baseline_extension_20260928/whitebox/lumina/features>) |
| LettuceDetect | 6 轮训练流程 | 1998.96 秒（33.32 分） | 含各轮验证与检查点保存 | [TRAIN_COMPLETE.json](<D:/Projects/Multi_Agent_Graph_Analysis/intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/baseline_extension_20260928/semantic/lettuce_v1_run/TRAIN_COMPLETE.json>) |
| LettuceDetect | 271 条验证预测 | 未单列 | 按实际 receipt；未记录的时段不估算 | [validation_PREDICTION_RECEIPT.json](<D:/Projects/Multi_Agent_Graph_Analysis/intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/baseline_extension_20260928/semantic/lettuce_v1_run/validation_PREDICTION_RECEIPT.json>) |
| LettuceDetect | 279 条测试预测 | 未单列 | 按实际 receipt；未记录的时段不估算 | [test_PREDICTION_RECEIPT.json](<D:/Projects/Multi_Agent_Graph_Analysis/intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/baseline_extension_20260928/semantic/lettuce_v1_run/test_PREDICTION_RECEIPT.json>) |
| MiniCheck | 271 条验证预测 | 94.33 秒（1.57 分） | 按实际 receipt；未记录的时段不估算 | [validation_PREDICTION_RECEIPT.json](<D:/Projects/Multi_Agent_Graph_Analysis/intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/baseline_extension_20260928/semantic/minicheck_run/validation_PREDICTION_RECEIPT.json>) |
| MiniCheck | 279 条测试预测 | 85.77 秒（1.43 分） | 按实际 receipt；未记录的时段不估算 | [test_PREDICTION_RECEIPT.json](<D:/Projects/Multi_Agent_Graph_Analysis/intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/baseline_extension_20260928/semantic/minicheck_run/test_PREDICTION_RECEIPT.json>) |
| 本文 HGB | 既有 4 候选融合训练/选择 | 65.52 秒（1.09 分） | 仅融合器，非语义/白盒分支总训练时长 | [COMPLETE.json](<D:/Projects/Multi_Agent_Graph_Analysis/intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/fusion/nonlinear_window_v1/COMPLETE.json>) |

## 可复核文件

审计状态：PASS。共同指标、原预测哈希、逐方法输入/坐标审计及报告来源索引均保留。

| 方法 | 验证预测 | 测试预测 |
|---|---|---|
| Lookback Lens（验证调参） | [完整验证JSONL](<D:/Projects/Multi_Agent_Graph_Analysis/intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/baseline_audit/predictions/lookback_tuned_validation/predictions.jsonl>) | [完整测试JSONL](<D:/Projects/Multi_Agent_Graph_Analysis/intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/baseline_audit/predictions/lookback_tuned_test/predictions.jsonl>) |
| RAGognizer（同域重训，NF4） | [完整验证JSONL](<D:/Projects/Multi_Agent_Graph_Analysis/intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/baseline_audit/ragognizer_validation_predictions_v1/predictions.jsonl>) | [完整测试JSONL](<D:/Projects/Multi_Agent_Graph_Analysis/intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/baseline_audit/ragognizer_test_predictions_v1/predictions.jsonl>) |
| ReDeEP（论文公式，FP16） | [完整验证JSONL](<D:/Projects/Multi_Agent_Graph_Analysis/intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/baseline_extension_20260928/whitebox/redeep/validation_predictions.jsonl>) | [完整测试JSONL](<D:/Projects/Multi_Agent_Graph_Analysis/intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/baseline_extension_20260928/whitebox/redeep/test_predictions.jsonl>) |
| LUMINA（NF4） | [完整验证JSONL](<D:/Projects/Multi_Agent_Graph_Analysis/intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/baseline_extension_20260928/whitebox/lumina/validation_predictions.jsonl>) | [完整测试JSONL](<D:/Projects/Multi_Agent_Graph_Analysis/intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/baseline_extension_20260928/whitebox/lumina/test_predictions.jsonl>) |
| LettuceDetect v1（同域重训） | [完整验证JSONL](<D:/Projects/Multi_Agent_Graph_Analysis/intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/baseline_extension_20260928/semantic/lettuce_v1_run/validation_predictions.jsonl>) | [完整测试JSONL](<D:/Projects/Multi_Agent_Graph_Analysis/intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/baseline_extension_20260928/semantic/lettuce_v1_run/test_predictions.jsonl>) |
| MiniCheck-FlanT5（原生句级） | [完整验证JSONL](<D:/Projects/Multi_Agent_Graph_Analysis/intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/baseline_extension_20260928/semantic/minicheck_run/validation_predictions.jsonl>) | [完整测试JSONL](<D:/Projects/Multi_Agent_Graph_Analysis/intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/baseline_extension_20260928/semantic/minicheck_run/test_predictions.jsonl>) |
| 本文方法（固定 HGB） | [完整验证JSONL](<D:/Projects/Multi_Agent_Graph_Analysis/intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/fusion/nonlinear_window_v1/best_validation_predictions.jsonl>) | [完整测试JSONL](<D:/Projects/Multi_Agent_Graph_Analysis/intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/fusion/nonlinear_window_v1/test_predictions.jsonl>) |

- 完整共同结果：[SIX_BASELINES_RESULTS.json](<D:/Projects/Multi_Agent_Graph_Analysis/intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/baseline_extension_20260928/audit/SIX_BASELINES_RESULTS.json>)。
- 统一审计：[SIX_BASELINES_AUDIT.json](<D:/Projects/Multi_Agent_Graph_Analysis/intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/baseline_extension_20260928/audit/SIX_BASELINES_AUDIT.json>)。
- 冻结模型/预测/阈值及完整哈希：[EXTENSION_FROZEN.json](<D:/Projects/Multi_Agent_Graph_Analysis/intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/baseline_extension_20260928/audit/EXTENSION_FROZEN.json>)。
- 原生分数审计：[WHITEBOX_INPUT_AUDIT.json](<D:/Projects/Multi_Agent_Graph_Analysis/intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/baseline_extension_20260928/audit/WHITEBOX_INPUT_AUDIT.json>)、[LETTUCE_INPUT_AUDIT.json](<D:/Projects/Multi_Agent_Graph_Analysis/intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/baseline_extension_20260928/audit/LETTUCE_INPUT_AUDIT.json>)、[MINICHECK_INPUT_AUDIT.json](<D:/Projects/Multi_Agent_Graph_Analysis/intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/baseline_extension_20260928/audit/MINICHECK_INPUT_AUDIT.json>)。
- 完整来源路径、SHA-256与计时依据：[REPORT_SOURCES.json](<D:/Projects/Multi_Agent_Graph_Analysis/intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/baseline_extension_20260928/REPORT_SOURCES.json>)。

| 关键脚本/配置 | SHA-256 |
|---|---|
| [generate_report.py](<D:/Projects/Multi_Agent_Graph_Analysis/intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/baseline_extension_20260928/audit/generate_report.py>) | `e5131f623a7a52510a36fa334b233809bc0250dd5cd377a57f55da258d291a47` |
| [evaluate_extensions.py](<D:/Projects/Multi_Agent_Graph_Analysis/intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/baseline_extension_20260928/audit/evaluate_extensions.py>) | `0430b3a34f919b902c234ae8c0b430849d3a59b32adb8e4333a5e76ed3fb7896` |
| [new_methods.json](<D:/Projects/Multi_Agent_Graph_Analysis/intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/baseline_extension_20260928/audit/new_methods.json>) | `778860fc038b4855eb32ccc895d76f285a85916d816fe440aac614a568d56745` |
| [run_whitebox.py](<D:/Projects/Multi_Agent_Graph_Analysis/intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/baseline_extension_20260928/whitebox/run_whitebox.py>) | `c4b707d02f3db60a7a5f76838c3f0743aa372601b9ac8bffc13bdd0b78d7ab91` |
| [CONFIG.json](<D:/Projects/Multi_Agent_Graph_Analysis/intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/baseline_extension_20260928/whitebox/CONFIG.json>) | `e2f5525fc403005ccfcb1dbd7a9795f13b8c004927af1bcfebd712fd7ae8203b` |
| [lettuce_v1.py](<D:/Projects/Multi_Agent_Graph_Analysis/intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/baseline_extension_20260928/semantic/lettuce_v1.py>) | `e8adc46d30b42714c7d2c5fa2ce4dae65fa2427a9a28dbc69cf014590b0e2ea6` |
| [minicheck_official.py](<D:/Projects/Multi_Agent_Graph_Analysis/intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/baseline_extension_20260928/semantic/minicheck_official.py>) | `20e9ce301610662ba44114ce51764e5b85999dbe6a9ef1509d12dbbb9312989b` |
| [common.py](<D:/Projects/Multi_Agent_Graph_Analysis/intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/baseline_extension_20260928/semantic/common.py>) | `743bc6358fde0b2389df734ec88f10d93b47cbb56078628efe79531cc6feb1f6` |

补充：Lookback默认参数版本的主测试回答F1为 0.6849，窗口F1为 0.5027；该版本不计入6种独立基线。
