# 最终开发集验收报告

状态：`PASS_DEVELOPMENT_ONLY`

## 1. 任务与方法

任务是在“资料＋问题＋模型回答”的联网情报汇总场景中，判断回答是否含有无依据或与资料冲突的事实，并定位到较粗的文本区域。

当前候选为 `semantic_claim__old_tree__large_weight0.4`，由两条风险支路组成：

1. 白盒与语义支路：把生成时的 Lookback 注意力、token NLL、MiniCheck 语义状态和陈述风险送入已训练 TCN，并做同陈述传播；再与 MiniCheck 末两层风险头通过深度 2 单调树合成为 `p_old`。该支路的树使用原 634 条人工 fit 回答训练。
2. 全上下文支路：ModernBERT-large 同时读取资料、问题和完整回答，输出窗口风险 `p_large`；使用扩充后的 3,680 条 fit 回答训练，冻结第 3 轮模型。
3. 最终窗口分数固定为 `0.6 × p_old + 0.4 × p_large`；整答分数取该回答所有可评窗口的最大值。

因此，这个候选是“生成模型内部信号＋证据语义核查”的离线融合检测器，不应表述为纯白盒探针或严格实时逐 token 检测器。

## 2. 统一评测口径

- 数据：共同 fit 为 3,680 答/615 个材料组，calibration 为 159 答/154 个材料组，材料组零重叠。候选旧支路只使用原 634 条人工 fit 回答；large 支路和正式基线按各自冻结协议使用共同 fit。
- 定位单位：4 个原始 BPE、步长 1；只评估映射字符含字母或数字的窗口。calibration 共 42,241 个窗口，其中风险窗口 5,984 个。
- 整答标签：回答中只要存在人工事实错误区间即为风险；calibration 共 100/159 个风险回答。
- 整答聚合：取本回答所有可评窗口的最大风险分数。
- 阈值：窗口和整答分别在 calibration 上按 `F1 最大 → precision 更高 → 阈值更高` 选择，判定规则为 `score >= threshold`。
- 指标：统一报告 precision、recall、F1、AUROC、AP；F1、AUROC、AP 均为越高越好。
- 基线：保留作者模型结构、特征、训练配方和原生分数，只允许模型外、无参数、标签无关的坐标映射。推理型基线正式入表前须完成全部 3,839 答覆盖。

## 3. 当前候选的完整 calibration 指标

| 粒度 | N / 正类 | TP / FP / FN / TN | Precision | Recall | F1 | AUROC | AP | 阈值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4-BPE 窗口 | 42,241 / 5,984 | 3,839 / 1,300 / 2,145 / 34,957 | 0.747032 | 0.641544 | **0.690281** | 0.913367 | 0.721406 | 0.657470 |
| 整答 | 159 / 100 | 90 / 12 / 10 / 47 | 0.882353 | 0.900000 | **0.891089** | 0.906102 | 0.946840 | 0.456674 |

按当前验收目标，整答约 0.9、窗口约 0.7，候选达到预实验目标区间。

## 4. 正式基线比较

每格为 `AUROC / AP / F1`。

| 方法 | 状态 | 统一 4-BPE 窗口 | 统一整答 |
|---|---|---:|---:|
| 当前候选 | 已冻结并复算 | **0.913367 / 0.721406 / 0.690281** | **0.906102 / 0.946840 / 0.891089** |
| Lookback Lens | 完成并独立复算 | 0.872400 / 0.638061 / 0.600882 | 0.848814 / 0.906297 / 0.845455 |
| GHOST | 完成并独立复算 | 0.632713 / 0.239092 / 0.320361 | 0.588983 / 0.734942 / 0.772201 |
| LUMINA | 完成并独立复算 | 0.691406 / 0.270062 / 0.331299 | 0.702203 / 0.790876 / 0.785047 |
| ReDeEP | 完成、身份绑定并独立复算 | 0.669274 / 0.257497 / 0.329560 | 0.590678 / 0.732595 / 0.772358 |
| RAGognizer | 全 3,839 答完成，独立审计 PASS | 0.824938 / 0.367650 / 0.507511 | 0.706949 / 0.745699 / 0.806867 |

RAGognizer 正式分数与先前 provisional 逐值一致。其正式产物清单 SHA256 为 `44ee9aae88dc5351219730a5595f19a3b8435b6cde37813bd53e2d12787d3e16`。

## 5. 机械验收结果

验收只比较两个预先约定的 F1 指标：

| 检查项 | 当前候选 | 最高正式基线 | 最高基线 | 差值 | 判定 |
|---|---:|---:|---|---:|---|
| 4-BPE 窗口 F1 | 0.690281399 | 0.600882124 | Lookback Lens | +0.089399275 | **通过** |
| 整答 F1 | 0.891089109 | 0.845454545 | Lookback Lens | +0.045634563 | **通过** |

两项都不低于全部五个已完成正式基线，因此当前候选的**开发集验收通过**。候选的窗口与整答 AUROC/AP 也均高于表中正式基线，但它们不参与本次预注册验收判定。

## 6. 结论边界

- calibration 159 已被多轮模型、权重和阈值开发反复使用，当前数字属于开发集成绩，不能写成独立测试或稳定泛化结论。
- 旧 QA test 已退出最终无偏评测；新的独立 holdout 尚未构造并盲测。因此，本次可以写“统一开发评测领先正式基线”，尚不能写“独立测试领先”或“SOTA”。
- 4-BPE 滑窗表示粗粒度风险区域，不等于精确指出每个错误字符。

## 7. 证据入口

- 候选定义与限制：[`large_fixed_convex_v1/CANDIDATE.md`](../../results/large_fixed_convex_v1/CANDIDATE.md)
- 候选完整数值：[`current_span_position_diagnosis_v1/DIAGNOSIS.json`](../../results/current_span_position_diagnosis_v1/DIAGNOSIS.json)
- 正式基线总表：[`FORMAL_BASELINE_RESULTS.md`](../../FORMAL_BASELINE_RESULTS.md)
- RAGognizer 正式结果：[`ragognizer_formal_baseline_v1/CAL_RESULTS.json`](../ragognizer_formal_baseline_v1/CAL_RESULTS.json)
- RAGognizer 独立审计：[`ragognizer_formal_baseline_v1/SCORE_INDEPENDENT_AUDIT.json`](../ragognizer_formal_baseline_v1/SCORE_INDEPENDENT_AUDIT.json)
- 统一迁移规则：[`BASELINE_PROTOCOL.md`](../../BASELINE_PROTOCOL.md)
- 新 holdout 协议：[`replacement_holdout_v1/BLIND_HOLDOUT_PROTOCOL.md`](../replacement_holdout_v1/BLIND_HOLDOUT_PROTOCOL.md)

