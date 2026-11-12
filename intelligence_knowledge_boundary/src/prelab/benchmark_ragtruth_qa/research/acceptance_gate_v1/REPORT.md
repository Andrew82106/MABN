# 独立验收门禁 v1

审计时间：2026-09-13。当前结论：**INCOMPLETE**。本文候选的数值、统一粒度和 fit/cal 切分均复算通过；已完成的三条正式基线也通过。ReDeEP 只有待独立终审的暂定结果，RAGognizer 尚无正式结果，所以目前不能签发最终 PASS。

本审计只打开明确列出的 fit、calibration 和结果文件，没有读取任何 test、holdout 或 sealed-test 内容，没有运行 GPU，也没有修改主结果表或基线实现。

## 1. 当前候选到底是什么

固定候选为 `semantic_claim__old_tree__large_weight0.4`：

1. `p_old` 是原先训练好的深度 2 单调树输出。树的两个输入是 `semantic_claim` 窗口风险与 `tail2` 窗口风险；树只在原始 634 条 fit 回答上拟合。上游分数对这些 fit 行不是完整交叉拟合，因此存在堆叠过拟合风险，但没有使用 calibration 行训练树。
2. `p_large` 是读取完整资料、问题和回答的 ModernBERT-large 窗口风险。它在扩充后的 3,680 条 fit 回答上训练；这 3,680 条仍只来自 634 个来源、615 个材料组。
3. 最终窗口分数固定为 `0.6*p_old + 0.4*p_large`；整答分数是该回答所有合格 4-BPE 窗口分数的最大值。
4. calibration 上分别选择窗口阈值 `0.657469850500832` 和整答阈值 `0.4566737821091461`。

独立脚本从冻结 NPZ、原始窗口标签和回答标签重算得到：

| 粒度 | 样本数 | 正类 | TP/FP/FN/TN | F1 |
|---|---:|---:|---:|---:|
| 4-BPE 窗口 | 42,241 | 5,984 | 3,839 / 1,300 / 2,145 / 34,957 | **0.6902813989** |
| 整答 | 159 | 100 | 90 / 12 / 10 / 47 | **0.8910891089** |

冻结阈值、F1Opt、混淆计数和保存结果完全一致；每个保存的整答分数与对应窗口最大值逐值一致，最大绝对误差为 0。候选 JSON、分数 NPZ 和既有独立审计的 SHA256 也全部匹配。

因此，“整答约 0.9、定位约 0.7”这个开发结果本身可信。这里的“定位”严格指 4 个原始 BPE、步长 1 的窗口，不是单字符定位。

## 2. 切分与泄漏检查

机械复算结果：

| 检查 | 结果 |
|---|---:|
| 扩充 fit 回答 / 材料组 | 3,680 / 615 |
| calibration 回答 / 材料组 | 159 / 154 |
| fit/cal 回答 ID 重叠 | 0 |
| fit/cal 来源 ID 重叠 | 0 |
| fit/cal 材料组 ID 重叠 | 0 |
| 原始 fit / calibration 4-BPE 窗口 | 168,123 / 42,241 |

所以没有发现训练行、来源或材料组直接跨到 calibration 的**切分泄漏**。

但 calibration 已经承担了多重开发用途：ModernBERT-large 的 6 个训练轮次用它选了 epoch 3；最终融合在 6 个固定权重中用它选了 0.4；多个底座家族也在它上面比较；最后两个阈值仍在同一批标签上取 F1 最优，而且此后还有大量实验反复查看这 159 条结果。这属于**验证集反复选型**。它没有进入训练梯度，但会使 0.6903/0.8911 的点估计偏乐观，不能称为独立测试成绩、统计不劣或 SOTA。

另外，旧树使用上游模型在同一 fit 上的 in-sample 分数，fit 窗口 F1 0.8082 到 calibration 0.6903 的落差也显示过拟合。这是训练设计风险，不是 fit/cal 行重叠。

候选自己的记录声明没有打开旧 official test；不过该 test 后来在另一项 ReDeEP 结构审计中被载入，已经永久退出最终无偏测试。最终论文结论仍需在模型、映射和阈值全部冻结后，使用新建且此前未触碰的 holdout 一次评测。

## 3. 正式基线文件的当前状态

`BASELINE_PROTOCOL.md` 的核心结构正确：项目统一数据、4-BPE 窗口、`answer=max(window)`、人工标签、F1Opt 规则和 F1/AUROC/AP；基线保留作者方法层，只允许冻结输出之后做无参数、无标签的坐标映射。

`FORMAL_BASELINE_RESULTS.md` 当前已登记并独立复算的只有：

| 原版方法迁移 | 4-BPE 窗口 F1 | 整答 F1 |
|---|---:|---:|
| Lookback Lens | 0.600882 | 0.845455 |
| GHOST | 0.320361 | 0.772201 |
| LUMINA | 0.331299 | 0.785047 |

`FORMAL_BASELINE_FREEZE.md` 仍是旧登记：ReDeEP 被描述为未完成身份，且没有 RAGognizer 条目。因此公共 `RESULTS/FREEZE` 现在尚未反映正在生成的新产物。

ReDeEP 现有暂定主身份是**论文公式**，其冻结结果为窗口 `0.3295598937`、整答 `0.7723577236`。全 3,839 答原生特征、score freeze 和四例逐值重放身份检查已经存在；但按根任务要求，在独立终审和公共登记完成前，本门禁仍把它列为 provisional。官方源码公式的 `0.332532/0.772201` 只作诊断，不允许按结果替换论文主身份。

RAGognizer 必须使用官方 Llama-2 LoRA 加集成 `hallu_head_neg_16` 的 BF16 路线。独立 MLP 路线不是同一模型，不能参与择优。它仍需完成全 3,839 答推理、无标签适配冻结、cal159 统一评分、独立复算和公共登记。

RAGLens、RefChecker、MVA 等仍为有说明的 N/A，不进入数值最大值。因此最终文字只能说“高于本项目成功运行且可比较的正式基线”，不能扩大成“高于所有已有方法”。

## 4. 最终机械判定

先做资格门禁。每条计分基线必须同时满足：

- 作者主方法身份、checkpoint、源码和参数已冻结；
- 可训练方法只用共同 fit，推理型方法按协议冻结全 3,839 答输出；
- 4-BPE 映射无参数、无标签，且分数在打开 calibration gold 前冻结；
- calibration 分母精确为 42,241 窗口、159 整答，正类为 5,984/100；
- 结果、哈希链和独立复算通过，并登记到 `FORMAL_BASELINE_RESULTS.md` 与 `FORMAL_BASELINE_FREEZE.md`。

资格通过后，分别计算两个包络，而不是先挑一个对我们有利的“综合最强”基线：

```text
B_window = max(每条合格原版基线的 4-BPE 窗口 F1)
B_answer = max(每条合格原版基线的整答 F1)

PASS 当且仅当：
ours_window + 1e-12 >= B_window
且 ours_answer + 1e-12 >= B_answer
```

两个最大值可以来自不同基线；相等算“不低于”。ReDeEP 官方代码诊断、RAGognizer 独立 MLP、作者原生非统一粒度指标，以及任何加入本地融合/平滑/新头的版本都不进入包络。

只看目前三条合格基线，两个最大值都来自 Lookback Lens。本文候选的窗口优势为 `+0.089399`，整答优势为 `+0.045635`。ReDeEP 暂定值更低，不会改变包络；现在唯一未得到数值答案的是 RAGognizer。由于两条待办基线尚未全部完成资格门禁，当前状态严格保持 **INCOMPLETE**。

## 5. 可执行门禁

只读脚本是 `check_acceptance.py`，当前输入快照是 `acceptance_inputs.current.json`。脚本只允许 benchmark 根目录内明确登记的文件，并显式拒绝路径分量为 test/holdout 的输入。它会复算候选切分、`answer=max(window)`、阈值和 F1，再逐项验证基线哈希与共同分母。

当前运行命令：

```powershell
.\prelab\.venv\Scripts\python.exe prelab\benchmark_ragtruth_qa\research\acceptance_gate_v1\check_acceptance.py --allow-incomplete
```

结果已保存为 `CURRENT_RUN.json`，状态为 `INCOMPLETE`，候选审计为 PASS，未决项精确为 ReDeEP 和 RAGognizer。两者终审后，只需把最终结果路径、JSON pointer、哈希和 `independent_verification=true` 写入输入快照，再去掉 `--allow-incomplete` 运行：退出码 0 为 PASS，1 为 FAIL，2 为仍有未决基线。

