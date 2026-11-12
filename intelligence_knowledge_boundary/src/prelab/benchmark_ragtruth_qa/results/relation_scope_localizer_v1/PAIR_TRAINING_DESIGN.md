# 最小关系修复对训练：冻结设计

## 决策

下一项只做 **FAVA 单处关系修复对的相对训练**。同一资料、同一回答只替换一个实体或关系片段；模型必须给原错误片段更高风险、修复片段更低风险。它直接补当前模型缺少的“词都出现过，但主体、动作、条件或属性组合错了”监督。

本轮缓存读出已经失败：显性冲突命中从 195/997 增至 232/997，但 cal 窗口 F1 只有 0.666208，FP 比当前候选多 468。故不再增加 claim 聚合、词面规则或浅层融合。

## 固定输入

- 数据只取 `auxiliary_fava_local_pairs_v1` 已冻结的 10,040 个单处修复对；字符结构检查不通过的 4 对在分词前按既有标志排除。最终数量以两侧目标均能映射到至少一个 lexical token 的 CPU 门禁结果为准，门禁后立刻冻结，不能按 QA 成绩删样本。
- 5,335 个 entity 对负责主体/客体绑定，4,705 个 relation 对负责谓词、属性和局部限定。两类都保留；不按当前 12 个失败案例挑词或挑关系。
- 每对有两个完全独立的模型输入：`原资料 + 空问题 + 原错误回答` 与 `同一原资料 + 空问题 + 单处修复回答`。输入中不出现另一版本、编辑标记、作者类型或“哪个更好”的提示。
- 只监督 `candidate_pairs.jsonl` 给出的两侧目标字符范围。回答其余位置保持未知，不填成安全标签；修复后整答也不标成正确。
- QA 阶段精确复用 `full_context_fava_transfer_v1` 的 3,680 fit 回答、权重、前三个顺序和 159 calibration 回答。official test 不读取。

## CPU 准备与坐标门禁

1. 分别按 `model_inputs.jsonl` 对原版和修复版做完整 tokenizer 编码，保留 answer 字符范围、token offset、输入 ID 和哈希。
2. 目标 token 定义为：token 的字母数字字符与目标字符范围有正交集。纯标点目标不强行生成词元标签。
3. 每对必须同时通过：原切片 exact、修复切片 exact、范围外字符 exact、反向补丁恢复 exact、两侧目标 token 非空、无截断、输入长度不超过 checkpoint 上限。任一失败则整对隔离，不能只留一侧。
4. 在冻结可用集合上重算 `材料组 → 原回答 → 局部对` 等权；一对的两个版本共同占一份质量。保存 pair 顺序、权重、字符/词元映射和全部源文件 SHA256。
5. 复用 `src/local_repair_pair_loss.py` 已通过的公式与数值检查，不另改 margin 或损失系数。

这一步只需 CPU，不读取 QA calibration 标签。完成后才允许 GPU smoke 和训练。

## 模型与目标

- 模型：与 `full_context_fava_transfer_v1` 完全相同的通用 ModernBERT-base 初始化、revision、fresh 两类 token head 和 seed `20261005`。不能从已看过 QA 或旧 FAVA 的 checkpoint 开始。
- 每个目标 token 的风险 logit 为 `z = logit(risk) - logit(safe)`。对一对原错误目标 `b`、修复目标 `r`：

  `L_bce = 0.5 * [mean(BCEWithLogits(z_b, 1)) + mean(BCEWithLogits(z_r, 0))]`

  `L_rank = softplus(1 - mean(z_b) + mean(z_r))`

  主方法固定 `L = L_bce + L_rank`。margin=1、rank 系数=1，不搜索。
- 唯一匹配控制为相同两侧输入、相同可用对、相同 batch/order/更新数的 `L = L_bce`。这不是第二个候选网格；它只回答相对排序项是否提供增量。
- 每侧先在目标 token 内求均值，再各占一半；目标长短不能改变该侧质量。范围外 logit 不直接进入损失，但仍可通过共享编码器收到间接梯度。

## 训练

1. 两个分支都从相同 fresh 初始化独立训练。
2. 辅助阶段跑完整 1 epoch。一个 pair 的两侧在同一优化步内；有效 batch 固定为 4 pairs（8 次 answer forward），最后短 batch 用实际 pair 数归一。AdamW、`lr=1e-5`、`weight_decay=.01`、clip=1、无 scheduler；BF16 forward/checkpoint recompute，参数、梯度、Adam 状态和损失保持 FP32。
3. 辅助阶段不看 QA calibration，不选 checkpoint。保存最终模型、优化器、RNG、实际更新数、输入 token 数、时间和峰值显存。
4. 同一优化器连续进入原 QA fit，跑固定 3 epoch；复用既有 QA 权重和前三个 answer order，不重置 optimizer 或 RNG。
5. 两分支在数据、前向次数、更新数和 QA 阶段完全匹配；唯一差异是 rank 项。

## 读出与评测

- 每个 QA lexical 原 BPE 取 token risk probability；原 stride-1 4-BPE 窗取其 lexical token 最大值，整答取全部 eligible 窗最大值。原标签、拒答和短窗规则不改。
- 每个 QA epoch 同时保存 fit/cal 全分数。沿 `full_context_fava_transfer_v1` 现有规则，在 3 个 QA epoch 内用同一 calibration 选择键和独立窗口/整答阈值；同时明确这是反复使用的开发集结果。
- 主比较是 rank 分支减 paired-BCE 控制，必须同时报告：窗口与整答 P/R/F1/AUROC/AP、TP/FP/FN、Evident/Subtle Conflict 窗口召回、每个人工风险 span 的 any-hit/全漏、以及新增 TP/FP 和原 TP 丢失数。
- 当前候选 `semantic_claim__old_tree__large_weight0.4` 只作冻结外部参照，不参与训练、权重或阈值选择。正式 baseline 不修改。

## 解释边界

FAVA 修复是有噪声银标，不能证明修复片段或整答完全正确。它能直接训练局部实体/谓词关系，却没有 passage 1/2/3 的错引监督，也不能完整覆盖跨句重复事件。若相对目标仍不能提高显性冲突召回并控制 FP，下一步才值得新增 QA-fit 内的来源/步骤最小对；本 v1 不混入该扩展。
