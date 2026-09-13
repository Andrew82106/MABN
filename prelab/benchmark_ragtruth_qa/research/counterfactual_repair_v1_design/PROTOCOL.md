# CERP-v1.1：证据引导反事实修复探针（fit-only 预注册）

## 1. 目标

当前模型在 4-BPE 窗口上的主要缺口是 Evident Conflict：既有开发审计只检出
195/997 个校准冲突窗口。冻结 ModernBERT NLI 在 4,109 个 fit-only 最小对上却能
以 0.9642 的准确率把受控扰动端排在支持端之前。因此 v1 不再继续增加“整条主张
风险”，而是检验一个更直接的信号：

> 若把回答中的一个局部槽位替换成资料里的值后，资料对新陈述的支持明显提高，
> 则风险只落在原回答中被替换的字符及覆盖它的 4-BPE 窗口。

本方案称 **CERP-v1 (Counterfactual Evidence Repair Probe)**。它属于本文方法，
不修改任何正式 baseline。

## 2. 冻结数据边界

- 人工主监督：`fit_expansion/data/fit.jsonl` 的 3,680 个 fit 回答、615 个
  source-connected group；共 983 EBI、271 EC、609 SBI、30 SC span。
- 原子主张：`research/atomic_relation_expanded_fit_v1/microclaims_fit3680.jsonl`；
  34,941 条，现有共同窗口接口可用其中 34,919 条。
- 检索候选：已有 evidence-union，固定为 attention top-3 与每个 passage 的
  BM25 top-2 的并集；共 237,739 个 claim-sentence 实例。
- 银标排序辅助：`research/semantic_conflict_augmentation_v1/
  pairs_fit_silver_strict.jsonl` 的 4,109 对。它只是受控扰动银标，不当成人工金标。
- 共同评测：现有 653,979 个 4-BPE、stride-1 fit 窗口；整答分数取窗口最大值。
- 本版所有生成、门禁、选型和阈值只读 fit。代码不得出现 calibration、旧 official
  test 或 replacement holdout 的加载路径。过去的 calibration 诊断已影响研究假设，
  所以本轮即使严格 fit-only，也只算开发；最终结论必须依赖新的未触碰测试集。

## 3. 无标签修复候选生成

候选生成器先冻结，再允许读取 fit 标签做覆盖审计。每个原子主张只检查已冻结的
evidence-union 句子，最多保留 12 个候选。每个候选必须满足：

1. 只改变主张中的一个连续字符区间；插入时允许零长度原区间。
2. 替换文字逐字来自对应资料句；整条资料句不能直接复制成新主张。
3. 至少保留两个原主张内容词，且原主张 lexical token 保留率不低于 0.45。
4. 资料句与主张至少共享两个非停用内容词，或属于数值、日期、来源编号等强类型槽。
5. 原区间最多 5 个 lexical token，替换区间最多 6 个 lexical token。
6. 去重键为 `(response_id, microclaim_id, edit_start, edit_end,
   repaired_text, evidence_sentence_sha256)`。

按以下固定规则产生候选：

- `citation`：把显式 Passage/Source/Document 编号换成候选证据句所属 passage。
- `number`：整体替换数值及相邻货币、百分号、温度或单位；同亚型优先，但允许
  不同单位修复，以覆盖 `$35` 与 `35%` 这类角色错误。
- `temporal`：替换年份、月份、星期、时刻和时长槽。
- `negation_direction`：替换或删除 not/no/without/never，以及
  bigger/smaller、before/after、increase/decrease、cause/prevent、
  include/exclude 等冻结方向词。
- `entity`：替换原主张已有的专名候选；资料端只取同一句中的专名连续块。
- `aligned_relation`：对主张和资料句的 lexical token 做确定性
  `SequenceMatcher` 对齐，只保留两侧合计至少两个锚点、单一小差异块的替换。

候选排序完全不看标签：先按来源/主张内容重合度降序，再按 union 检索位次、
槽位优先级、编辑长度和字符坐标稳定排序；每个原区间最多保留 3 个替换，总数 12。

## 4. CPU 覆盖门禁

标签隔离分两阶段：`prepare` 只写无标签候选并冻结 SHA256；`audit` 才读 fit 标签和
共同窗口。进入任何 NLI/GPU 阶段前必须同时满足：

- 271 个 EC span 的候选区间覆盖率至少 0.60；301 个 EC+SC 合计至少 0.60。
- EC+SC 风险窗口的候选触达率至少 0.50。
- 完全无错误回答中的正常窗口触达率不高于 0.25。
- 冲突窗口触达率 / 完全无错误回答窗口触达率至少为 2.0。
- 按 `sha256("cerp-qc-v1"|candidate_id)`、每类最多 10 条抽出的冻结样本中，
  人工复核 `clear` 至少 0.80、`reject` 不高于 0.10。v1.1 的抽样盐为
  `cerp-qc-v1.1`。`clear` 要求单一编辑、语法
  可接受、替换确实来自所列资料句且没有整句复制；`ambiguous` 不当 clear。

若任一门禁失败，v1.1 停在 CPU 阶段；不能因为看到覆盖结果再改规则并沿用 v1.1 名称。
改规则必须新建 v2、重新冻结并使用新的 QC 样本盐。

候选本身不等于“错误”或“正确修复”。特别是正常答案也会有替代写法。所有候选
在训练前都保持无标签；禁止把“出现候选”直接当正例，以免污染负例。

## 5. 反事实分数与特征

若 CPU 门禁通过，使用已冻结的 `tasksource/ModernBERT-base-nli` revision
`de4ab7e77845098b7fab7f6ab9d370ddff27b19c`。原陈述对证据句的 E/N/C 尽量复用
现有 occurrence-bound cache，只对修复后陈述新增推理。设原陈述概率为
`(E_o,N_o,C_o)`，修复后为 `(E_r,N_r,C_r)`，核心差值为：

`m_repair = [logit(E_r)-logit(E_o)] + [logit(C_o)-logit(C_r)]`。

候选级固定特征包括：两端 6 个 E/N/C、`delta_E`、`delta_C`、`m_repair`、
union attention/BM25 位次与分数、问题词覆盖、锚点数、保留率、两端编辑长度、
替换在 passage 中的出现次数、跨 passage 唯一性、来源编号一致性和 6 类槽位 one-hot。

4-BPE 窗口只聚合编辑区间与该窗口相交的候选；零长度插入映射到包含边界的窗口。
窗口特征固定为：候选数、最大与 top-2 均值 `m_repair`、最大候选可靠度、最大
`E_r`、最大 `C_o`、候选来源数、来源熵、替换歧义率、无候选标志和六类最大值。
不把分数铺到整条 claim，也不做无条件 `max/OR` 融合。

## 6. 三段小模型

1. **候选可靠度头**：StandardScaler + L2 LogisticRegression，`C=0.1`，用 4,109
   银标对的两个方向训练：`corrupted→supported=1`，`supported→corrupted=0`。
   split component 等权，类型使用逆平方根频率且最大权重 4。它只判断“这个局部
   修改是否提高证据支持”，不产生 QA 金标。
2. **冲突窗口头**：StandardScaler + L2 LogisticRegression，`C=0.01`，输入上述
   修复聚合特征，目标只是在窗口内是否有 EC 或 SC。权重依次做到 group 等权、
   answer 等权、window 等权，再二分类平衡。
3. **最终融合头**：StandardScaler + L2 LogisticRegression，`C=0.01`，目标为
   benchmark any-error。输入冻结 token-source-attribution-v4 的 fit-OOF logit、
   冲突头 logit、二者乘积、`conflict*(1-base)`、替换歧义、无候选标志及编辑位置
   的 generation NLL/source-attribution 摘要。它可以学习压低伪修复，而不是只能
   增加报警。

所有超参数固定，不做 C、层、头、候选上限或损失权重搜索。若
token-source-attribution-v4 的 fit 阶段未完整通过审计，本版停止，不临时更换底座。

## 7. 防泄漏交叉拟合

- 外层沿用 5 折 source-connected group。一个 source group 的所有生成器回答始终
  在同一折。
- 外层 held group 对应的银标 pair，以及 `split_component_groups` 与 held group
  相交的全部 pair，一律从该折候选头训练中删除。
- 为训练最终融合头，在每个外层训练集内再做 4 折 group cross-fit，产生完全
  out-of-fold 的冲突头分数；外层 held 分数由外层训练集重拟合的冲突头产生。
- 底座输入必须是其既有 fit-OOF 分数；禁止拿 full-fit 对 fit 的 in-sample 分数。
- 窗口阈值和 answer 阈值只从完整外层 OOF 预测分别选择；规则仍为 F1、precision、
  更高阈值。随后用全 fit 重拟合三段模型并冻结哈希。

## 8. fit-only 晋级与停止规则

主比较只看 634 个目标 Llama-2-7B fit 回答的外层 OOF，其他生成器可参与训练，
但不能稀释目标域评价。相对同一折、同一窗口上的冻结 v4 底座，必须同时满足：

- window F1 提高至少 0.020，且 5 折中至少 4 折不下降；
- EC window recall 提高至少 0.15，EC span 至少命中率提高至少 0.10；
- 新增预测窗口 precision 至少 0.50；
- answer F1 下降不超过 0.005；
- 615 group 固定 OOF bootstrap 5,000 次，window F1 差值的 95% 区间下界大于 0。

未同时达到就停止，不读取 calibration。达到后才允许用冻结模型和 fit 阈值做一次
开发集评价；该结果仍不用于改结构。进入新的独立测试前，开发集严格 window F1
至少 0.73、answer F1 至少 0.89，且两项均不低于当前候选和最强忠实 baseline。
论文最终接受标准是在未触碰替代测试上 window/answer F1 均至少约 0.75，并优于
同一评测要求下、未改结构的最强正式 baseline。

## 9. 消融与资源

固定消融只解释贡献，不参与选型：`base only`、`raw NLI without repair`、
`repair only`、`base+repair`、去掉银标候选头、逐一去掉六类槽位。

最大 34,919×12=419,028 个修复候选。按现有 RTX 3070 上 158,929 个 NLI 请求
537.6 秒的实测吞吐，预计新 NLI 约 25--45 分钟、峰值显存约 1.3--2 GB；CPU 候选
生成与嵌套逻辑回归约 20--60 分钟，内存约 2--4 GB，工作文件约 1--2 GB。
v1.1 不重新 teacher-force Llama，不训练 NLI 编码器，也不占用正式 baseline 目录。

## 10. v1 中止记录

首次 v1 CPU 预检发现一个上游 native atomic JSONL 混放了 634 条 fit 与 159 条
calibration。加载器解析到第 635 行后因 partition assert 中止；它未读取标签、未写出
候选、未训练或使用 GPU，但已解析一条无标签 calibration 布局。该次冻结保存在
`FREEZE_ABORTED_V1.json`。v1.1 改用物理独立的 `data/fit.jsonl` 634 行与独立的
expanded-fit 3,046 行，不再打开上述混合文件，并更换版本号和 QC 盐。
