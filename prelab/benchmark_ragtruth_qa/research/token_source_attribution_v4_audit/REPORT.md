# token_source_attribution_v4 训练/评分只读审计

审计范围：`src/score_token_source_attribution_v4.py`，并只读核对其直接调用的 token 特征与 gold 构造代码。未读取 fit、calibration 或 official test 数据及任何分数结果。

## 结论

未发现会必然算错本轮窗口分数或 F1 的实现错误。4-BPE 索引、标签、特征行、group-held-out 预测、双阈值和 `answer=max(window)` 均按代码声明执行。

严格 calibration 数值可成立，但要满足三个尚未由 scorer 自证的条件：fit/cal group 确实互斥；没有重复查看 calibration 后再改方案；答级阳性与可定位的窗口阳性采用了预期定义。以下三项应在把结果称为“正式严格评测”前处理或明确记录。

## 可能影响本轮结论的条件性问题

1. **答级目标可能与窗口监督不一致。** `answer_scores()` 对所有窗口取最大值，却直接使用 `answers_*.jsonl` 的答级标签（`score...py:1007-1013`）；scorer 没有断言 `answer.label == max(window.label)`。上游 gold 把“存在官方 span”定义为答级阳性，但 risk BPE 只覆盖字母数字字符，因此理论上允许“答级阳性、全部窗口阴性”。若本轮存在这种记录，answer F1 的计算本身没错，但模型从未获得对应的正窗口监督，答级任务与训练目标不闭合。建议在 fit 打开后先冻结并报告不一致计数；按预注册规则选择保留、排除，或增加答级目标，不能看 calibration 后决定。

2. **scorer 未自行验证 fit/cal group 互斥。** fit 内的 `GroupKFold` 确实保证 train/held group 不交叉（`1087-1106`），但 `fit_complete.json` 不保存 fit group 集合或摘要，`evaluate()` 也不与 calibration group 比较。上游 `build_gold.py:314` 有互斥断言，所以在产物未被替换时大概率安全；当前 scorer 的“无泄漏”保证仍依赖上游。建议在 fit 冻结 group-id 摘要，在 evaluate 首次开标签时断言交集为空。

3. **“calibration 只打开一次”没有 fail-closed。** `evaluate()` 写入 `calibration_evaluation_started.json` 后，若崩溃、`complete.json` 缺失或被删除，再运行会再次打开 calibration（`1153-1172`）；结果仍写 `calibration_evaluations: 1`。一次正常成功运行不会因此改变数值，但该门禁无法排除重复查看后的人为调参。建议 started 已存在且 complete 不存在时直接拒绝继续，并由外部封存记录处理恢复。

## 已核对正确的实现

- **4-BPE 几何和标签对齐正确。** 每个窗口都核对连续 raw-BPE 索引、长度 `min(4, token_count)`、token id、绝对位置、字符区间、lexical 子集，以及 `label == any(risk_mask[token_indices])`（`788-845`）。短答案保留一个原长度窗口。
- **特征行对齐正确。** materialized lexical BPE 索引必须等于 `flatnonzero(lexical_mask)`；窗口聚合用 `searchsorted` 精确匹配，并与 `window.lexical_token_indices` 全等（`853-866`, `726-750`）。risk BPE 在 gold 构造中只来自字母数字字符，因此不会把正标签落到被特征过滤的纯非 lexical token 上。
- **候选矩阵切片正确。** full 为 513 维、full+NLL 为 514 维；band4 为 65 维、band4+NLL 为 66 维。无 NLL 候选返回属性矩阵前缀，带 NLL 候选才包含最后一列（`869-934`）。
- **group OOF 没有窗口/答级串折。** group 来自每个窗口，所有同组答案完整进入同一折；四候选共享同一折划分、同一训练权重和同一标签（`1087-1114`）。
- **阈值代码的 `>=` 与候选枚举一致。** 相同分数先合并，再在可实现阈值上最大化 F1，并依次偏好 precision 和更高阈值（`971-987`）。窗口阈值与答级阈值分开冻结。
- **`answer=max(window)` 实现正确。** 每个答案覆盖全部 eligible 窗口，取最大 score 后配对该答案标签（`1007-1013`）。
- **未见直接 calibration 标签泄漏。** `fit()` 只调用 `load_partition_meta("fit")`；候选、最终模型和双阈值先写入并冻结，`evaluate()` 才调用 calibration（`1067-1150`, `1153-1199`）。提前生成的 calibration 特征只读文本/模型输出，不读 gold 标签。历史上由同一 calibration 选过超参的 Lookback/large 列被明确排除。该判断只覆盖本模块可见的数据流，不能证明代码设计历史未受 calibration 反馈影响。

## 不会算错本轮数值，但协议表述应收紧

1. **“no_claim_projection / 精确 4-BPE”表述过强。** 窗口索引确实精确，但每个 lexical token 的 NLI 部分来自其所属整条 microclaim：`combine_token_features()` 用 `nli[owners]`，再按 token 的句子归因加权（`run_token_source_attribution_v4.py:813-836`）。一个 claim 可跨越当前窗口，因此这是“精确窗口聚合的 claim-conditioned token 特征”，不是完全局部的 4-BPE 语义特征。它不使用 gold，故不是标签泄漏；会产生跨窗口语义扩散，解释结果时应明说。

2. **fit OOF F1 不是无偏泛化估计。** 每个候选都在同一组 OOF 标签上选最优窗口/答级阈值，随后又在这些标签上选候选并报告 F1（`1108-1124`）。这不污染之后一次性的 calibration，但 fit OOF 数字含阈值拟合和四选一的乐观偏差。正式结论应以未反馈的 strict calibration 为准；若要报告无偏 OOF，需外层 group CV，内层选候选和阈值。

3. **OOF 阈值直接迁移到全 fit 模型。** pooled OOF score 来自五个不同 scaler/LR，最终 calibration score 来自全量 fit 的第六个模型。概率尺度可能漂移。这是常见近似，不是代码错误；可用嵌套 group cross-fitting 检查各折阈值稳定性，并预注册稳健的折阈值汇总或 OOF 校准映射。

4. **`answer=max(window)` 有长度效应。** 长答案窗口更多，负例的最大噪声分数自然更高。建议仅用 fit OOF 按窗口数分层报告 FPR/F1；若未来改聚合，可预注册 top-k mean、noisy-OR 校准或长度校正，不能用本轮 calibration 选择。

5. **GroupKFold 不分层。** 它主要平衡窗口数，不平衡正例率、答数或训练时的等 group 权重。候选之间仍公平，但折间概率尺度与阈值可能不稳。建议冻结一个同时平衡 group、答数和正窗口数的分组折表，或使用可复现的 StratifiedGroupKFold。

6. **四候选程序待遇相同，但“同一 C”不等于容量公平。** 四者共享 folds、weights、solver、阈值规则和选择规则；这一点公平。固定 `C=0.01` 对 65/66 与 513/514 维表示的有效正则化压力不同。若研究问题是“固定 readout 下哪种表示更好”，当前比较成立；若要比较各表示的最佳能力，应在 fit-only 内层 group CV 中给每个候选同一组 C 搜索预算。

7. **候选 tie-break 的“precision”实际只取 window precision。** `1123` 没有使用 answer precision。它对四候选执行一致，不会造成实现性偏袒；协议应写成 `window precision`，或明确加入答级 precision。

## 性能与内存

- fit full 矩阵约 330 MiB，band4 约 42 MiB，共约 372 MiB；selected full calibration 矩阵最多约 83 MiB。memmap 降低常驻矩阵内存，但 `x[train]`/`x[held]` 是高级索引，会把每折数据复制到 RAM；Scaler/liblinear 还可能再复制或转成 float64，full 候选峰值可超过 1 GiB。
- 共执行 `4 × (5 OOF + 1 final) = 24` 次 scaler+LR 拟合。两种表示各自只建一次磁盘矩阵，但 full 与 band4 分两遍逐答案、逐窗口聚合，重复解压每个 token-feature NPZ。
- `load_partition_meta()` 把 168,123 个完整窗口 JSON 对象及其文本、列表字段全部放入 Python 内存；这很可能比紧凑数值矩阵本身更浪费对象内存。
- 大 `.npy` 在每次缓存复用时重新 SHA-256，并用 `np.isfinite(...).all()` 全矩阵扫描；会增加磁盘 IO，并可能生成较大的临时布尔数组。

建议优先：一次遍历同时写 full/band4；按答案用 prefix-sum + 长度 4 rolling max 向量化 mean/max/NLL；把窗口元数据投影为紧凑数组；分块预测和分块 finite 校验；每个大文件每进程只验一次哈希。若保持 liblinear，需为 full 折训练预留约 1–2 GiB RAM；资源估计应以实测峰值为准。
