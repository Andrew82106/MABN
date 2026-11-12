# Semantic source attribution v1 scorer：独立只读审计

审计对象是 `src/run_semantic_source_attribution_v1_score.py` 及已冻结的 `protocol.json`、`PREPARATION.json`、`CPU_SELFCHECK.json`。结论为 **fit 前需修正**；GPU attribution 抽取本身可继续，正式 baseline 也未被改动。

## 已确认正确

- **没有发现 runner 内的 calibration/test 直接泄漏。** `prepare` 只读取连续的 fit 元数据；fit 标签到 `fit()` 才打开；calibration 标签到 `evaluate()` 固定完模型分数后才打开；没有 official-test 路径。fit/cal 的 group、response、source 交集均为 0。由于项目过去已反复使用这 159 条 calibration，未来结果仍只能称开发结果，不能称独立测试结果。
- **top-3 NLI 句子身份正确。** 对抽取期间已经原子提交的 320 条回答、4,470 个 claim、13,410 个 top-3 链接做了独立复算：选句排序错误 0、句子文本哈希错误 0、已有 atomic pair 的 premise/hypothesis/owner 身份错误 0、数组形状错误 0。其中 9,281 个链接命中旧 NLI 缓存，4,129 个需要同一冻结 ModernBERT 补推理。
- **特征维度和顺序正确。** semantic 为 `1024 + 288 + 66 + 4 + 12 = 1394` 维，名称 1,394 个且无重复；clean aligned 为 517 维，和实际拼接 `63 + 6×72 + 10 + 12` 一致。
- **历史 incumbent 特征确实删除。** 原 aligned 的最后 6 维为 `incumbent_claim_max/q90/q75/mean/q75_minus_answer_median/q75_within_answer_rank`；clean 矩阵实际不含这 6 维。
- **GroupKFold 和基本权重接线正确。** 五折 train/held group 交集均为 0；所有训练 claim 权重有限、正数，held 权重为 0；LR 通过 `logisticregression__sample_weight` 收到对应训练权重。
- **标签及投影正确。** any-error 与 conflict-only 分开生成，`conflict <= any`；9,055 个 fit claim 中 any-error 1,291、conflict 195。随机 claim 分数的独立复算与 `claim→4-BPE window max→answer max` 完全一致，最大绝对误差 0。
- **门控会保留所有 base 阳性。** 对固定 window 阈值，结果严格等于 `base_positive OR semantic_addition`。当 `answer_threshold > window_threshold` 时，新增窗被放在两个阈值之间，因此 strict answer 二值判定和 base 完全相同。

## fit 前必须处理

### 1. 稀有 conflict 的 sample weight 被最后一次组归一化重新压低

`claim_weights()` 先做类别平衡，随后又强制每个 group 总权重相同；后一步破坏了前一步。五折最终负类:正类权重质量如下：

| target | 五折范围 |
|---|---:|
| any-error | 2.17:1 – 2.22:1 |
| conflict-only | 6.85:1 – 7.38:1 |

这不等于“类别平衡”，而且对只有 195 个正例的 conflict head 尤其不利。协议目前也没有记录这套 sample-weight 配方。fit 前应二选一：明确冻结并命名为“等 group 质量、最终不等类”，或者把类别平衡放到最后/使用迭代配平，并把最终每折两类质量写进产物。

### 2. `no-add` 不是实际关闭门控

当 fit OOF 判定“不新增最好”时，`choose_add_threshold()` 返回 `nextafter(fit 中最大 semantic score)`。这个阈值只保证 fit 上不新增；full-fit 模型在 calibration 产生更高分时仍会新增。独立反例中 fit 的 no-add 阈值为 `0.30000000000000004`，calibration 分数 `0.31` 仍触发新增。

应保存明确的 `gate_enabled: false`，或使用 `+inf` 作为关闭值；`gated_scores()` 必须据此完全返回 base。否则产物里写“no-add”会与实际 calibration 行为不一致。

### 3. 冻结链缺少关键源码与金标哈希

`source_snapshot.json` 绑定了 scorer、atomic inputs 和 attribution 产物，却没有绑定：

- `run_aligned_evidence_head_v1.py`，它决定 517 维 clean 特征；
- `run_development.py`，它决定阈值和指标；
- fit/cal 的 answers、tokens、windows 金标文件；
- calibration NLI link/score 输入在最终 summary 中的完整来源链。

这些文件改变后，当前冻结检查不一定阻止 fit/evaluate，结果也无法从 `fit_complete.json`/`summary.json` 唯一复现。fit 前应把上述文件的 SHA256 加入 protocol/source snapshot，并在每个阶段重新核验。

## 建议同时修正

- `infer-nli` 在 NPZ 写出、JSON 尚未写出时不可安全重跑；`evaluate` 在 summary 写出、REPORT/complete 尚未写出时也会卡住。建议使用 commit 文件或允许“哈希一致则续跑”。
- `infer_missing_nli()` 复用了上游 3 GiB free-memory 门槛，但没有调用已有的 peak-memory gate。当前 attribution 抽取占用约 5.2 GiB，剩余约 2.8 GiB，因此 NLI 推理不能与它并行；抽取进程退出后再运行。冻结 ModernBERT 权重约 598 MB，旧 atomic NLI 的 pair 长度最大 223 token，顺序运行预计适配 8 GiB GPU。
- `gated_scores()` 用当前整批命中样本的最大 semantic 分数做缩放，所以连续 risk score 会随评测批次改变。二值 window/answer 判定不受影响，但 AUROC/AP 和线上单条部署的分数解释会受影响。宜改为只依赖 fit 冻结常数的逐样本映射。
- LR 没有断言 `n_iter_ < max_iter`。正式 fit 应把每折和 full-fit 的迭代次数、收敛警告写入结果。

## 审计判定

- Attribution 抽取、top-3 link 和缺失 NLI 补推理：**可继续，必须串行使用 GPU**。
- Scorer 的 fit/evaluate：**修正上述三项后再运行**。
- 正式 baseline：**没有被这个 runner 修改；它只是本文方法的独立输出目录**。

## 修正后复核（runner `627fb3bd…`）

三项核心修正均已真实生效：

- `no-add` 现在保存 `enabled=false, threshold=null`；独立反例复算中输出分数与 base 逐值相同，新增命中为 0。
- sample weight 现在只在等 group/answer/claim 初始质量后做一次最终类别平衡，不再二次等组。真实 fit 五折的 any-error 和 conflict-only 两类最终质量最大差分别低于 `2.1e-10` 和 `2.3e-10`；每折都有运行时断言，full-fit 也有同样断言和记录。
- `source_snapshot.json` 已存储并由 `check_prepared()` 复核 `run_development.py`、`run_atomic_microclaim_nli_v1.py`、`run_aligned_evidence_head_v1.py` 以及 fit/cal 六份 answers/tokens/windows 金标的 SHA256。

本轮只剩两个小漏项：snapshot 已存 `atomic_preparation_sha256`，但 `check_prepared()` 尚未复核它；`PREPARATION.json` 中的 `source_snapshot_sha256` 也尚未与当前 snapshot 文件重新比对。建议补两条断言后再次冻结 scorer SHA。除此之外，原三项 fit 阻断问题已解决。
