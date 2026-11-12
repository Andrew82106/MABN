# Onset + Segment Probe v1：fit/train-only 设计与容量审计

## 结论

v1 已完成 human-only 严格 fit OOF，结论是：**停止 971 条银标特征回放，不打开 calibration**。

人工 fit 有 646 个发布 span、639 个唯一原始起点 BPE；映射成可由二状态解码器表达的 lexical 风险段后有 568 个起点。每个固定 hash fold 有 101--148 个二值起点，足够训练两个强正则线性头，不足以支撑深序列网络。RAGognize no-hint 可再提供 797 个自动 span、714 个二值起点，能用于预训练。

RAGTruth 634 个 fit 回答已有 `hidden_last[4096]`、`lb[1024]`、`nll` 缓存，因此本轮按冻结的 68 维输入、双线性头、25 epoch 和五折 hash group split 跑了 human-only 投资诊断。它不冒充最终 silver→human 模型。窗口有排序信号，但主任务离可用仍很远：窗口 OOF F1 只有 0.366242，整答 max AUROC 0.547944，起点窗召回 0.297922。

结构性原因很清楚：continuation 条件样本中 16,591/17,159（96.69%）是“继续”，stop 只有 568 个。即使损失内做类平衡，当前读出仍容易在一次假起点后持续传播风险；clean-window FPR 达 16.25%，整答阈值把 634 个回答中的 613 个判为阳性。银标预训练不能保证修复这个状态模型缺口，故 v1 在投入昂贵回放前停止。下一版应先在 fit/train-only 上加入直接 risk emission 锚点，并校正 continue/stop 的平衡或先验。

全过程未读 calibration/test 标签，未启动 GPU，未改 baseline。唯一训练是本报告明确标注的 CPU human-only OOF 诊断。

## Human-only 严格 OOF 结果

每个 `group_id` 只进入一个 held fold；标准化、类权重和两个线性头都只在另外四折拟合。五折合并后一次选择窗口和整答阈值。完整 token、window、answer、span 预测及各折参数保存在 `human_only_oof_v1/`，并由独立脚本重新构造 4-BPE 几何、answer=max、阈值和主指标。

现有缓存没有 EOS hidden；诊断将答尾 stop 明确标记，并复用最后一个 lexical representation。这是 v1 可执行化约定，也是结果限制，不能隐去。

| 层级 | F1 | Precision | Recall | AP | AUROC |
|---|---:|---:|---:|---:|---:|
| 4-BPE window | 0.366242 | 0.298832 | 0.472925 | 0.280414 | 0.715829 |
| answer=max | 0.682253 | 0.523654 | 0.978659 | 0.550045 | 0.547944 |

窗口阳性率仅 0.127746，AP 是阳性率的 2.20 倍，且五折 window AUROC 都高于 0.55，说明白盒特征确有局部信息。但它没有转化成可靠定位或整答排序：起点窗召回 0.297922、内部延续窗召回 0.496513、干净窗 FPR 0.162514。646 个 span 中任意命中 502 个，但完整覆盖只有 13 个；Evident Conflict 任意命中仅 49/109。

起点头本身的 OOF AUROC 为 0.767927（发布起点辅助目标）/0.775426（二值段起点），说明失败不等于“起点无信息”。问题在状态递推：极少的 stop 事件难以约束假起点之后的尾部，最后被 `answer=max` 放大。预先冻结的投资门槛要求窗口与整答排序、起点和延续召回同时过线；本次因 answer AUROC、起点召回和延续召回失败，门禁为 `STOP_971_REPLAY`。

## 读取边界

审计脚本只有 7 个精确白名单输入：

- RAGTruth：`data/answers_fit.jsonl`、`data/fit.jsonl`、`data/tokens_fit.jsonl`；
- RAGognize：`conversion_v1/all_train/{answers,sequences}.jsonl` 与 `conversion_v1/no_unanswerability_hint/{answers,sequences}.jsonl`。

脚本拒绝任何带 `test`、`calibration` 或 `validation` split 组件的输入路径，只用 Python 标准库。字符级 oracle 对 634 + 971 + 1842 条回答逐条重建 lexical/risk mask，并验证窗口分区；结果为 PASS。no-hint 971 是 all-train 1842 的严格子集，二者不能相加当作独立数据。

跨来源按 NFC、空白归一后的 question、retrieved passages、response 精确哈希均为 0 个交集。这只能排除精确重复，不能证明没有语义近重复；正式 OOF 仍以材料 `group_id` 整组移动。

## 容量

“发布起点窗”指一个有效 4-raw-BPE 窗包含至少一个发布 span 的首个 lexical BPE；一个窗同时含起点和延续时归到起点。内部延续窗是风险窗但不含发布起点。干净窗不含风险 lexical BPE。

| 训练源 | 回答 / 组 | 风险答 | 发布 span | 唯一发布起点 | 二值段起点 | 起点窗 | 内部延续窗 | 干净窗 | 有效窗 | span 中位 lexical BPE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RAGTruth human fit | 634 / 615 | 328 | 646 | 639 | 568 | 2,551 | 18,926 | 146,646 | 168,123 | 19 |
| RAGognize no-hint silver | 971 / 491 | 542 | 797 | 797 | 714 | 2,745 | 25,071 | 71,833 | 99,649 | 20 |
| RAGognize all-train silver | 1,842 / 917 | 1,007 | 1,483 | 1,483 | 1,338 | 5,158 | 48,683 | 143,611 | 197,452 | 20 |

RAGTruth 的 646 个 span 只有 639 个唯一起点位置，其中 71 个唯一起点被 lexical 二值并集合入已有风险段，最终只有 568 个 `0->1` 起点。来源包括 7 个含重叠 span 的回答，以及 span 间只有空格/标点、移除 nonlexical BPE 后没有干净状态的情况。v1 保留两种真相：起点辅助损失监督 639 个唯一发布起点；转移解码器只监督 568 个可实现的段起点。不能为了凑齐 646 段而伪造干净 token。

窗口只是相关重复观测。RAGTruth 的 2,551 个唯一起点窗来自 639 个起点，不能当作 2,551 个独立阳性；真正的容量上限仍接近 span、回答和组数。continuation 也一样：17,159 个人工风险 lexical BPE 并不等于 17,159 个独立错误。

### span 长度

| 来源 | 1--4 BPE | 5--8 | 9--16 | 17--32 | 33+ | p90 | 最大 |
|---|---:|---:|---:|---:|---:|---:|---:|
| RAGTruth human fit | 70 | 66 | 151 | 204 | 155 | 50 | 277 |
| RAGognize no-hint | 66 | 89 | 185 | 259 | 198 | 61 | 322 |
| RAGognize all-train | 114 | 168 | 337 | 508 | 356 | 61 | 417 |

本地 QA span 明显不只是一两个 token：人工中位数 19 BPE、p90 50。因而只学起点会漏掉大段内部窗口；必须有延续/停止读出。反过来，长 span 使 continuation token 很多，却没有同比增加独立事件量。

### 人工错误类型

| RAGTruth 类型 | span | 回答 | 组 | span 中位 lexical BPE |
|---|---:|---:|---:|---:|
| Evident Baseless Info | 343 | 200 | 196 | 23 |
| Evident Conflict | 109 | 87 | 87 | 14 |
| Subtle Baseless Info | 187 | 112 | 110 | 16 |
| Subtle Conflict | 7 | 5 | 5 | 14 |

关系冲突的人工监督总量只有 116 个 span，其中 Subtle Conflict 仅 7 个；固定 hash 五折里有两折为 0。RAGognize 的 797/1,483 个自动 span 没有发布错误类型，不能被规则重命名成 conflict。结论是训练一个全局起点/延续头，按类型做同阈值诊断；不训练类型专头，也不声称银标补足了关系冲突监督。

## 冻结结构：OSR-Linear-68

模型在 lexical BPE 序列上运行，nonlexical BPE 只保留为 4-BPE 几何上下文。

每个 token 的基础输入为：生成模型 teacher-forced 最后一层 hidden、1,024 维 Lookback、chosen-token NLL，以及当前 NLL 差分、过去四个 lexical token 的因果 NLL 均值、归一化位置。hidden 和 Lookback 分别用固定 Rademacher 投影压到 32 维，总计 68 维。投影不看标签；均值/方差只在每个 outer fold 的 human 训练组拟合。

两个头都为线性 sigmoid：

```text
p_on(i)   = sigmoid(w_on · x_i + b_on)
p_cont(i) = sigmoid(w_cont · x_i + b_cont)

r_1 = p_on(1)
r_i = (1-r_{i-1}) p_on(i) + r_{i-1} p_cont(i)
```

`p_on` 负责安全状态进入风险；`p_cont` 在已处于风险时判断继续还是停止。两个头合计 138 个监督参数。软风险 `r_i` 进入主评测；同一转移概率可跑 Viterbi，仅用于输出连续 span，不用 Viterbi 硬标签算主分数。

训练损失为二值并集路径的转移 NLL，加 `0.25 ×` 发布 span 首 token 的平衡 BCE。前者在 gold 前态为 0 时训练 onset，在 gold 前态为 1 时训练 continue/stop，并为答尾风险加入 EOS stop；后者保留被二值并集吞并的原始 span 起点信息。所有标签都由发布 span 确定性派生，不改答案、不注入冲突、不造新正例。

可选的 with-reference / no-reference 分支只在未来有完整缓存时启用：按 CORTEX 定义计算 `Delta h_i`，再用 reference attention 得到上下文残差 `c_i`。若这些量送入本设计的 onset/continuation 头，结果仍叫 ours，不叫 CORTEX baseline。

## 严格 group OOF 与银标迁移

RAGTruth outer fold 固定为：

```text
uint64_be(SHA256("ragtruth_fit_human" + NUL + group_id)[0:8]) mod 5
```

每折依次执行：先完全留出该折所有材料组；只用其余四折 human token 拟合标准化；在全部 no-hint RAGognize train 组上预训练 15 个固定 epoch；再在四折 human 上微调 25 个固定 epoch；最后仅给 held human 组出分。不早停，不按折改超参，不使用全 fit 拟合的上游分数。若未来发现跨来源精确材料重复，相关 silver 组从对应 human held fold 的训练中剔除。

五折 human OOF 分数合并后，窗口阈值只在 fit OOF 上按 `(F1, precision, recall, threshold)` 词典序选；整答分数为该回答全部有效窗最大值，整答阈值同法独立冻结。然后用全部 human fit 重训一次，保存 checkpoint 和特征/阈值哈希。

原冻结协议只允许 no-hint 成为 v1 的银标方案；human-only 与 all-train superset 只能做 fit OOF 归因诊断。现在投资门禁已失败，因此三者都不进入 calibration。任何学习型投影、scaler、score stacking 或 baseline 融合也必须在组内交叉拟合；现有 full-fit incumbent 分数不能作为本模型的 OOF 输入。

首次 calibration 只允许一次：冻结后读取标签，按固定阈值报告窗口与 `answer=max` 的 P/R/F1/AP/AUROC，同时用同一阈值报告 span any-hit/full-cover、起点窗召回、延续窗召回、干净窗 FPR 和类型召回。不得在 calibration 上选阈值、epoch、特征或结构。历史候选 **0.6902813989 窗口 F1 / 0.8910891089 整答 F1** 只作用户给定的开发门槛；要替换它，窗口必须严格更高且整答不得更低。这个 calibration 已被项目反复开发使用，即使过门也仍是开发证据。

## 与两篇论文的边界

[First Hallucination Tokens Are Different from Conditional Ones](../../../../doc/ref_paper/intelligence_knowledge_boundary/hallucination_detection/First_Hallucination_Tokens_author_preprint_v4.pdf)（v4，PDF SHA-256 `9ae5ccf282c6fa27a80f9cf003f93ea0c9a1b2ae3df0cf4e0cd4009fb1809e5d`）在 §2--§3 按 span 内位置比较 logit 派生信号，发现首 token 通常比条件续写 token 更可探；§4.1 明确说它是简化分析，不是可部署分类器，也没有使用 RAGTruth 类型。它支持“单独建模起点”这个动机。它没有提出本报告的双头、转移递推、损失、银标迁移或 4-BPE 读出，不能作为这些组件的原版 baseline。

[CORTEX](../../../../doc/ref_paper/intelligence_knowledge_boundary/hallucination_detection/CORTEX_2026_preprint.pdf)（v1，PDF SHA-256 `0a0a1c68b4dae9664daba6df919a552f92e048f70fab21b47b3d84411e34fd69`）在 §3.1--§3.4 使用相同 answer 的 reference/no-reference 最后层 hidden 差、attention contextual residual、三层 MLP 加权 BCE，以及固定自循环概率的对称 label-persistence smoothing；整答取 token max。以下才是本报告的新方法：发布首 token 辅助监督、独立且非对称的 start 与 continue/stop 头、由两个头驱动的风险递推、automatic-silver→human OOF 训练顺序，以及本项目的 onset/continuation/clean 4-BPE 映射。

若要报告“原版 CORTEX baseline”，必须另行保持其 paired features、MLP、weighted BCE 与 persistence smoother；把 CORTEX 特征塞进 OSR 后得到的是组合新方法。现有 baseline 文件、参数、阈值和结果均未改动。

## CPU 可行性、回放准备与当前门禁

低维矩阵很小：human fit 的 139,518 个 lexical token 以 float32 保存 68 维约 38 MB；加 no-hint 的 88,133 个约再加 24 MB。138 参数的双线性头和五折递推本身适合 CPU，主要成本是白盒特征准备。

human-only 五折 OOF 实测 27.3 秒，包括重新哈希约 2 GB fit NPZ、投影、训练、预测和序列化；独立复核再用 2.4 秒。该成本不是当前阻塞。

RAGognize no-hint 的标签盲回放 plan/runner 已准备，但未加载模型。971 条回答共有 684,655 个完整输入 token、102,778 个回答 token；输出 `lb+nll+hidden` 原始数组约 1.965 GiB，按 RAGTruth fit 压缩比估计 NPZ 约 1.097 GiB，后续 68 维 float32 训练缓存约 26.7 MiB。本地 checkpoint 文件共 13.48 GB。CPU 耗时没有模型 smoke，不能用 GPU 历史时间冒充；runner 要求先做最短/最长两条及两条 RAGTruth fit 缓存兼容检查，再据实外推。

当前 Python 环境缺少 `torch/transformers/bitsandbytes/accelerate`，CPU NF4 backend 也尚未验证。更关键的是 human-only 投资门禁已失败，所以 runner 会在导入这些库和加载模型之前 fail closed。当前状态为 **STOP_971_REPLAY_AFTER_HUMAN_ONLY_OOF**。只有新的 fit/train-only v2 先通过 human-only group OOF，才值得重新授权 971 回放；v1 不进入 calibration。

## 复现

```powershell
py -3 .\audit_capacity.py --mode capacity
py -3 .\audit_capacity.py --mode selfcheck
py -3 .\onset_segment_core.py --selfcheck
py -3 .\run_human_only_oof.py --selfcheck
py -3 .\run_human_only_oof.py --run
py -3 .\verify_human_only_oof.py
py -3 .\ragognize_no_hint_replay.py --plan
py -3 .\ragognize_no_hint_replay.py --selfcheck
```

`CAPACITY.json` 保存容量；`human_only_oof_v1/METRICS.json`、`FOLD_MODELS.json` 和四个压缩 JSONL 保存全部 OOF 结果；`human_only_oof_v1/VERIFY.json` 保存独立复核；`ragognize_no_hint_replay_v1/PLAN.json` 与 `SELF_CHECK.json` 保存未加载模型的回放计划和资源审计；`SHA256SUMS.txt` 绑定最终交付文件。
