# 强制证据引用白盒探针 v1

## 结论与边界

这是一个 **256 答、fit-only、标签盲选**的小样本协议。每答都必须是本地被测模型同族的原生 `llama-2-7b-chat` 答案；再对其每个已冻结原子 claim，让同一个 Llama-2-7B-Chat checkpoint 从原三篇资料中生成一条逐字引用，读取引用内容及其生成时的 logprob、hidden 和 attention。它不启动 GPU、不读取 calibration/official test、不改任何 baseline；本目录也不含实验成绩。

现有材料足够执行：3,680 个 fit 答案/615 个资料组、34,941 行纯 fit/无标签原子文件、本地 Llama2 权重和既有白盒钩子均已齐。原答案的白盒缓存可直接作为“原 probe”对照，但**不能**冒充新引用轨迹；生成引用及机械引用都需在同一 prompt 下另做前向。

本方案受 [RLSeek（ACL 2026）](https://aclanthology.org/2026.acl-long.1492/) 的“先引用证据再核查”启发，但不是 RLSeek baseline。RLSeek 的[作者代码](https://github.com/WaldenRUC/RLSeek)用 Qwen2.5-7B、RL/多 rollout 和带引用的核查推理来输出幻觉跨度；本方案不做 RL、不生成核查 CoT、不用外部裁判，只让冻结的被测 Llama2 贪心引用一次，并读取**同一模型**的内部信号。因此后续只能称为“forced-evidence quote whitebox probe”。

## 1. 冻结样本

只在 `fit_expansion/data/fit.jsonl` 的 3,680 答/615 组中抽样。盐固定为 `forced_evidence_quote_probe_v1`：

1. 先按公开模型身份硬过滤 `model == "llama-2-7b-chat"`。fit 中共有 634 个原生答案，覆盖全部 615 组：599 组各1答、14组各2答、1组3答、1组4答；故无需 surrogate 降级。
2. 组键为 `SHA256(salt + NUL + "group" + NUL + group_id)`，按 `(hash, group_id)` 升序取前 256 个**有原生 Llama2 答案**的组。
3. 每组只在原生 Llama2 候选内计算 `SHA256(salt + NUL + "answer" + NUL + response_id)`，按 `(hash, response_id)` 升序取第一答。
4. 模型身份是被测对象限制；除此之外不按长度、质量或任何 gold 字段分层、补样、替换。
5. 五折号为 `uint64_be(SHA256(salt + NUL + "fold" + NUL + group_id)[0:8]) mod 5`；折大小固定为 40/56/62/45/53。

重算得到 256 个不同资料组、256 个原生 Llama2 答、3,776 个原子 claim，平均每答 14.75 个（2–42）。排序后的 response ID 加尾换行之 SHA256 为 `bf1fd595bc5bc1b6700d54404023f42cc230b4406e0f12c6539d1931dc4691e3`；排序后的 group ID 同口径为 `e24c27298c7604208bba9d6cbbb25d3e7aba4ff802d73446244c2c911ffb1f27`。任一重算不符即停止，不能另抽一批。

**冻结前修订记录。** 初稿曾在每组六种生成器中任取一答，得到 2,423 claims / response-ID digest `7b2586ae…`。该版本会把 P0 的“Llama2 回放别家答案”与 P3 的“Llama2 自己生成引用”混为同一被测模型，身份不成立；在任何 GPU/拟合前废止。当前原生限制后的 `bf1fd595…` 是唯一有效 cohort lock。

claim 只来自无标签文件 `research/atomic_relation_expanded_fit_v1/microclaims_fit3680.jsonl`（34,941 行，SHA256 `c1731ab6…`）。当前 256 答对应 3,777 行；固定 `any(char.isalnum() for char in text)` 后仅排除一条纯 `":`，留下 3,776 claims。CPU sanitizer 从 `fit.jsonl` 只取身份、模型、问题和三篇原文，并对每行调用 `src/prepare_atomic_microclaim_relation_expanded_v4.py`（SHA256 `9dac8461…`）的原样 `hypothesis_for`，物化唯一字段 `claim_prompt_text`。该函数只读无标签行的 `text/children_in_parent/child_index/antecedent_subject`；3,776 个 `(microclaim_id, claim_prompt_text)` 锁的 SHA256 为 `a59b8b52…`。

GPU manifest 只含身份、三篇原文、raw claim 坐标及 `claim_prompt_text`；prompt 中 `{claim}` **只能**取 `claim_prompt_text`，不得在 GPU 时改用 raw `text`、v4 `hypothesis` 或任何 gold-bearing examples。GPU 特征进程只读这份已哈希 manifest，不得打开 annotations、labels、calibration 或 test 路径。4-BPE 坐标和 gold 只由特征冻结后的 CPU evaluator 加入。

## 2. 唯一生成 prompt

以下 UTF-8 字节模板原样使用；三篇资料顺序及正文逐字保留，不使用 released prompt 中的 “Unable to answer” 指令。

```text
<s>[INST] You are given one claim and three passages.
Copy the single shortest complete sentence from the passages that most directly supports the claim.
Return exactly one verbatim quotation and nothing else. Do not explain, assess, paraphrase, or combine text from different places.

Claim:
{claim}

Passage 1:
{p1}

Passage 2:
{p2}

Passage 3:
{p3}

Required format: <quote>copied source sentence</quote> [/INST] <quote>
```

模板 SHA256 是 `2561177f99ba3017aef2d1687dad36ae3ad4bd8393ed6abe8ec8f63ce75b4b01`。它没有“无资料/拒答/支持很弱”等出口，也不把 gold 类型或风险定义告诉模型。

模板已逐字包含唯一的 literal `<s>`，故所有 prompt、机械 quote、生成后 replay 与 relation 分支都对**整段文本**调用 tokenizer，固定 `add_special_tokens=False`；不得再自动加 BOS/EOS，也不得把各片段分别 tokenize 后拼 ID。执行时必须断言首 token 为唯一 BOS，否则整批失败。

模型固定为本地 `NousResearch/Llama-2-7b-chat-hf` revision `351844e75ed0bcbbe3f10671b3c808d2b83894ee`，沿用已审计的 NF4 double-quant、BF16 compute、SDPA、单卡、TF32 off、eval、local-only 配置。逐 token 自定义解码，`do_sample=false`、`num_beams=1`，每步取最大 logit；完全相同时取最小 token ID。禁止 temperature、top-p、重复惩罚和 beam search。

`max_new_tokens=160`，其中包括闭合标记。每步把新 token ID 追加到完整 `generated_ids`，再以 `tokenizer.decode(generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)` 重建**整段生成前缀**；禁止逐 token decode 后拼字符串。该整段前缀首次完整出现 `</quote>` 后停止；EOS 或达到 160 先发生则停止并标为无效，不续写、不重试、不修复。EOS ID 单独保存且不进入逻辑引用文本。以冻结 `claim_prompt_text` 重算的最长输入 799 token，连同上限仅 959，小于模型 4,096；若执行时不一致或越界，整批失败，禁止静默截断。

## 3. 引用解析与关系信号

保存模型生成的原始 token、原始解码字节和停止原因。逻辑引用是助手前缀 `<quote>` 与首次 `</quote>` 之间的内容，仅去除两端 ASCII 空格、tab、CR、LF；原始 inner text 同时保存。缺闭合、嵌套/第二个标签、标签外非空内容、空引用均无效。

`source_exact_substring` 只在 **每篇正文分别**做逐字子串匹配，不跨 passage 拼接，不改大小写、标点或 Unicode。保存所有 `(passage_id, char_start, char_end)`；重复匹配不任意消失，坐标展示才以 `(passage_id, start)` 最小者为主。另报 raw-inner exact 和字符 LCS 比率，但不能用 LCS 把非逐字引用改判为 exact。

claim–quote 关系分两层。P1 的 **21 列及顺序**固定如下：

1. `parse_valid`；2. `source_exact_substring`；3. `exact_match_count_gt1`；4–6. `stop_close/stop_eos/stop_max` one-hot；
2. 7. `quote_bpe_len`；8. `quote_char_len`；9–11. 在 passage 1/2/3 任一处 exact 命中的三个 multi-hot；12. 主匹配的相对字符起点；13. 全部 exact 匹配数；
3. 14. `claim_prompt_text` 独立词项被 quote 覆盖率；15. quote 独立词项被 `claim_prompt_text` 覆盖率；16. 独立词项 Jaccard；17. exact 等于无标签句切分候选时的最大 BM25，否则 0；18–19. 字符 LCS 分别除以 `claim_prompt_text`、quote 字符数；20. 数字集合对称差率；21. 两者是否含否定词的 XOR。

词项正则固定为 `[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?`，casefold、不删 stopword；空集合分母用 1。数字正则固定为 `(?<![A-Za-z0-9])[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)%?`，去千分逗号后取集合，对称差率为 `|A△B|/max(1,|A∪B|)`。否定表固定为 `no, not, never, none, neither, nor, without, cannot, can't, doesn't, don't, isn't, aren't, wasn't, weren't, won't, didn't`。主匹配按 `(passage_id,start)` 最小，相对起点为 `start/max(1, passage_chars-quote_chars)`；无 exact 时 9–13、17 均填 0。不得按标签改列、正则或词表。

whitebox 条件在引用完成后才追加固定后缀 `Relation task: A = quote supports claim; B = quote contradicts claim; C = neither. Return one letter.\nAnswer:`。relation 分支统一构造 `rendered_prompt + logical_quote + "</quote>" + suffix`，再按整段以 `add_special_tokens=False` tokenize；缺闭合时也只在这个隔离分支合成一个 `</quote>`，不得改变生成、停止、`parse_valid` 或 exact 结果。EOS、畸形标签及标签外文本不进入 relation 上下文，quote 部分固定为去边空白后的 `logical_quote`。5 列依次为三类 `softmax([logit_A,logit_B,logit_C])` 概率、三类自然对数熵、原始 logit 的 top1−top2 margin。该分支不回流到引用生成，且仍是同一 Llama2，不是外部 NLI/LLM。

## 4. 白盒读取

“内容 token”固定为整段 prefix decode 后所得字符范围与去边空白后的逻辑引用有重叠的 token，闭合标签 token 不计。自定义贪心循环可用 KV cache 选 token，并保存每个内容 token 的 selected-token logprob、float32 全词表熵、`selected logit - max(other logits)`、token ID；这些生成统计以 cached 路径为准。主条件用真实生成前缀；机械白盒条件在同一 prompt 下逐字 teacher-force 机械句子，因此两者不能混称同一种轨迹。进入分类器的 **11 列 token 统计及顺序**为：logprob 的 mean、population-std、min、线性插值 P10、first、last、对归一位置 `t/max(1,T-1)` 的 OLS slope；entropy 的 mean、max；signed margin 的 mean、min。`T=1` 时 std/slope 为 0。

每个内容 token 读取最后 RMSNorm 后的 4,096 维 hidden，按 `(h-pca_mean) @ components.T` 立即用已冻结、fit-only 的 `hidden_pca.pkl` 投影到 64 维；磁盘只留 float16 token 投影。分类器按列拼接 float32 mean、population-std、first、last，固定 **256 维**。PCA 不重拟合。

attention 沿既有 Q/K 重建口径，在“读入当前引用 token 后”的 query 上计算。prompt 字符坐标映射出 passage1/2/3、claim、其他指令和 quote-so-far，区域须互斥。对 token `t`、层 `l`、头 `h`，`R[t,l,h] = mean(attn to all passage keys) / (mean(attn to all passage keys)+mean(attn to quote keys through t))`；空池不允许。每 claim 保存 `R` 沿 token 的 mean/min/max 三个 32×32 汇总，以及三篇 passage 与 claim 的 attention **mass**（区域 key 上求和）沿 token/head 的均值。

现有 Q/K hook 只允许 `past_key_value is None`。P3 停止后必须把 prompt 与已生成 token IDs 组成完整序列，再做一次 batch=1、`use_cache=False`、`past_key_value=None` 的 full replay；hidden 与 attention 只取该 replay，禁止在 cached decode 上挂现有 hook。P2 机械 quote 及 relation 分支也各自做整段 no-cache full forward。预注册 8-claim smoke oracle：cached 与 replay 在每个生成位置的 argmax token ID 必须完全一致，所选 token logprob、entropy、signed margin 分别满足 `np.allclose(rtol=5e-3, atol=5e-3)`；最短/最长 claim 的重复 replay 所导出 hidden64/attention 数组满足 `np.allclose(rtol=0, atol=1e-6)`。任一 ID 不同、非有限值或超容差即停止，不放宽容差、不改数值来源。

最终 attention **256 列及顺序**固定为：每层 `mean_t,h R`（32）；每层 `min_t(mean_h R)`（32）；每层 `max_t(mean_h R)`（32）；每层 `std_h(mean_t R)`（32）；每篇每层 `mean_t,h passage_mass`（3×32=96）；每层 `mean_t,h claim_mass`（32）。所有 std 为 population-std，不试其他聚合。

因此 P1 恰为 21 维；P2/P3 恰为 `21 + 11 token统计 + 5关系读出 + 256 hidden + 256 attention = 549` 维。无效引用仍使用实际已生成内容；只有 `T=0` 时 token统计、hidden、attention 三块全填 0。无 exact 时依前述规则填坐标/BM25 相关列；关系 5 列仍实际计算。任何非有限模型值使该 claim/整批失败，不以 0 掩盖。除既有 21 个状态/关系列外，不追加 missing indicator。

必须做两类执行自检：固定 8 个、按 prompt 长度分位选出的 label-free claim 检查显存/吞吐；其中最短和最长各重复一次，要求生成 token IDs 完全相同、浮点有限且白盒 replay 误差在预注册容差内。自检失败只报告并停止，不换 prompt、长度、精度或样本。

## 5. 四个固定条件

| ID | 条件 | 输入特征 |
|---|---|---|
| P0 | 原 probe | 从 raw `lb_prefix_pre_header.npy` 读取原答案 1,024维 4-BPE 窗均值；按冻结 index 只取本 pilot 行，固定 `C=1e-4` 五折重拟合。每个 inner/outer 模型均另拟合训练侧 scaler，禁止旧标准化缓存和旧分数。 |
| P1 | 只用机械 quote | 按已哈希的 label-blind 句切分/BM25代码（k1=1.2, b=0.75）先取每篇 top1，再取全局最大；tie 依次取 passage_id、sentence_id、text SHA 最小。只用上述 21 项机械关系/表面特征。 |
| P2 | 机械 quote + whitebox | P1 同一句在唯一 prompt 下 teacher-force，加 logprob、relation logits、hidden64 和固定 256维 attention 视图。 |
| P3 | 强制生成 quote + whitebox | 唯一候选：贪心生成引用的 21 项关系/表面特征及同轨迹 whitebox。无效引用保留，数值填 0 并用状态位区分。 |

P0 lineage 固定为：raw 矩阵 `fit_expansion/llama_baselines_v1/matrices/lb_prefix_pre_header.npy`（SHA256 `f8e99009ab045781911035188b41eeb07090e8fdec0669286ac84f061f251fc0`，float32 `[696220,1024]`；fit 前 653,979 行）、token index `fit_expansion/llama_baselines_v1/token_index.json`（`fb5af32803033f9cad1825aaf7a4971f8effb219d0b4087664483816386fe1fe`）、window index `fit_expansion/data/windows_k4_fit.jsonl`（`cfaf5af088eaac422ee2685c9150a774171d930666426e936f86d0cee905eec3`）、answer index `fit_expansion/data/answers_fit.jsonl`（`8adef0c27d5021acdf559c1566db2d6d949ecccb88ad2532ae626381add0aa33`）及生成代码 `fit_expansion/run_llama_baselines.py`（`d338b15b64ee7768d8bfd1ed576dddb602b38357169c6637ef6a7be26dcc3f91`）。这些哈希由 `llama_baselines_v1/preparation_complete.json`（`84d211dd085d18c40b0be537da0d53d457d643a06dabec89a62c41cd5739e252`）锁定。明确禁止读取 `prefix_pre_header_fit_standardized.npy`（`d20c06f4bb3894d8e1555c5cc7341a22a33160b0214f8fe39cc52c1f358a995f`）、旧模型或旧 score；P0 只借 raw 矩阵与 frozen index，在每个训练子集内拟合 scaler、类别平衡和 LR。

不新增 surface-only、融合权重、prompt 变体、解码变体或第二个分类器。P1/P2/P3 使用同一个 weighted StandardScaler 和 L2 logistic regression 规范：`C=0.001`、liblinear、`max_iter=2000`、seed `20261023`；P0 固定 `C=1e-4`。所有拟合都遵守下一节的 inner/outer 局部口径。

## 6. 4-BPE 与答案输出

P1–P3 先产生每个原子 claim 的风险概率。映射完全无参数：

```text
window_score(response, w) = max claim_score(c)
    over c whose raw claim [start,end) owns a lexical BPE in window w
uncovered eligible window = 0
answer_score = max over all eligible 4-BPE windows
```

不按引用篇号、exact 状态或答案位置另设 gate/权重。P0 直接给窗口概率；所有条件沿相同 4 raw-BPE、stride 1、原 gold 坐标和 answer=max 几何评测。

五个外折各预测一次。对外折 `o`，`outer_train` 是其余四折；对每个 `j ∈ outer_train`，`inner_train` 恰为再除去 `j` 的三折，`j` 是 inner-valid。每个 inner 模型都只用自己的三折重算：训练行、每答总权重 1（P1–P3 均分 claims；P0 均分 eligible windows）的基础权重、按基础权重拟合的 StandardScaler、训练侧两类质量 `m0,m1` 及类别因子 `(m0+m1)/(2*m_y)`，LR sample weight 为基础权重乘类别因子；任何训练子集缺一类即协议失败，不设 fallback。四个 inner-valid 分数拼成 inner-OOF 后，候选阈值固定为 `np.nextafter(max_score,+inf)` 加全部 distinct score；判正固定 `score >= threshold`。window-F1 固定 `2TP/(N_pred+N_pos)`（分母 0 时为 0），precision 固定 `TP/N_pred`（`N_pred=0` 时为 0），按 `(F1, precision, threshold)` 最大选择。

随后只用完整 `outer_train` 四折按同一公式重拟合 scaler、权重、类别因子和 LR，把已选阈值应用于外折；answer=max window，并复用该折阈值。禁止看外折标签选阈值。AP 固定为 scikit-learn 1.6.1 `average_precision_score`、正类=risk；无正例折 AP=0.0、全正例折 AP=1.0，训练子集单类仍直接失败。所有主指标不加权：每个 eligible window/answer 各计 1；pooled AP 用 raw outer-OOF 概率，pooled F1 用各折阈值所得二值预测。主指标为 pooled outer-OOF window AP/F1 与 answer AP/F1；再固定报告 EC/SC/EBI/SBI recall/AP、exact/无效引用率和五折差值。样本冻结后才允许 evaluator 进程载入 fit gold。

预注册推进条件：`exact_rate = sum(parse_valid AND stop_close AND source_exact_substring) / 3776 >= 0.95`，所有 invalid 固定计失败且分母不得缩小；P3 window AP 同时至少为 `P0+0.01`、`P2+0.01`，且 P3−P2 的逐折 window AP 在至少 4/5 折为正；`P3 answer AP >= P2 answer AP - 0.01`。若 pilot 中 EC+SC 正窗少于 50 或正答少于 10，冲突分项记 N/A，不能据此宣称冲突改善。未达任一条件即停止，不在这 256 答上调 prompt、C、特征、阈值或融合。

## 7. 资源估算

标签盲 CPU 统计：3,776 prompts 共 1,827,311 输入 token，均值 483.93、P95 687、最大 799。机械最佳句连 `</quote>` 共 124,455 token，均值 32.96、P95 82、最大 148；它只作为生成长度代理，真实生成最多 604,160 token。加入 P3 生成后的 full no-cache replay 后，两种 whitebox 条件、relation 和 replay 的预期总计算代理为 329,524,320，相当于既有 793 答实测代理的 2.062 倍。

既有五-LB 路径 159,781,000 代理量实测 1,566 秒。考虑本方案有 3,776 个小序列及逐 token 解码，按 1.5–3.0 倍调度开销，预计 **81–162 GPU 分钟**；若 P3 全部撞到 160 token 上限并 replay，规划上界约 **242–483 分钟**。这是资源估计，不是新路径实测；执行前 8-claim smoke 只校验资源，不允许据此改科学参数。

按每 token 保存 ID/logprob/entropy/margin/hidden64，并按 claim 保存完整 head attention 汇总，预计新增约 0.18 GB，硬上限低于 0.30 GB；预留 0.75 GiB。本地 FP16 权重文件约 13.48 GB 已存在，不下载新模型。NF4 单卡峰值显存尚未在该生成路径实测，故在 smoke 前记 N/A。CPU 物化、五折 LR 与审计预计 5–20 分钟。

## 8. 防泄漏硬门

- 样本 ID、fold、prompt、停止规则、四条件、特征列、C、映射、指标和推进门先冻结；gold 只在特征 manifest 完成后由独立 evaluator 打开。
- GPU manifest 不含 `labels`、`gold_label`、`risk_*`、`label_type`、质量、答案生成器或已有风险分数；运行时对这些键和 `cal|test|official_test` 路径 fail-closed。
- 不调用网络、外部 LLM、MiniCheck/NLI、RLSeek checkpoint 或人工改写。已冻结原子 claim 只读，不重新分解。
- calibration 与 official test 保持未读；任何结果只是 fit-only 探索证据，不是新 baseline 成绩，更不是 RLSeek 复现。
