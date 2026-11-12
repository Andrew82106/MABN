# Forced-Evidence Quote Probe V2：数值执行预注册

状态：**FROZEN / PRE-REGISTERED；尚无 V2 runner，尚未获准启用 GPU。**  本文件只冻结 V2 的数值来源、数值容差和失败规则，不改变数据、提示、模型、四个实验条件、特征维度、分类器或评测指标。

## 1. 为什么必须另立 V2

V1 已按原协议永久停止，不能原地放宽。V1 的固定门要求 cached decode 与 full no-cache replay 的 selected log probability、全词表 entropy 和 signed margin 都满足 `np.allclose(rtol=5e-3, atol=5e-3)`。首个 label-free smoke claim 的结果是：

| 量 | V1 首条诊断 |
|---|---:|
| cached/replay argmax mismatch | `0 / 29` |
| selected log probability 最大绝对差 | `0.023988783359527588` |
| entropy 最大绝对差 | `0.027759075164794922` |
| signed margin 最大绝对差 | `0.25` |

该诊断没有读取 gold、calibration 或 official test，也没有评分。诊断文件 SHA-256 为 `fdca1a0a34446a0620e026eaeee1f9825ed1e55e10f2edf2ed9f118315c858d6`；V1 失败记录 SHA-256 为 `e3e986e1ef93d4fe4e6953b0655952bb5f02f066c820520bbdf485e45cca7184`。

V1 的离散决策完全一致，连续量却超过 `5e-3`，符合当前数值栈的性质。NF4 权重在两条路径中相同，本身不是主要路径差异源；差异来自 NF4 固定权重下的 BF16 activation/`lm_head`，以及 cached decode 与 full replay 的不同张量形状和 SDPA/GEMM 归约顺序。把 BF16 logits 转成 float32 后再做 softmax 能避免继续以 BF16 归约，但不能恢复进入 softmax 前已经发生的舍入差。PyTorch 也明确说明，数学等价的 batched/slice 计算不保证逐位一致，浮点归约顺序会改变结果；Hugging Face 的 KV-cache 文档同样说明 cached 与非 cached 输出可能因矩阵乘内核而略有差异。

参考：

- [PyTorch Numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)
- [PyTorch float32 matmul precision：BF16 有 7 个显式尾数位](https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html)
- [Hugging Face KV-cache optimization](https://huggingface.co/docs/transformers/v4.40.1/en/llm_tutorial_optimization)

## 2. V2 与 V1 的边界

V2 只替换 V1 的 cache/replay 数值一致性协议。以下对象继续按 V1 已冻结内容执行：

- 唯一 label-free 输入 `label_free_inputs.jsonl`，SHA-256 `ae6bf145a0e48ff310cdde5f54457eec7e8986c9b012d52cc463bb9e357e2777`；
- 3,776 个 claim、256 个 answer、原 prompt、`max_new_tokens=160`、停止和解析规则；
- 本地 `NousResearch/Llama-2-7b-chat-hf` revision `351844e75ed0bcbbe3f10671b3c808d2b83894ee`；
- NF4 double quant、BF16 compute、BF16 `lm_head`、SDPA、batch 1、TF32 off、eval/inference、local-only；
- P1 为 21 维，P2/P3 为 `21+11+5+256+256=549` 维；
- hidden PCA、attention 定义、P0 lineage、fold、分类器和最终推进门。

V1 的协议和计划分别冻结在 SHA-256 `78e5558b61592238b0979c7c60aea23c79b8c7d3181e88113d3972b4d056f7aa` 与 `6d439ee257dde65d2ef2c9974cd3dd716f4f0d06890cfb7df9c8af6da562680c`。V2 不覆盖或续写 V1 的 GPU 产物；V2 必须使用新的 runner version、结果目录、运行签名和独立审查文件。

## 3. 唯一权威数值来源

P3 的不同信号按其实际含义固定来源，不做平均、校正或二选一：

| 信号 | V2 权威来源 | 理由 |
|---|---|---|
| 生成 token ID 与停止位置 | KV-cached greedy decode | 这是实际生成路径。每步 `argmax`；完全相同时取最小 token ID。 |
| selected log probability、全词表 entropy、signed selected-vs-best-other margin | 与该 token ID 同一步的 cached logits | 这些量要表示“模型实际选择该词元时”的置信度。 |
| P3 content-token hidden64 | prompt 加 cached token IDs 后的一次完整 `use_cache=False, past_key_values=None` replay | 当前 Q/K 读取只在完整无 cache 路径上有已审计实现。 |
| P3 attention256 | 与 hidden64 同一次完整 replay | 保证 hidden 与 attention 来自同一序列、同一次前向。 |
| replay log probability、entropy、margin、argmax | 仅作数值 QA | 不进入 11 维 token 统计，不进入分类器，不决定样本去留。 |

cached 和 replay 都必须先得到 BF16 `lm_head` 输出，再把**完整词表 logits**显式转换为 float32。随后固定用 float32 `log_softmax`；selected log probability 从该结果 gather，entropy 为 `-sum(exp(logp) * logp)`，margin 为 selected logit 减去其余 token 的最大 logit。输出保存为 float32；比较时再转为 float64 做差，避免比较本身引入额外 float32 舍入。

内容 token、闭合标签排除、EOS 保存方式继续沿用 V1。cache/replay 比较覆盖本次生成的全部位置，包括闭合标签或终止 EOS；分类特征仍只汇总内容 token。

## 4. 固定容差如何得到

V2 只允许使用 V1 已经产生的**第一条 label-free 数值诊断**确定容差。对每个连续量的 V1 最大绝对差 `d`，使用同一条机械规则：

```text
tolerance(d) = 2 ** (ceil(log2(d)) + 1)
```

即把观测最大值先包入二进制幂边界，再预留一个完整的二进制 guard band。该规则不给三个信号分别手调十进制常数，也不使用标签或成绩。它给出以下一次性固定值：

| 比较量 | `rtol` | 固定 `atol` | V1 首条最大差 / 新容差 |
|---|---:|---:|---:|
| cached vs replay selected log probability | `0` | `0.0625` (`2^-4`) | `0.3838` |
| cached vs replay entropy | `0` | `0.0625` (`2^-4`) | `0.4441` |
| cached vs replay signed margin | `0` | `0.5` (`2^-1`) | `0.5` |

使用纯绝对容差是有意的：大量 log probability 和 entropy 接近 0，relative tolerance 在这些位置没有稳定含义；margin 又直接继承 BF16 logit 的二进制台阶。`0.5` 也给“两个 logits 的差”留出比单个 BF16 logit 更大的误差预算。

这三个值在 V2 的其余 smoke 执行前冻结。不得根据另外 7 条 smoke、全量提取或任何分数再次调整。

## 5. Smoke：跨路径门只在固定 label-free 样本上执行

继续使用 V1 在 GPU 运行前已经由 prompt 长度分位数固定的 8 个 label-free 索引，顺序不变：

```text
[2826, 994, 1533, 1695, 1299, 1143, 3352, 3752]
```

对应 prompt 长度为 `[255, 354, 385, 435, 508, 568, 634, 799]`。索引 2826 已用于得到上面的 V1 容差锚点，因此报告时必须标为 `tolerance_anchor`；其余 7 条是未用于定容差的 `validation_smoke`。8 条都必须运行和报告，任何一条都不能替换。索引 2826 仍要通过下列门，以防 V2 实现或运行环境相对诊断时发生变化；独立的容差验证结论只由其余 7 条提供。

每条 smoke 的硬门如下：

1. replay 的 teacher-forcing target IDs 必须逐位等于 cached 生成序列。replay 自己的 argmax 不是生成决策，因此 near-tie 翻转不自动判失败。若两路 argmax 不同，只允许同时满足：cached 路径的生成 token signed margin `<= 0.5`，且 replay 路径对该 cached token 的 signed margin `>= -0.5`。任一条件不满足即为高置信路径冲突并失败。mismatch count/rate 必须完整报告，但不另设一个从未观测数据估出的比例阈值。
2. 在全部生成位置上，以 float64 计算 `max(abs(cached - replay))`；log probability 不超过 `0.0625`，entropy 不超过 `0.0625`，signed margin 不超过 `0.5`。比较固定为 `rtol=0`，边界相等算通过。
3. cached/replay 的 logits 派生量、hidden64 和 attention 全部有限；任何 NaN 或 Inf 立即失败。
4. 最短索引 2826 与最长索引 3752 各自完整重复一次。两次 cached token IDs 必须完全相同；两次 cached 三种统计分别满足 `rtol=0, atol=1e-6`。两次 full replay 的 hidden64 与 attention 原始数组分别满足 `rtol=0, atol=1e-6`。该 `1e-6` 沿用 V1 对**同一路径重复执行**的冻结标准，并非由本次失败调得。
5. prompt IDs、生成 IDs、absolute positions、content indices、字符 offset、停止原因和解析结果在重复执行间必须完全一致。

V2 smoke 必须保存逐 claim 的 token 数、argmax mismatch 位置、每个 mismatch 两路 signed margin、三个差值的 max/mean/P50/P95、最大差位置两侧数值、显存和运行签名。独立 reviewer 必须核对 runner、数值协议、输入、模型资产、软件栈和 smoke 文件的 SHA-256 后，才能允许全量提取。

任一 smoke 硬门失败即停止 V2：不得放宽容差、改 dtype/backend、换 smoke 样本、缩短输出、删除失败 token 或重试到通过。可以保存只含 label-free 数值的诊断，但必须另立后续版本。

## 6. 全量提取：强制结构正确，不以 cache/replay 尾部极值筛数据

**推荐 smoke-only 强制跨路径容差。** 8 条分位 smoke 已用于验证实现和长度覆盖；在 3,776 个 claim、最多约 60 万生成 token 上再以 `max` 强制 cached/replay 一致，会把“是否完成整批”变成对样本量和偶发近并列 token 的函数。cached 与 replay 本来就是不同的 BF16 执行日程，而且 replay argmax 不是本研究使用的生成决策。全量中不应按这种数值尾部事件删除、重跑或选择样本。

全量每条 claim 仍必须执行并硬性检查：

- cached token IDs、停止位置和内容 token 映射严格对齐；长度在冻结范围内，无静默截断；
- cached 权威统计全部有限，`logprob <= 1e-6`、`entropy >= -1e-6`、生成 token 的 cached signed margin `>= -1e-6`；
- replay 的 hidden64、attention 及其汇总全部有限，shape 和冻结列顺序完全正确；
- replay 的 target IDs 必须逐位等于 cached 生成序列；这里检查的是 teacher-forcing 输入身份，不要求 replay 自己的 argmax 等于 target；
- 所有文件、运行签名、软件、模型、PCA、prompt 与 input hash 匹配；原子写入和 resume 校验通过；
- 不得因为 parse-invalid、非 exact quote、cache/replay 漂移大小或 replay argmax mismatch 而删行、补零、改写或重试。

全量每条 claim **必须**保存下列描述性、非门禁摘要：argmax mismatch count/rate，以及三种 cached/replay 绝对差的 max、mean、P50、P95、P99；整批完成后还必须按全部生成 token 汇总同一组指标。原始逐 token replay QA 数组可以不长期保存，但摘要及其计算代码/hash 必须进入 feature-freeze manifest。所有这些数值不进入特征、不用于过滤、不触发容差修改，也不能在看标签后升级成排除规则。

全量的硬失败只包括：非有限权威特征、shape/ID/offset/hash 不一致、运行栈漂移、OOM 或原子文件损坏。发生硬失败时整批保持未完成状态，可在相同冻结环境从最后一个完整原子记录恢复；不得读取 gold 后再决定如何处理。

## 7. 防泄漏与版本门

- V2 numerical smoke 和全量 GPU 提取的唯一数据输入仍是哈希锁定的 label-free manifest；运行时对 `label`、`gold`、`risk`、已有分数及 `cal|test|official_test` 路径 fail closed。
- V2 特征全部冻结、完成独立数值审查以前，不得启动 evaluator，不得打开 fit gold；calibration 与 official test 继续封存。
- V1 的 smoke、失败和诊断仅作为公开 provenance；不得把 V1 首条输出混入 V2 正式 claim cache。
- V2 runner 必须在 GPU 前通过 CPU selfcheck 和独立静态审查。GPU smoke PASS 后还需单独的 smoke 独立审查，才可进入 full extraction。
- 数值诊断永远不成为模型输入、训练特征或标签。

## 8. 明确推荐

采用以下 V2：**cached greedy 决定 token 并提供 token-level log probability/entropy/margin；同一固定 token 序列的 full no-cache replay 只提供 hidden/attention；跨路径连续量的 `0.0625 / 0.0625 / 0.5` 容差与高置信 argmax 冲突检查只在固定 8 条 label-free smoke 上作为硬门；全量只强制权威信号的有限性、shape、ID 和 provenance，并完整报告跨路径漂移而不据此筛样本。**

这保留了“生成时置信度”的真实含义，也承认 NF4+BF16 下两种执行日程不会满足 FP32 风格的逐位近似。若新 runner 在剩余 7 条 validation smoke 中仍失败，应停止 V2，下一版本改为同一路径的 cache-aware hidden/attention 读取或更高精度执行；不能继续扩大容差。
