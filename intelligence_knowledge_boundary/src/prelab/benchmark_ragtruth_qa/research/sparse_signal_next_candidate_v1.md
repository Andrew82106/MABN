# 下一候选设计审计：Atomic Evidence Router v1

更新：2026-09-12。目标仍是统一 4-BPE、stride-1 窗口 F1 ≥ 0.75；official test 封存，正式 baseline 不改。

## 结论

下一候选应以**原子微主张**为训练单位，而不是继续把 65 万个高度重叠窗口当独立样本。模型先判断一个微主张是否被资料支持，再把分数按现有字符/BPE 坐标映射回 4-BPE 窗口。现有微主张几何在 calibration 的金标窗口上限为 0.889351，因此这种粒度没有堵死 0.75 目标。

建议方法名为 `atomic_evidence_router_v1`：

1. 关系证据视图：复用 `atomic_microclaim_nli_v1` 的 315 维无标签 evidence/relation 特征。
2. 内部状态视图：从现有隐藏状态计算 HARP bottom-205 投影，并在每个微主张内池化。
3. 生成轨迹视图：复用 NLL、GHOST、LUMINA 的逐词元缓存，在微主张内计算局部统计量。
4. 用 K=4 的 FLaG 式原型路由和 log-marginal 读出融合三类信号。

第一版不新抽 Llama 特征、不训练 SAE。它先回答“关系证据与现有白盒轨迹经过机制路由后能否互补”。若它仍低于当前 0.690281，再训练一个适配当前 Llama-2 隐状态的小 SAE；不要直接把不兼容的公开 SAE 接进来。

## 三篇工作与现有缓存的对应关系

| 方法/信号 | 当前能否直接用 | 现有证据 | 缺口或处理 |
|---|---|---|---|
| RAGLens：逐词元 SAE 激活 | **不能原样用** | 已缓存全部 708,506 个回答 BPE 的 Llama-2 final hidden | 官方示例是 Llama-3.2-1B `layers.6.mlp` 与模型/层绑定 SAE；当前缓存是 Llama-2 final RMSNorm，坐标和分布均不兼容 |
| RAGLens：max pooling、训练折内 MI 选特征、GAM | **可直接实现** | 标签、微主张边界和词元坐标均齐 | 只能称 RAGLens 式读出；没有 SAE 时不能称复现 RAGLens |
| HARP：unembedding 尾部 SVD 基 | **可直接用** | `data/harp_basis.npz` 已有 bottom-256；4096 的 5% 约为 205 | 取其中 bottom-205 即可，不需要重新 SVD |
| HARP：逐词元投影 | **可从缓存计算** | 原 793 答已有 bottom-256；扩充 3,046 答已有 final hidden | 流式投影约新增 0.55 GiB，CPU 即可；现成 bottom-64 已覆盖全部 3,839 答 |
| FLaG：回答末状态、回答均值 | **可直接计算** | final hidden 覆盖全部回答词元 | 局部版改为“微主张末状态、微主张均值” |
| FLaG：回答与 prompt 均值漂移 | **缺失** | 没有保存 prompt token hidden | 精确复现需一次新 Llama 前向；v1 不做，用 Lookback/关系证据表示资料依赖 |
| FLaG：selected-token log probability 的均值/最小/标准差、低概率比例、长度 | **可直接计算** | NLL 覆盖全部 708,506 BPE | 在微主张内计算，阈值只由 fit 训练折确定 |
| FLaG：全词表预测熵、top1-top2 logit margin | **精确值缺失** | GHOST 有 top-10 重归一熵，LUMINA 有选中词概率和最大概率 | v1 使用这些已存代理量；精确值以后与 prompt mean 合并一次抽取 |
| FLaG：原型路由、组内线性头、log-marginal | **可直接实现** | 只需缓存特征 | 原论文 K=64 面向回答级；当前仅 615 个资料组，固定 K=4，不能照搬 64 |

依据：RAGLens 官方实现先对回答 token 的 SAE 激活做 max pooling，再在训练集做互信息筛选并拟合 GAM；其 token 热区来自特征峰值解释，不是 span 标签监督。HARP 用 unembedding 尾部约 5% 奇异方向投影逐 token 隐状态，再以 token 最大分数形成回答分数。FLaG 使用隐藏几何与七个概率轨迹统计，经原型软路由和 log-marginal 汇总；原方法仍是回答级任务。来源：[RAGLens 官方代码](https://github.com/gzxiong/RAGLens)、[HARP](https://arxiv.org/abs/2509.11536)、[FLaG](https://arxiv.org/abs/2606.00301)。

## v1 固定输入

每个微主张一行：

```text
group_id, response_id, microclaim_id
answer_char_start, answer_char_end, owned_BPEs
relation_evidence[315]
harp_mean[205], harp_signed_maxabs[205]
probability_trace[*]
microclaim_label
```

`relation_evidence[315]` 直接读取 `atomic_microclaim_nli_v1` 冻结的原始特征，不先读取它的监督预测。它包含各 passage 的单句/联合 E/N/C、BM25 与覆盖、主体/谓词、否定、比较、数量、时序、条件和来源匹配。

HARP 对每个微主张保存两种池化：逐维均值，以及绝对值最大词元对应的带符号值。只取 max 会丢掉负方向，只取均值会稀释短错误，两者同时保留。

生成轨迹固定使用：

- NLL：均值、最大、最小、标准差、超过 fit 训练折固定分位点的比例；
- GHOST 四列：各自均值、最大值、标准差；
- LUMINA 七列：各自均值、最大值、标准差；
- 微主张 BPE 数和其在回答中的相对位置。

以上原始缓存均不依赖风险标签。第一版不加入当前 incumbent 分数、已训练 large/NLI/FAVA 分数，避免训练内上游分数导致新的分布错位。

## 模型

三种视图分别在当前训练折内做稳健标准化和线性投影：

```text
relation 315 -> 24
HARP 410 -> 24
trace ~40 -> 16
concat 64 -> MLP(64, 32, 16) -> r
```

固定四个可学习原型 `c_g`。路由与 FLaG 一致：

```text
pi_g = softmax(cos(r, c_g) / 0.1)
s_g  = w_g^T r + b_g
risk = logsumexp(log(pi_g) + s_g)
```

主损失：

```text
L = weighted_BCE(risk, label)
    + 0.25 * softplus(1 - risk_positive + risk_negative)
```

正负对优先在同一回答内配对，其次同一资料组；没有同组对时不强造。权重保持“资料组等权 → 回答等权 → 微主张等权 → 两类平衡”。模型约 2 万个可训练参数，远小于旧 1089 维 TCN；K=1 用作固定消融，不参与选择主候选。

## 训练、验证和映射

1. 只用 fit。按 source-connected `group_id` 做五折；同一资料、同一问题、六个生成器回答和反事实对不得跨折。
2. 标准化、NLL 低概率阈值、任何特征筛选、训练轮数判断都只能在外折训练组内部完成。若使用内部早停，再从训练组按 group 划验证；不得看外折和 calibration。
3. 每个 fit 微主张只得到一次外折预测。由这组 OOF 分数选择窗口阈值和整答阈值。
4. 全 fit 按同一固定配置重训一次，只预测 calibration。calibration 不选 K、宽度、损失权重、轮数或阈值。
5. 一个 lexical BPE 继承与其字母数字字符相交的微主张风险；一个 4-BPE 窗和整答分别取覆盖微主张风险的最大值。评测标签、标点占位、短窗规则均保持现状。
6. 报告窗口/整答 P、R、F1、AUROC、AP、TP/FP/FN，以及 Evident/Subtle Conflict、无依据类型和 170 个风险 run 的命中情况。当前候选 0.690281/0.891089 只作冻结参照。

该流程不会把 65 万个重叠窗口误当成 65 万个独立事实；真正的监督规模是 615 个资料组和约 34,941 个 fit 微主张。

## 执行顺序与停止条件

### 阶段 A：最小候选

- 先在原 634 fit + 159 calibration 上使用正在产出的 315 维微主张特征。
- 从缓存补 HARP205 和 trace，训练固定 K=4 主模型及 K=1 消融。
- 只有主模型的 fit OOF 同时改善 relation-only 与 K=1，才打开一次 calibration 报告；否则直接判定路由没有贡献。

### 阶段 B：扩充 fit

若阶段 A 有组外收益，再把相同 NLI 特征抽取扩到现有 34,941 个 fit 微主张。原 98,854 个 NLI pair 对应 11,322 个微主张；按规模线性估计，扩充版本约 30 万 pair。必须另建版本与冻结 manifest，不把两次结果混在同一目录。

### 阶段 C：RAGLens 式 SAE

只有阶段 B 仍低于 0.75 且 HARP/trace 消融显示内部状态仍有增量时，才在**fit-only final hidden**上训练本模型专用 TopK SAE：4096→8192、TopK=32、3 个固定 epoch。编码后只保存每词元 32 个非零索引和值；每个外折的 MI 特征选择只能看该折训练组。它是“RAGLens-inspired local SAE”，不是官方 RAGLens 复现。

不要下载 Llama-3.2 的公开 SAE直接处理当前 Llama-2 hidden；SAE 与模型、层和 hookpoint 绑定，这种拼接没有方法学意义。

### 阶段 D：补精确 FLaG 信号

只有 trace 代理量确有独立收益时，才在一次 Llama 重放中同时保存 prompt hidden mean、全词表熵和 top1-top2 margin。这样只增加一次抽取，不为每个缺失量分别重跑。

## 资源估计

| 工作 | GPU/显存 | 预计时间 | 说明 |
|---|---:|---:|---|
| HARP205 全量流式投影 | 无 | CPU 2–5 分钟 | 按已有 213,159 token ×256 投影 36.4 秒外推；实际受 15 GiB 压缩 hidden 读取影响 |
| trace 聚合、K=4 五折训练 | 无 | CPU 10–25 分钟 | 小模型；主要成本是流式读 708,506 token 和组折重放 |
| 当前原 793 微主张 NLI | 约 0.8 GiB | 约 25–35 分钟 | 98,854 pair；既有 35,316 pair 实测 572.5 秒，仅作线性估计 |
| 扩充约 30 万 NLI pair | 约 0.8–1.5 GiB | 约 80–110 分钟 | 长度分布变化会影响时间 |
| 本地 2× TopK SAE，3 epoch | 约 1.5–2.5 GiB | 约 20–45 分钟 | 只读缓存，不加载 7B；输出稀疏存储约 0.15–0.25 GiB |
| 精确 FLaG 缺失量一次重放 | 约 3.8–4.5 GiB | 约 25–40 分钟 | 可同时抽 prompt mean、全熵和 margin；参考 GHOST 全量重放 23.9 分钟 |

当前 D 盘约有 96 GiB 可用。阶段 A 预计新增不足 1 GiB；无需为候选复制现有 3 GiB 的 `raw1089.npy`。

## 风险判断

- 0.75 不能由结构设计保证。当前连续分数的简单非线性组合最高仍约 0.70，说明路由若没有新的关系证据，只会重新拟合旧错误。
- v1 的主要新信息来自 315 维主体—谓词—客体与限定条件证据；HARP 和 trace 主要负责区分“低概率生成、资料脱离、过度自信”等机制。
- 原子微主张标签仍是“内部任一词元有风险即整条为风险”，会把风险扩散到同一微主张的正常词。0.889351 的几何上限说明有余量，但真实模型需要同时控制 FP。
- calibration 已被反复查看，只能作为开发集。达到 0.75 后仍须冻结模型与阈值，再一次性打开 official test；否则不能写成稳定 SOTA。
