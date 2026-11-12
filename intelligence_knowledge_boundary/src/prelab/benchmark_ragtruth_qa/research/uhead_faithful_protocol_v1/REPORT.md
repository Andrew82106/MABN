# UHead 忠实迁移与局部注意力探针协议 v1

日期：2026-09-12  
状态：研究与 CPU 协议已冻结；未运行模型、未开 GPU、未训练、未产生分数。  
触发条件：仅在 `exact-subset` 候选按其冻结规则失败后，才启动本文候选。

## 结论

UHead 的论文、MIT 源码和若干预训练 head 均已公开，所以**作者结构可以核对，匹配作者骨干的 checkpoint 也可以条件式执行**。但在本项目当前固定的 Llama-2-7B-chat NF4 重放设置下，**不能声称已具备“官方原样 UHead baseline”**：作者没有发布 Llama-2 checkpoint 或 Llama-2 最优配方；论文、checkpoint-era 代码和当前主分支在概率变换及局部注意力索引上也不完全一致；论文使用的原子 claim 提取链没有冻结精确依赖版本。缺项必须记为 `N/A`，不得用本地选择补成作者设置。

因此冻结两种身份：

1. **正式作者 baseline 候选**：匹配作者 Llama-3.1-8B checkpoint、原子 claim 粒度、checkpoint-era 特征语义和原 head 权重。只有全部依赖和严格对齐通过时才可称“官方 checkpoint 迁移”；当前结果为 `N/A / not run`。
2. **我们的单一候选**：`RawLocalLogP-K2-MLP-v1`，在固定 Llama-2 重放中读取两格未汇总局部注意力和实际 token 的条件 log-prob，直接预测固定 4-BPE 窗口。它受 UHead 启发，但不是 UHead baseline。

把每个 4-BPE 窗直接当成 UHead claim，会改变作者的原子 claim 单位、claim marker、全序列 Transformer 和 claim 内池化；这种做法只能列为我们的方法。按训练标签选 128 个 attention heads 也不是无参适配器，不能放进正式 baseline。

## 研究边界与本地绑定

本轮只读取了论文、公开代码、公开模型元数据、本项目协议/源码和汇总清单。没有打开 fit/calibration/test 的回答正文、逐行标签或分数数组；没有读取封存 test；没有调用 CUDA。以下本地汇总来自既有冻结文档：

| split | answers | source-connected groups | raw BPE | eligible 4-BPE windows | positive answers | positive windows |
|---|---:|---:|---:|---:|---:|---:|
| fit | 634 | 615 | 170,361 | 168,123 | 328 | 21,477 |
| calibration | 159 | 154 | 42,798 | 42,241 | 100 | 5,984 |
| test | sealed | sealed | sealed | sealed | sealed | sealed |

本轮所见文件 SHA-256：

| 文件 | SHA-256 |
|---|---|
| [`BASELINE_PROTOCOL.md`](../../BASELINE_PROTOCOL.md) | `777e2f196ea52a14bca452cdc58162d012b51e3d5a3a793f88ffc7838bca0377` |
| [`FORMAL_BASELINE_FREEZE.md`](../../FORMAL_BASELINE_FREEZE.md) | `4b7413449fc81f765ff046285ccd27b0e3437ef955cb73cf95e646f38cd38bb6` |
| [`src/feature_qa.py`](../../src/feature_qa.py) | `2b5c4011b9fea11c6888fc72e0930068ec42ce8a1ad79d2b7e4a8c6d27d78d73` |
| [`FEATURE_DEFINITION_AUDIT_QA.json`](../../results/development_v1/FEATURE_DEFINITION_AUDIT_QA.json) | `27c9144507164ae76a42583ff2654ec37e703c72d7afec1ac07bacc162ae1ee9` |
| [`data/feature_signature.json`](../../data/feature_signature.json) | `131815b0e3b53a978c9a462ff289e6b95ab8e56dc7921d925538518691ee9f09` |

固定本地骨干是 `NousResearch/Llama-2-7b-chat-hf@351844e75ed0bcbbe3f10671b3c808d2b83894ee`，Llama 2 Community License；重放为 NF4、BF16 compute、teacher forcing。完整输入是 `<s>[INST] ` + released prompt（含检索资料）+ ` [/INST] ` + 原回答，不截断，offset 从完整字符串 tokenizer 得到。现有缓存只有 Lookback/NLL/hidden 等派生量，**没有逐层逐头 raw local attention**，所以无法从已有 Lookback 比率反演本文新特征。

## 作者方法的可核对事实

主要来源为 [EMNLP 2025 论文](https://aclanthology.org/2025.emnlp-main.1809/)、[官方仓库](https://github.com/IINemo/llm-uncertainty-head)、[作者 Hugging Face 组织](https://huggingface.co/llm-uncertainty-head) 和 [公开训练数据页](https://huggingface.co/datasets/llm-uncertainty-head/train_akimbio_mistral)。仓库研究时 HEAD 为 `fc213edc26526b446e76e8253f7fe3b964a54be7`；最早公开实现为 `93b23972a4f814a1a4a618386b35368a10004e79`，本文用 `3d4202c035167f9f40447f386634ad460fcf2786` 核对 checkpoint-era 特征与 head。

### 原生任务、输入和粒度

- 原生输出单位是**原子事实 claim**，不是 token，也不是固定窗口。作者用 GPT-4o 从生成回答抽取 atomic claims，再判为 supported / unsupported / unknown；主指标是 claim-level PR-AUC，unsupported/hallucination 为风险方向。
- 训练问题主要来自 biography 等作者数据；公开 Mistral 训练集有 3,300 个回答、68,241 个 claims，Gemma 有 3,300 个回答、83,716 个 claims。作者没有把 RAG 三篇资料作为独立证据编码器；UHead 只读取骨干对完整 prompt + answer 的内部量。
- 公开数据 schema 含 `question`、`answer`、`input_ids`、`reply`、`claims[].aligned_token_ids`、`verified`、`uncertainty_labels`。公开 Mistral 数据 split 为 train 3,300、eval 184、test 100。

### token 特征

对生成目标 token `y_t`，作者保留所有指定层和 heads 的局部注意力，不做层/头汇总，并拼接 next-token 分布最高 `m` 项的值：

- 论文写作 `log(top-m probability)`；公开 `token_probabilities.py` 实际执行 `softmax(...).topk(m)`，没有取 log，也没有保留 top-token identity。发布 checkpoint 的 config 使用 `m=4`。执行作者 checkpoint 时必须沿用它训练时的概率尺度；目前最直接证据是公开代码的 probability 值，不能擅自改成 log 后仍称同一 checkpoint。
- 论文把局部窗口 `k` 搜索在 1–5 附近，并报告 raw attention + probability 优于 raw attention；Mistral biography dev PR-AUC 从 raw attention 的 `.617` 到二者合用的 `.642`。
- checkpoint-era 训练代码先把 full-forward attention/logits 去掉最后一行，再把 claim mask 去掉 BOS。于是目标 token 在绝对位置 `p` 时，对齐的 feature row 来自**预测它的 query `p-1`**；旧 `basic_attention.py` 的第 `j` 格读取 key `p-2-j`。这是一种 pre-read、排除 query 自身的语义。
- 2025-10-28 的提交 `a49dcb8841a871887fa64e33cb325b041b9809db` 重写索引，第一格改为当前 query 对自身/对角线。当前 HEAD 延续新语义。它和旧 checkpoint-era 代码不同，不能混用而不披露。

### head 结构

公开 `uncertainty_head_claim.py` 的实际结构为：

1. 每个 token 的拼接特征经过 `Linear(d_in, 2d) → LayerNorm → GELU → Dropout → Linear(2d, d) → LayerNorm → GELU`。
2. 对每个 claim，给整段 prompt + answer 的 token 加二值 claim/non-claim embedding；同一回答有几个 claims，就重复几次整段序列。
3. 整段序列进入 1–2 层 PyTorch `TransformerEncoder`。旧实现未传 causal mask，故是离线全序列 self-attention；默认 FFN 宽度为 2,048。源码中的位置 embedding 被注释掉。
4. 在 claim mask 内做 masked mean，再经 `Linear(d,d) → LayerNorm → GELU → Dropout → Linear(d,1)`；sigmoid 后为 claim 风险。
5. 冻结原 LLM，只训练 UHead；损失为 `BCEWithLogitsLoss(pos_weight=...)`。

### 论文训练配方和超参

论文写明 Adam、线性学习率衰减与 warmup，并用 W&B Bayesian optimization 按 biography validation claim PR-AUC 选参。搜索空间为：

- positive weight `{1,3,4,5}`；learning rate `{1e-5,3e-5,5e-5,1e-4,2e-4,5e-4,1e-2}`；epochs `2..15`；warmup `{0,.05,.1}`；weight decay `{0,.01,.1}`；dropout `{0,.05,.1,.2}`；
- attention history `{1,2,3,4,5,7,10}`；UHead layers `{1,2}`；head dimension `{128,256,512,768,1024}`；internal attention heads `{8,16}`。

论文只公开三组“notable optimal values”：

| 骨干 | learning rate | epochs | local history `k` |
|---|---:|---:|---:|
| Gemma-2-9B | `2e-4` | 6 | 2 |
| Mistral-7B-Instruct-v0.2 | `2e-4` | 7 | 2 |
| Llama-3.1-8B-Instruct | `1e-4` | 6 | 5 |

论文主结果对应的 batch size、seed、最终 positive weight、warmup、weight decay，以及各骨干完整结构组合均未完整公开，记为 `N/A`。当前仓库示例 config 的 `k=3, m=4, d=768, layers=2, heads=8, dropout=.1, pos_weight=3, epochs=10, lr=1e-4, warmup=.05, wd=.1, batch=32, grad_acc=1, max_grad_norm=1` 只是**当前示例**，与论文最优表不一致，不得回填为论文主实验参数。runner 没有显式冻结 `optim` 和完整依赖版本，实际 Transformers 默认 optimizer 也不能替代论文的精确记录。

论文训练使用 8 张 NVIDIA RTX 5880 Ada；单模型超参搜索约 150 GPU 小时。这说明本文当前“CPU 协议”只能做静态审计和小张量验算，不能假装复现作者训练。

## checkpoint 与复现缺口

作者公开的最清楚 checkpoint 是：

- `llm-uncertainty-head/uhead_claim_Llama-3.1-8B-Instruct`
- immutable revision：`4b5f22b6ce047dbb18e308dea90d31cd51d02d1a`
- config：all layers/heads、`k=5`、top-4 probabilities、`d=768`、2 Transformer layers、8 internal heads、dropout `.1`
- `weights.pth`：82,731,233 bytes，LFS SHA-256 `872b773e004bc83c814f3e2214c0c2f4ec90fd9d742be3a42f630ec875297e77`

Mistral 有至少两个匿名 `exp5/exp6` claim checkpoints；二者都为 `k=2, top-4, d=768`，但一个是 1 layer/8 heads/dropout `.1`，另一个是 1 layer/16 heads/dropout `.2`。公开材料没有说明哪一个生成论文主表数字，所以 Mistral 主结果的 checkpoint identity 为 `N/A`。作者 README 示例还引用了当前不存在或已改名的 `uhead_Mistral-7B-Instruct-v0.2`。

端到端忠实性缺口如下：

| 项目 | 可确认 | 正式处理 |
|---|---|---|
| Llama-3.1 head 架构/权重 | 有 immutable config 与 weight hash | 可条件式执行 |
| 本地 Llama-2 head | 作者未发布 | `N/A`；不得换骨干加载权重 |
| Llama-2 最优 `k`/LR/epoch | 未发布 | `N/A`；`k=2` 不是作者 Llama 设置 |
| atomic claim 提取 | 论文给方法，代码调用 `lm-polygraph@dev` | 精确 commit/model snapshot 未冻结，端到端逐值复现 `N/A` |
| probability 变换 | 论文为 log(prob)，代码/checkpoint path 为 prob | checkpoint 执行锁代码；差异单列 |
| attention 首格 | 旧代码排除 query，自 2025-10-28 起包含 query/self | 锁旧语义；两者不可混合 |
| Mistral 主表 checkpoint | 两个匿名候选 | `N/A` |
| paper optimizer/batch/seed 等 | 部分缺失 | `N/A`，不猜 |
| head weights license | HF repos 无 model card/license tag | `N/A / unstated` |

## 正式 baseline 的允许形式

### A. 官方 checkpoint 迁移

正式名：`UHead-official-Llama31-claim`。

只有同时满足以下条件才能列为官方 checkpoint baseline：

1. 精确使用 Llama-3.1-8B-Instruct 对应 revision 和上述 head revision/weight hash，不换成 Llama-2。
2. 锁定能与该权重兼容的 checkpoint-era feature indexing；top-4 使用 checkpoint 代码的 probability 尺度。
3. 使用作者 atomic-claim extractor 和其 `aligned_token_ids`；不把 4-BPE 窗、项目金标 span 或本地 microclaim 当作作者 claim。
4. 原生输出保持每个 claim 一个 sigmoid 风险；统一 4-BPE/answer 输出只经过下一节的无参映射。
5. 完整记录骨干、tokenizer、chat template、claim extractor、依赖和源码 hashes。任一项无法锁定或任一回答无法精确对齐，则整项正式结果记 `N/A`，不做模糊修补或静默丢样本。

本项目答案并非该骨干原生生成，且 prompt 含 RAG passages；即使满足以上条件，它仍是对固定答案的 teacher-forced、跨生成器/跨域 checkpoint 迁移，不是论文数值复现。若统一实验硬性要求全部方法使用本地 Llama-2 特征骨干，则此项直接为 `N/A`。

### B. 论文结构迁移

用本地 Llama-2 从头训练 atomic-claim UHead，最多称 `UHead-paper-structure-transfer`。由于 Llama-2 checkpoint、完整最优超参和完全匹配的 claim 标签都缺失，任何补充的 `k`、结构、训练权重或 claim 生成选择都必须列为本地选择。它不能占“官方原样 baseline”一行。

## 无参数、标签独立的统一输出映射

本映射只适用于已合法得到的作者 atomic-claim 风险 `u_c ∈ [0,1]`。它在训练、阈值选择和查看任何 gold label 之前冻结。

1. 对 exact original response string 同时取得作者骨干 tokenizer 和项目固定 Llama-2 tokenizer 的 offset mapping；不做 Unicode 归一化、decode/re-tokenize 或 fuzzy matching。
2. 把作者 `aligned_token_ids` 还原为每个 claim 的 answer-character interval union。特殊 token 没有字符区间，不参与覆盖。
3. 项目 4-BPE 窗 `w` 保持原定义。若 claim 的字符 union 与窗内至少一个项目 token 的**字母或数字字符**相交，则称 `c` 覆盖 `w`。只在空格/标点上相交不算覆盖，避免 leading-space token 把分数外溢到相邻窗。
4. `s_w = max({u_c : c covers w} ∪ {0})`。重叠 claims 取最大风险；无 claim 覆盖的窗口为 0。
5. `s_answer = max({s_w : w is eligible} ∪ {0})`。这与项目固定的 answer=max-window 聚合一致。

适配器只读取 response text、两个 tokenizer 的 offsets、作者 claim masks 和 `u_c`。它不能读取窗口/答案标签、source group、阈值、错误类型或已有模型分数，也不做学习、平滑、广播或分数融合。任何 offset 越界、原文 hash 不同、special-token 映射不唯一、claim 空 mask，都会让**整个正式 run fail closed**；不能只删掉问题样本。作者原生 claim PR-AUC 另列，不替代统一窗口与 answer 指标。

统一评测继续遵守 [`BASELINE_PROTOCOL.md`](../../BASELINE_PROTOCOL.md)：4 raw BPE、stride 1、固定 eligible-window 规则；窗口与 answer 分别报告 F1/AUROC/AP，calibration 上各自选 F1-opt threshold，tie-break 为 precision 更高、再 threshold 更高；冻结后才能用于 test。本报告不读取或计算 calibration/test。

## 我们的唯一轻量候选

正式名：`RawLocalLogP-K2-MLP-v1`。身份：**ours / UHead-inspired local window probe**。

### 特征

对固定答案 token `y_t` 的完整序列绝对位置 `p_t`：

```text
a_t = vec_{lag,layer,head}(
    A[layer, head, query=p_t-1, key=p_t-2],
    A[layer, head, query=p_t-1, key=p_t-3]
)
ell_t = log P(y_t | fixed prompt, y_<t)
x_t = concat(a_t, ell_t)
```

- 使用 Llama-2 的全部 32 层 × 32 query heads，固定两个 lag，flatten order 为 `[lag, layer, head]`，共 2,048 个 raw attention 值；不池化、不选 heads。绝对 predecessor 不存在时补 0。
- `ell_t` 是实际观测 token 的 selected-token log-prob，等于现有 exact NLL 的相反数；它不是作者 top-4 sorted probability vector。采用这一维是为了更轻且与目标 token 精确对齐，所以明确属于 ours。
- query 使用 `p_t-1`，与 checkpoint-era UHead 的 pre-read 时序一致；不使用本地现有 Lookback 的 post-read `p_t` query，也不包含 attention diagonal/self。
- 不读取 hidden state、Lookback 比率、NLI、claim、evidence encoder或旧候选分数。每个 token 特征自身严格 pre-read；窗口 readout 只看该窗的四个有序 token，不看窗外 future token。

### 固定模型

```text
per-token: Linear(2049,64) -> LayerNorm(64) -> GELU -> Dropout(0.1)
window:    concat(z_t, z_t+1, z_t+2, z_t+3)             # 256 dims
readout:   Linear(256,64) -> GELU -> Dropout(0.1) -> Linear(64,1)
risk:      sigmoid(logit)
```

总可学习参数必须恰为 **147,841**；FP32 约 0.56 MiB。窗口严格是项目已冻结的连续 4 raw BPE、stride 1；不增加 lexical mask feature，lexical eligibility 只决定是否进入评测/损失。answer 风险固定为最大 eligible-window risk。

### 固定训练

- 只用 fit；五折 `GroupKFold`，不可拆分 source-connected group。输入 row order 和每个 group 的 held fold 在 prepare 时写入并 hash；每个 fit window 只得到排除其 group 的一次 OOF 分数。
- fold 训练权重：每个 source-connected group 总质量相等；group 内每个 answer 相等；answer 内每个 eligible window 相等。再乘训练 fold 内 `negative_mass / positive_mass` 的正类因子，最后归一到该 fold window 数。held fold 标签不参与该因子。
- PyTorch CPU head 为 FP32；seed `20260912`；deterministic algorithms；AdamW `lr=1e-3`、`weight_decay=1e-2`；batch `4096 windows`；固定 10 epochs；每 epoch 用 seed+epoch 的 CPU generator 打乱；gradient clip `1.0`；weighted BCE-with-logits；不早停、不调参、不跑第二架构。
- 先产出五折 fit OOF；窗口与 answer 各自在 fit OOF 上按共同 tie-break 选阈值，仅用于这个门控。只有当它在同一 fit OOF 协议下窗口 F1 和 answer F1 都不低于冻结 incumbent，且至少一项严格更高，才允许训练 10-epoch full-fit 并进入一次 calibration。若未过门，停止；不因结果改 `k`、宽度、seed、loss 或聚合。
- calibration 只在模型与 source hashes 冻结后运行一次，按共同协议报告；不能再回改候选。test 继续封存。

raw-attention 缓存预计：fit 约 `170,361 × 2,048 × fp16 = 0.650 GiB`；fit+cal 约 `0.813 GiB`。不物化重叠窗口大矩阵，只按 token index gather 四个向量。缓存值可用 FP16 存储，CPU 训练加载后转 FP32；`ell_t` 存 FP32。

## 可审计 CPU 执行计划

当前阶段只允许执行第 0–2 步；第 3 步需要将来另行启动 GPU 特征提取，因此目前状态是 `blocked_pending_raw_attention_cache`，不是实验失败。

### 0. 来源冻结

输出 `SOURCE_LOCK.json`，记录：论文 DOI/URL；GitHub `3d4202c...`、语义漂移提交 `a49dcb...` 和研究 HEAD；HF repo/revision/config/weight SHA；本地五个文件 SHA；Python/PyTorch/Transformers/sklearn 版本；许可证状态。检查实际文件 hash 与本报告完全一致。

### 1. dataset-free CPU oracle

只构造 `B=2, layers=3, heads=2, T=7, vocab=11` 的带唯一数值小张量，不读取任何项目行：

- 断言目标 `p` 的 attention 首两格恰为 `[p-1,p-2]` query-key 坐标中的 `A[p-1,p-2]`、`A[p-1,p-3]`，并验证早期越界为 0；
- 断言 flatten order 是 `[lag,layer,head]`，维度在真实配置为 2,048；
- 用显式 `logsumexp` oracle 验证 `ell_t = logits[p-1,y_t] - logsumexp(logits[p-1])`，并验证其等于 `-NLL`；
- 断言四 token 顺序不变、相邻窗口索引正确、answer=max-window；
- CPU forward/backward 一次，检查参数数精确为 147,841、所有 gradient/value finite、同 seed 重跑 bitwise identical；
- 在脚本开头断言不把任何 tensor 放到 CUDA，产出 `CPU_ORACLE.json` 和源码 SHA。

### 2. manifest-only prepare 审计

只读取冻结 metadata，不读取回答/标签内容：核对 fit/cal group 不交叉、计数、模型 revision、prompt wrapper、token/offset 规则和已有 feature keys。必须明确断言现有缓存没有 raw attention；若缺失则正常输出 `blocked_pending_raw_attention_cache=true`，不能用 Lookback 比率、均值或随机数填充。生成未来缓存 schema：

```text
answer_id, exact_text_sha256, model_revision, tokenizer_revision,
raw_token_ids, raw_token_offsets,
attn_k2_fp16[num_tokens,2,32,32],
selected_logprob_fp32[num_tokens], finite_mask
```

### 3. 未来 GPU 提取（本轮禁止）

只有 `exact-subset` 明确失败且另行启动后才执行。固定 Llama-2 revision/NF4/BF16/full prompt/teacher forcing；只提取 query `p-1` 对 keys `p-2,p-3`，不请求或保存完整 `T×T` attention。每个 answer 写临时文件、fsync、hash 后原子 rename。至少抽一个**只来自 fit**的短回答，用 CPU eager 小范围重算作坐标 oracle；任何 token id、offset、logprob、attention 或 hash 不一致即全 run 失败。cal/test 在此阶段仍不提取。

### 4. CPU 五折训练（未来 cache 就绪后）

只打开 fit features/labels，冻结 `protocol.json`、fold assignment、weights 与源 SHA；按上述唯一配置跑 5 个 folds，保存每 fold 初始/最终 state hash、epoch loss、held indices hash 和 OOF scores。独立 audit 重算：group 零泄漏、每窗恰一份 OOF、权重公式、parameter count、answer max、F1/AUROC/AP、threshold tie-break。先与 frozen incumbent 的同协议 fit OOF 比较，门控失败立即停止。

### 5. full-fit、calibration、test

这两步不属于本轮。过 fit gate 后才允许固定 10 epochs full-fit，随后一次性生成 cal scores；cal 只用于共同评测和阈值，不用于选择结构。最终候选和两个阈值冻结后，才可由正式 test 流程读取 test。任何提前读取 cal/test、修改架构或新增特征，都使 `v1` 失效并需新版本名。

## 许可证

- ACL Anthology 论文及 2016 年后的 ACL materials：CC BY 4.0。
- 官方 GitHub 源码：仓库 `LICENSE` 为 MIT；本轮 HEAD 的 `pyproject.toml` 也指向 MIT。
- Hugging Face head repos 没有 model card 或 license tag，权重许可证记为 **N/A / unstated**，不能自动从 GitHub MIT 推断。
- Llama-3.1、Llama-2 等骨干各自许可证独立适用；本地骨干为 Llama 2 Community License。
- 公开作者训练 dataset 未见可确认的 license 字段时同样记 `N/A`；本协议不复制或再分发它。

## 冻结判定

当前可回答“能否忠实迁移”为：**结构可核对，匹配 checkpoint 可条件式迁移；本地 Llama-2/固定 4-BPE 环境下没有可直接称原样的作者 baseline。** 正式作者 baseline 暂为 `N/A / not run`；本文唯一可执行的新模型是 `RawLocalLogP-K2-MLP-v1`，并明确归入 ours。它当前仍缺 raw local attention cache，任何数字都尚不存在。
