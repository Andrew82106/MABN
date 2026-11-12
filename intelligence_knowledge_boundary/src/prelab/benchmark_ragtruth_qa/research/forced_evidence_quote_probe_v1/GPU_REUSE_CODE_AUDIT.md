# forced-evidence quote probe：GPU 代码复用审计

审计对象：`PLAN.json` SHA256 `6d439ee257dde65d2ef2c9974cd3dd716f4f0d06890cfb7df9c8af6da562680c`；`PROTOCOL.md` SHA256 `78e5558b61592238b0979c7c60aea23c79b8c7d3181e88113d3972b4d056f7aa`。

结论：NF4 加载、no-cache Q/K 重建、最后层 hidden 读取、PCA64 投影、P0 raw 矩阵及原子写盘框架均有可复用实现。仓库内没有满足本协议的 cached greedy 生成器，也没有现成的 256 维 attention 聚合器；这两部分必须新写。当前 `src/run_forced_evidence_quote_probe_v1.py` 只有 CPU skeleton，GPU 方法会主动抛错。

## 可直接复用

| 需求 | 文件与符号 | 可复用合同 |
|---|---|---|
| Llama2 NF4 加载 | `src/run_exact_subset_attribution_v1.py::load_nf4`（820–864） | 加载前逐文件核对 `model_download_manifest.json`；固定 revision、NF4 double quant、BF16 compute、SDPA、`cuda:0`、local-only、TF32 off、eval、参数冻结，并断言首个 `q_proj` 确为 NF4。比单独调用 `run_feature_qa.load_nf4` 更完整。 |
| GPU 独占与清理 | 同文件 `exclusive_gpu`（780–817）、`clean_gpu`（932–935） | 复用全局锁、外来 GPU 进程检查、加载前空闲显存门和 finally 清理。进程异常退出会遗留锁，不能自动删除；须人工核对 PID 后处理。 |
| Q/K hook、RoPE、GQA | `src/feature_qa.py::architecture, rotate_half, LookbackHooks`（191–275） | 这是主复用核心；只允许无 cache 的 batch=1 full replay。 |
| hidden 与逐 token LM readout | `src/feature_qa.py::extract_features`（289–360）；`src/run_exact_subset_attribution_v1.py::selected_token_logprob`（867–889） | `model.model(...).last_hidden_state` 已是最后 RMSNorm 后状态；目标 token 由位置 `i-1` 的 hidden 经 `lm_head` 预测。LM head 按小块运行，避免保存全序列全词表 logits。 |
| PCA64 | `src/run_development.py::fit_pca/build_matrices`（138–176） | 锁定 pickle 是 dict：`mean[4096]`、`components[64,4096]` 等；既有精确公式为 `(hidden.float64-mean) @ components.T`，随后转 float32。新路径不得重拟合。 |
| P0 raw Lookback | `src/feature_lookback_controls_v2.py::ControlHooks/extract`（66–140）；`fit_expansion/run_llama_baselines.py::pool/prepare`（98–181） | token 特征 `lb_prefix_pre_header` 是 float32 `[token,1024]`，窗口 raw 矩阵已是对应 4-BPE token 的均值。P0 直接读取锁定 `[696220,1024]` 矩阵的选中 fit 行，不能再次池化。 |
| P0 训练骨架 | `fit_expansion/run_llama_baselines.py::fit`（184–220） | 仅可复用 `StandardScaler(sample_weight=...)`、liblinear L2 LR 的调用方式；旧函数用全 fit 与 calibration 选阈值，不符合新 nested OOF，不能直接运行或复用旧模型/分数/scaler。 |
| 机械引用 | `src/prepare_forced_evidence_quote_probe_v1.py::best_mechanical_quote`（235–247）；句切分/BM25 在 `src/run_retrieved_evidence_nli_v1.py::sentence_spans, bm25_rank` | GPU manifest 只有三个 passage 正文，需先给每篇正文重建连续 `sentence_id` 行，再按每篇 top1、全局 `(-bm25, passage_id, sentence_id, text_sha256)` 取最小。 |
| 写盘与断点校验 | `src/run_exact_subset_attribution_v1.py::atomic_npz, cache_signature, validate_cache, extract`（139–146、902–1044） | 先写 `.pending` 再 replace；NPZ hash、输入/运行签名与 sidecar 绑定；存在单边文件或 hash 不同即停，不静默覆盖。 |
| cache API 参考 | `prelab/round23_local_evidence_probe/src/extract23.py::prefix_cache, branches, uncached`（74–106） | 仅是 Transformers `DynamicCache`/cache-vs-full 参考，模型是 Qwen、任务不是生成；不能直接当本协议 greedy runner。 |

## Q/K full replay 的实施合同

1. 每层 `self_attn` 先注册 `forward_pre_hook(with_kwargs=True)`，保存 `position_embeddings=(cos,sin)`，并断言 `past_key_value is None`；`q_proj` hook 保存未旋转 Q；`k_proj` hook 收到 K 后立即处理本层。hook 必须在 `finally/__exit__` 中移除。
2. 主前向只能是 `model.model(input_ids, attention_mask=全1, use_cache=False, output_attentions=False, output_hidden_states=False)`。P3 cached decode 阶段不挂 hook；停止后删除 KV cache，再以 `prompt_ids + generated_ids` 做这次权威 replay。
3. Q reshape 为 `[heads, S, head_dim]`；K reshape 为 `[kv_heads, S, head_dim]`。按现有审计口径分别执行 `x*cos[:,None] + rotate_half(x)*sin[:,None]`，随后 K 沿头轴 `repeat_interleave(heads//kv_heads)`。Llama2 当前是 32/32，但实现仍须保留 GQA 检查。
4. 只 index 引用内容 token 的绝对 query 位置，建议固定 `query_chunk=8`。每块计算 `(Q @ K.T) * module.scaling`，显式把 `key_position > query_position` 置 `-inf`，再用 float32 softmax。不能要求库返回整张 attention。
5. P3 的 claim/passage key 位置来自 frozen prompt 单次 tokenization 的 offsets；生成 quote 位置来自追加的生成 ID。P2 按协议把 `rendered_prompt + mechanical_quote + "</quote>"` 整串单次 tokenize，并从该整串 offsets 另算区域，不能拼接独立分词片段。claim、三个 passage 与 quote 区域必须互斥且非空；close-tag token排除。
6. 对每个内容 token `t`、层 `l`、头 `h`，先算所有 passage key 的 attention **均值**与 quote-content key 中位置不晚于 `t` 的 attention **均值**，再求 `R=passage_mean/(passage_mean+quote_mean)`。三篇 passage 和 claim 的额外信号是各自区域 attention **求和 mass**，不能与 R 的均值混用。
7. 256 列顺序必须直接从 token 轴计算：每层 `mean(t,h)R` 32；每层 `min_t(mean_h R)` 32；每层 `max_t(mean_h R)` 32；每层 `std_h(mean_t R, ddof=0)` 32；按 passage1、2、3 再按层的 `mean(t,h) passage_mass` 96；每层 `mean(t,h) claim_mass` 32。

`PLAN.json` 的 raw attention cache 只保存每头的 token mean/min/max `[32,32]`、passage `[3,32]`、claim `[32]`。这些值无法事后精确恢复 `min_t(mean_h R)` 和 `max_t(mean_h R)`。因此提取时必须同时在线计算并保存最终 `attention256 float32[256]`；不能把三个 per-head 汇总当成 256 维分类器视图。

## cached greedy 与 P2/P3 输出

仓库内没有合规实现。新循环应在首次 prompt prefill 后逐步使用 `past_key_values`；每步只保留末位置 float32 logits，`torch.argmax` 取最小并列 ID，保存 selected logprob、全词表熵和 `selected_logit-max(other_logits)`，追加 ID 后按冻结参数 decode **完整生成前缀**，检查首次 `</quote>`。EOS 单独保存并从逻辑 quote/replay 内容中排除；max-token 不续写。

cached-vs-replay oracle 不能只比较最后一步。no-cache replay 后，按每个生成目标的前一位置 hidden 分块过 `lm_head`，逐位置重算 argmax/logprob/entropy/margin，并执行协议中的 exact-ID 与 `5e-3` 容差。P2 teacher forcing 使用同一 pre-read 位置规则；即使目标 token 不是 argmax，margin 也必须保留负值。

建议每个 claim 的缓存至少明确保存：全部非终止 `generated_token_ids int32[G]`、`content_local_indices int32[T]`、内容 token 的 `selected_logprob/vocab_entropy/margin float32[T]`、`hidden_pca64 float16[T,64]`、`hidden256 float32[256]`、`attention256 float32[256]`、relation5、surface21、解析/停止/全部 exact 坐标，以及上述 raw attention audit cache。分类器 hidden256 应从已落盘的 float16 token PCA 转回 float32再聚合，避免“训练特征与审计缓存不同值”。完整 generated IDs 不能只留内容 token，否则无法复放或核对停止。

relation 分支整串单次 tokenize，最后上下文位置经 `lm_head` 后只取锁定 A/B/C IDs；三类 softmax、自然对数熵、原始 top1-top2 margin组成 5 列。不要把三类概率当全词表概率。

## 8GB 显存检查

现有同 checkpoint、同 NF4/SDPA、同 Q/K hook 的 `data/replay_selfcheck/manifest.json` 记录 RTX 3070（8,589,410,304 bytes）：模型加载后 3.603 GiB；622-token full no-cache replay 峰值 3.741 GiB。新协议最大总长 959；32层、双 K/V、32头、128维、BF16 的 959-token KV 理论量约 0.468 GiB。

只要生成后先删除 cache，再做逐层、query-chunk=8 的 Q/K 聚合，且 `output_attentions=False`，8GB 预计足够。这个判断仍是外推，必须由冻结 8-claim smoke 记录 `peak_allocated` 与 `peak_reserved` 后才能转成实测结论。若开启 `output_attentions=True`，959 token 的 32层完整 attention 单 BF16 张量总量约 1.75 GiB，连同中间 float32 softmax/模型很容易使 8GB 失败，因此明确禁止。

## 明显风险

- `run_feature_qa.load_nf4` 若脱离其 `prepare_signature` 单独调用，不会先重哈希全部权重；优先复用 `run_exact_subset_attribution_v1.load_nf4`。
- `best_mechanical_quote` 需要 `passage_rows`，sanitized manifest 没有该字段；需用锁定句切分函数从三个 passage 正文重建，不能打开原 fit 文件补字段。
- P3 的 BM25 表面列需给**所有**句候选计算 BM25；只保留机械 top1 会让模型生成一个非 top1 的逐字句子时，第17列错误地变成0。
- prompt 末尾与 quote 开头、quote 与 close 之间可能有跨边界 token。P3 replay 必须复用真实 `prompt_ids + generated_ids`；P2 必须服从整串 tokenize。不要用逐 token decode 拼字符坐标，也不要把 P2 的独立 quote token 数当权威 content 位置。
- P0 raw 矩阵含 fit 后还含 calibration 行；仅允许后续 CPU evaluator 通过锁定 index 选中 fit 前缀中的 256 答。GPU runner完全不需要 P0，也不应调用会读取 development/calibration/gold 的旧 fit 函数。
- `hidden_pca.pkl` 来自原回答状态，协议允许复用但存在轨迹分布偏移；不得据此重拟合或试新 PCA。

本审计只读取研究文档、Python 源码和无标签 GPU selfcheck 元数据；未打开任何 calibration、official test、gold/label JSONL，未加载模型，未初始化 CUDA，也未运行评分。
