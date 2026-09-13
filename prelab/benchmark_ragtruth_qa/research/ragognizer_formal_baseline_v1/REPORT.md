# RAGognizer 原版模型正式基线审计

结论：RAGognizer 原版模型正式基线已完成。全 3,839 答原生推理、696,220 窗无标签适配、score freeze、calibration 159 正式评分和独立复算均通过。主路径保留官方模型卡默认的 `transformer_heads` 模型、LoRA、检测头和原始 sigmoid 概率；项目只更换数据与计分接口。该方法是 **2026 年 arXiv 预印本**，不是已发表顶会论文。

## 已冻结身份

- 官方源码：`F4biian/RAGognizer` commit `1d3e8fdd6de638dc2d06265829a4d9ced387e0ba`。
- 官方模型：`F4biian/RAGognizer-Llama-2-7b-chat-hf` revision `2b58ab9aa73f6d499ec72a38ced0976caf78b26f`；下载文件共 345,138,041 bytes，逐文件哈希见 [MODEL_MANIFEST.json](./MODEL_MANIFEST.json)。
- `transformer_heads`：解析 commit `6a2ca2a12a25035ea290b0b0e03839fc16348e44`；作者没有锁定此依赖 commit，因此这是本次运行身份的一项公开限制。
- 原模型：Llama-2-7b-chat、released LoRA、`hallu_head_neg_16`（layer -16，4096→1024→1024→1，ReLU，无输出 bias），BF16 请求，`use_postprocessor=False`，原生分数为 head logit 的 sigmoid。
- 独立 MLP 是 4096→256→64→1，结构和参数身份不同，只作诊断，不替代主路径。

论文和模型卡支持“checkpoint 在 RAGognize 上训练、在 RAGTruth 跨数据集评测”。发布物没有精确训练行 manifest 和完整启动参数，因此不能对训练来源做逐行密码学证明。任务未下载或打开 RAGognize test。

## 全量覆盖门禁

[INFERENCE_PLAN.jsonl](./INFERENCE_PLAN.jsonl) 覆盖 fit 3,680 答/615 组/653,979 窗和 calibration 159 答/154 组/42,241 窗；组交集为 0，总计 3,839 答、696,220 窗。SHA256 为 `90d21b3bc9aa8e92dc7c6ef8fac3d7e178effa0236a5464831541ea401343a76`，两次独立构建逐字节相同。最长作者输入 1,235 token，低于 4,096。

推理型基线先产生全部 3,839 答的原生输出。fit 输出仅满足覆盖与未来冻结，未读取 fit 标签。全量 raw、无标签适配和 `SCORE_FREEZE` 完成后，评分器才打开 calibration 159 的 gold；顺序门禁已实际通过。

## Unicode 与短窗审计

3,836 答逐值复现作者 `_pack_probs` 的概率顺序和跨度。3 条 fit 回答（`15069`、`12426`、`15066`）含 UTF-8 byte fallback，作者逐 token decode 产生替换字符并使公开 packer 失败；外部坐标层改用同一 fast tokenizer 对完整字符串一次编码所得 offsets。所有回答字符均被覆盖，所有选中的原生概率均保留，未补造或丢弃概率，模型数值路径不变。

项目 tokenizer 与作者 tokenizer 身份不同，所以全量统一做 token→字符→项目 BPE 的确定性相交映射。每个本地槽取相交作者 token 概率均值。普通回答的窗口取 4 个实际槽均值；唯一短回答 `14641`（`exclude`，1 BPE）按冻结规则保留一个 1 槽短窗并按实际槽数求均值。整答取 `max(window)`。

独立计划审计 [PLAN_INDEPENDENT_AUDIT.json](./PLAN_INDEPENDENT_AUDIT.json) 与适配单元自检 [CPU_SELFTEST.json](./CPU_SELFTEST.json) 均 PASS。

## 8 GiB 可执行性

两个 base shard 共 13,476,872,576 bytes，无法直接装入 8 GiB RTX 3070。正式 A 路径在 CPU 加载同一 BF16 模型，再以 `accelerate.cpu_offload(..., offload_buffers=True)` 按需搬到 GPU；它只改变设备放置，不量化、不合并或修改权重。

全量模型输入共 2,419,983 token，平均约 630 token/答。calibration 159 实测推理 363.92 秒，即 2.289 秒/答。正式全量推理实测 8,423.74 秒，总运行 8,458.33 秒（2.3495 小时，2.194 秒/答）；峰值 allocated/reserved 为 1.368/1.449 GB，746 个参数 tensor 全为 BF16。CPU RAM 约需 14–20+ GiB。

作者加载器按发布元数据 `new_embeddings_added=1` 调用 `resize_token_embeddings(len(tokenizer)+1)`，并报告 checkpoint 缺少 `lm_head` 的两个 LoRA key。正式 runner 忠实保留此作者默认行为；这些 `lm_head` 权重位于检测头读出之后，不参与 `hallu_head_neg_16` 的 hidden-state 输入。跨三次独立加载，重复样本的检测概率逐值完全相同。

## 评分与证据链

作者阈值 `0.6523` 只在附录；主表对未改概率使用统一 calibration 阈值规则并报告 F1/AUROC/AP。逐行 plan hash、raw hash chain、GPU audit、adapter hash chain、score freeze 和 evaluator hash 构成闭合证据链，协议见 [PROTOCOL.md](./PROTOCOL.md)。

两次 1 答 BF16 offload smoke 均通过。由于全量估计超过 90 分钟，calibration 159 曾先按预先冻结的独立链给出 provisional；正式全量中的 159 条原生概率现已与其 159/159 逐值完全匹配。

正式主表结果：窗口阈值 0.633157，F1/AUROC/AP 为 **0.507511/0.824938/0.367650**；整答阈值 0.708809，F1/AUROC/AP 为 **0.806867/0.706949/0.745699**。作者原生阈值 0.6523 只在附录：本项目 calibration 整答 F1/AUROC/AP 为 0.788845/0.724407/0.781285。

[SCORE_INDEPENDENT_AUDIT.json](./SCORE_INDEPENDENT_AUDIT.json) 不导入 adapter 或 evaluator，独立流式重算全 3,839 答、697,623 个原生概率、708,506 个本地 BPE 槽、696,220 个合格窗、短窗、纯标点窗排除、逐行 hash chain 和三个指标块，结果 PASS。与正式结果的最大浮点差为 `6.11e-16`。正式 raw SHA256 为 `b0419fc5e156df29eb70f1d919a0223dd24cf0359a8a8aca80accfa9321190ca`，adapted SHA256 为 `ed2fb04db6b27166ebf0c5b4bedcc0facf90670c80564c74e630689e9fc37be1`，score freeze SHA256 为 `e8b8f97ac9a6db9e1e08e206862690d5847608b90c6a7dd4cb7e7d9058d587be`。
