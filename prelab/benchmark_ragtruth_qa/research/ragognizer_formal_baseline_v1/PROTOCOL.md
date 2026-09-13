# RAGognizer 原版模型正式基线协议

状态：全量 CPU 计划、独立几何审计、适配器自检、两次 1 答 GPU smoke、全 3,839 答推理、无标签适配、score freeze、正式评分和独立复算均已通过。本方法是 **2026 年 arXiv 预印本**（`2604.15945v1`），不声称已发表于会议或期刊。

## 方法边界

主基线固定为模型卡默认的 `transformer_heads` 集成路径：官方 Llama-2-7b-chat checkpoint、原 LoRA、`hallu_head_neg_16`、BF16 请求、`sigmoid` 原生逐 token 概率、`use_postprocessor=False`。Accelerate CPU offload 只改变设备放置。不得重训、换头、融合、平滑、校准或量化。独立的 4096→256→64→1 MLP 结构不同，只作诊断且不择优。

作者阈值 `0.6523` 只在附录展示。主表保留原始概率，在 calibration 上按项目统一规则分别选择窗口和整答阈值：先最大 F1，再较高 precision，再较高阈值，判断为 `score >= threshold`；报告 F1、AUROC、AP。

## 数据与全量计划

正式 plan 是 [INFERENCE_PLAN.jsonl](./INFERENCE_PLAN.jsonl)，SHA256 `90d21b3bc9aa8e92dc7c6ef8fac3d7e178effa0236a5464831541ea401343a76`。确定性重复构建逐字节相同，独立审计见 [PLAN_INDEPENDENT_AUDIT.json](./PLAN_INDEPENDENT_AUDIT.json)。

| 分区 | 回答 | source group | 合格窗口 |
|---|---:|---:|---:|
| fit | 3,680 | 615 | 653,979 |
| calibration | 159 | 154 | 42,241 |
| 合计 | 3,839 | 769 | 696,220 |

fit/cal group 交集为 0。计划只取官方 RAGTruth `train` split；没有下载或打开 RAGognize test。全量推理必须先冻结 3,839 答原生输出。fit 输出仅满足覆盖和未来冻结，不读 fit 标签；正式主评分只打开 calibration 159 的 gold。

每条输入严格构造为用户 `released_prompt` 与助手 `original_response`，无 system message、无 documents wrapper；`apply_chat_template(..., add_generation_prompt=False)` 后按作者路径 decode，再以 `add_special_tokens=False` 重编码。全量最长 1,235 token，小于 4,096。

## 无标签坐标与 4-BPE 映射

作者 tokenizer 与项目 tokenizer 的 `tokenizer.json` 哈希不同，因此正式全量统一使用字符相交映射。每个项目 BPE 槽取所有相交作者 token 原始概率的等权算术均值；空相交直接失败。映射不读取标签、无参数、也不改变原概率。

3,836 答由作者 `_pack_probs` 直接给出字符跨度，同时 fast-tokenizer 逐位置复核 token ID 和概率索引。另有 3 个 fit 回答（`15069`、`12426`、`15066`）含 UTF-8 byte fallback，作者逐 byte decode 为替换字符而无法完成打包；这些行只用同一 fast tokenizer 的全文 offsets 建立外部字符坐标。审计确认回答字符完整覆盖、所有原生概率位置均保留，没有补造或丢弃概率。

窗口几何按项目冻结规则：若回答有 `N>=4` 个项目 BPE，窗口是 `start:start+4`，分数除以 4；若 `0<N<4`，只保留 start=0 的唯一短窗，使用全部 N 个实际槽并除以 N。全 3,839 答仅 `14641` 属于后一种情况：回答 `exclude`，1 个 BPE（id 19060），窗 `14641__k4_00000` 为 `[0,7]`。不得丢弃、填充、复制或仍除以 4。整答分数为其全部合格窗口的最大值。

## 执行与哈希链

1. [run_inference.py](./run_inference.py) 默认执行全部 3,839 答；`--limit` 仅限 smoke/debug，`--resume` 会先校验既有逐行链。批大小固定 1。
2. 每个 plan 行有 `plan_row_sha256`。runner 写 `run_identity_sha256`、`raw_row_sha256` 和累计 `chain_sha256`；`GPU_RUN_AUDIT.json` 绑定 runner、plan、模型 manifest、源码、runtime alias、raw 文件和最终链。
3. [adapter.py](./adapter.py) 验证上述身份后写适配行链；`ADAPTER_AUDIT.json` 绑定 GPU audit、raw、adapter 和 adapted 文件。
4. [evaluate_shared.py](./evaluate_shared.py) 的 `freeze` 阶段要求 3,839 答且不打开 gold，生成 `SCORE_FREEZE.json`；随后 `evaluate` 只过滤 calibration 159，再打开 calibration gold 并计分。

Smoke 命令：

```powershell
& 'tmp/venvs/ragognizer_cpu_audit/Scripts/python.exe' `
  'prelab/benchmark_ragtruth_qa/research/ragognizer_formal_baseline_v1/run_inference.py' `
  --gpu-authorization REDEEP_GPU_RELEASED --limit 1 `
  --output 'prelab/benchmark_ragtruth_qa/research/ragognizer_formal_baseline_v1/SMOKE_NATIVE.jsonl' `
  --audit 'prelab/benchmark_ragtruth_qa/research/ragognizer_formal_baseline_v1/SMOKE_GPU_AUDIT.json' --overwrite
```

Smoke 通过后去掉 `--limit` 并输出 `NATIVE_PROBABILITIES.jsonl` / `GPU_RUN_AUDIT.json`；中断后用 `--resume`。本次全量运行未中断并完成 3,839/3,839 答。

实测 calibration 159 答为 2.289 秒/答，线性外推全量约 2.44 小时，超过 90 分钟。为尽快判断方法强弱，另在任何 gold 访问前冻结 [PROVISIONAL_PROTOCOL.json](./PROVISIONAL_PROTOCOL.json)，仅选择全量 plan 中的 calibration 子序列。该链同样先写 raw/GPU audit、再无标签适配、再写 `PROVISIONAL_CAL_SCORE_FREEZE.json`，最后才打开 calibration gold。正式全量实测推理 8,423.74 秒，总运行 8,458.33 秒（2.3495 小时），峰值 allocated/reserved 为 1.368/1.449 GB；746 个参数 tensor 全为 BF16。正式全量中的 159 条 calibration 原生概率与 provisional 逐值完全相同。

## 正式结果

全部 3,839 答原生输出和 696,220 个合格窗先经无标签适配并冻结，随后才打开 calibration gold。主表 calibration 159 的正式结果为：

| 粒度 | 阈值 | F1 | AUROC | AP |
|---|---:|---:|---:|---:|
| 4-BPE 窗口 | 0.633157 | 0.507511 | 0.824938 | 0.367650 |
| 整答 `max(window)` | 0.708809 | 0.806867 | 0.706949 | 0.745699 |

作者原生阈值 0.6523 只作附录：在本项目 calibration 整答标签上，原生 `max(token)` 的 F1/AUROC/AP 为 0.788845/0.724407/0.781285；这不是论文表中数字。独立复算器不导入 adapter 或 evaluator，重建全部映射、字母数字合格窗过滤、短窗、均值、max、hash chain 和指标，结果 PASS，最大浮点差 `6.11e-16`。

## 来源边界

论文与模型卡把该 checkpoint 描述为在 RAGognize 上训练，并把 RAGTruth 用作跨数据集评测。发布物没有精确训练行 manifest 和完整启动参数，因此能支持“RAGognize 训练、RAGTruth 评测”，但不能从 checkpoint 做逐行密码学证明。
