训练扩充已导出并冻结，当前只完成 CPU 准备，没有训练或新 GPU 推理。

| 项目 | 新增去重后 | 合并原训练集 |
|---|---:|---:|
| 答案 | 3,046 | 3,680 |
| 含人工风险的答案 | 799 | 1,127 |
| 人工风险片段 | 1,247 | 1,893 |
| 原始 Llama BPE | 495,347 | 665,708 |
| 可评四 BPE 窗口 | 485,856 | 653,979 |
| 阳性窗口 | 36,956 | 58,433 |

来源仍为 634 个、615 个原 fit 组。原 634 份答案、四份 gold 文件均作为原字节前缀保留，并全量重现原分词坐标。calibration 159 份只校验文件哈希，内容未解析；test 未打开或改动。生成模型名称只放来源元数据，不进入分词或核查输入。

同来源、同全文的 32 份重复答案已合并。26 组重复的完整标签（包括 meta 和 flags）均一致，没有冲突；`data/answer_provenance.jsonl` 保留所有原 ID、模型和温度，`data/label_conflicts.jsonl` 为空。以后若出现完整标签冲突，规则是隔离而非自动裁决。

`data/tokens_fit.jsonl` 和 `data/windows_k4_fit.jsonl` 沿用原 schema。新增坐标来自完整 `<s>[INST] {released_prompt} [/INST] {original_response}`，未调用 Llama 前向，不代表其他生成模型的历史词元轨迹。全部人工风险类型保留；标点计四 BPE 长度，纯非字母数字窗口另列，正常文字和质量 good 的拒答均保留。

MiniCheck 新计划有 19,699 个 claim、25,617 对文档/claim 输入；11 个长句按原冻结规则拆分，实际最长输入 497 tokens，零截断。原 634 份最后层缓存只读复用，逐文件哈希已核验；新缓存只计算新增 3,046 份。

额外第 22 层缓存预计 460.6 万至 642.6 万个有效完整输入 tokens，float32 状态约 **18.9–26.3 GB**；claim 最后层约 **1.71 GB**，另有少量坐标。准备时磁盘余量约 309 GB。区间来自各 claim 可选文档的最短/最长输入，实际选块尚未计算。旧 634 份第 22 层状态尚未补跑。

每答 `minicheck/encoder22_features/{response_id}.npz` 预定字段：

- `hidden22[sum_valid,1024]`：`encoder.layer[21] output[0]`，最后两层的入口，float32。
- `input_ids`、`attention_mask`：包括完整文档、特殊符号和 claim 的有效 tokens。
- `sequence_offsets[nclaim+1]`、`claim_index`、`document_index`、`batch_padding_length`。
- `answer_token_start/end[sum_valid]`：全原答字符坐标；非 claim 部分为 -1。

按每答不压缩 NPZ 保存。JSON 绑定计划、分数、原答和数组哈希。选块仍是原模型支持分最高的文档，平分取首个，不用标签。新增只读 hook 必须通过 logits、支持分和原最后层状态完全一致检查。缓存本身不微调模型。

只读验证命令：

```powershell
prelab/.venv/Scripts/python.exe prelab/benchmark_ragtruth_qa/fit_expansion/run_minicheck_expansion.py verify
```

**等待主代理明确释放 GPU 后**才执行：

```powershell
prelab/.venv/Scripts/python.exe prelab/benchmark_ragtruth_qa/fit_expansion/run_minicheck_expansion.py infer
```

该入口只有验证和推理，不包含训练、调参或评测。训练方案由后续单独冻结。`data/export_freeze.json` SHA256：`e68d662e53b3554c2cc28ba436c6d6525462619a023c40b17caebccea0980b86`。
