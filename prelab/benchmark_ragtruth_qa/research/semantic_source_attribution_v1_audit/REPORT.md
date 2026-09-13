# Semantic source-attribution v1 独立审计

审计时间：2026-09-13（Asia/Shanghai）  
审计对象：`src/run_semantic_source_attribution_v1.py`  
冻结 runner SHA-256：`a87b4c8bcc232cead05cdaa23deb67642c8551f11870b5fd1a640513e78b977b`

## 结论

**结论为“数值实现通过，单进程提取可继续；并发恢复协议有一个高优先级缺陷”。** 没有发现 Q/K/V、RoPE、因果遮罩、坐标或 CSR 顺序方面的致命错误。当前全量提取只有一个进程，因此所述并发缺陷不影响本轮数值结果。

## 已核实的实现

- Q/K/V 形状为 `[batch, head, token, head_dim]`；RoPE 调用与 Transformers 4.51.3 的 `LlamaAttention.forward` 相同。
- `repeat_kv` 按 `num_attention_heads / num_key_value_heads` 扩展 K、V。冻结的 Llama-2-7B 实际为 32 个注意力头、32 个 KV 头，扩展倍数为 1；CPU 检查另外覆盖了 4/2 的 grouped-KV 情形。
- 注意力先施加 `key_position <= query_position` 的因果遮罩，再做 float32 softmax。
- 查询位置是已经读入该回答词元后的 post-token 状态。previous-answer 严格使用 `< query`，排除了查询词元自身。
- 基础归因量为 `attention(q,s) * ||V_s,head||_2`。每个片段先在其词元内取 head-wise 最大值，再在一个 claim 的词元间求平均；紧凑特征最后在 32 个头间求平均。
- 每个 claim 的句子归因按 claim-major、sentence-major 顺序写入 CSR；每条边保留 32 层 × 32 头的 float16 值。

独立重跑 CPU oracle 的结果：

| 检查 | 最大绝对误差 |
|---|---:|
| source / previous / other / passage / sentence 的 float32 紧凑量 | `1.862645149230957e-09` |
| CSR float16 与 dense oracle | `7.595866918563843e-06` |

CPU 自检文件 SHA-256 为 `f0c1262e3929e1e648c5172ef58b735ceabfe57084e26401d33db117b772b081`。对 793 个真实布局全部构造 membership 后，最大 float32 行和误差为 `3.5762786865234375e-07`，低于固定门限 `5e-7`。

## 坐标与索引

- 793 个回答、11,322 个微事实、11,341 个来源句子全部能回到原始整串 tokenizer 坐标。
- 独立重算所有 claim 的词元集合，缺失或多余均为 0；所有字母数字字符都被覆盖。
- claim 边界处共有 4,329 个跨界 tokenizer 词元，界外部分全部只是空白，没有混入相邻事实的字母数字字符。
- 来源句子共有 6,931 个跨界词元，其中 6,920 个界外部分仅为空白，11 个仅含标点，0 个含界外字母数字字符；句子之间、段落之间的 token 集均无重叠。
- 所有句子均落在其标记段落 body 内。
- `global_csr_index.npz` 的 11,323 个 indptr、11,322 个 claim 行和 166,644 条边，按 response、claim、sentence、passage、sentence id、文本 SHA 全量独立重建后逐项一致。
- `microclaim_index` 是回答内局部编号；它与 `claim_response_indices` 联合后可唯一定位，不是全局编号。

## 标签与测试集隔离

runner 只读取冻结的 793 条开发集 replay plan、无标签 atomic input、模型清单与模型文件。输入键名全量扫描只发现声明性的 `labels_used=false` 和 `official_test_opened=false`，没有事实标签、风险标签或 official-test 内容。runner 不包含训练或评分入口。

## 一个已提交分片的检查

对最终签名下的回答 `16023` 做了只读检查：NPZ、metadata、commit 三重 SHA 一致；claim、sentence、passage、文本 SHA、top-k 排序均与冻结布局一致。CSR float16 重建 sentence compact 的最大漂移为 `5.015730857849121e-05`。该分片与修复 membership 前的同一回答 NPZ 字节完全一致，说明不受本次长 claim 行和修复影响。

## 无标签效度弱检查

在提取进行到 120 个已提交回答时做快照。将 1,024 个 layer-head 句子值取均值并排序；对回答自身带有 explicit citation 或继承 parent passage 的 claim，检查 top 句子是否来自所指 passage：

| 弱代理集合 | claim 数 | top-1 命中 | 随机期望 | top-3 命中 | 随机期望 |
|---|---:|---:|---:|---:|---:|
| explicit citation | 344 | 0.6424 | 0.3647 | 0.9564 | 0.7601 |
| 仅 parent passage | 119 | 0.6723 | 0.3538 | 0.9328 | 0.7608 |
| explicit 优先，否则 parent | 463 | 0.6501 | 0.3619 | 0.9503 | 0.7603 |

这说明归因信号至少能较明显地找回回答自己指向的资料段。**引用或 parent passage 是模型输出结构提供的弱代理，不是事实正确性的金标签，不能用这组数代替最终幻觉评测。**

## 风险与处理建议

### 高优先级：锁前 quarantine 存在竞态

`extract()` 在取得 `.exclusive_gpu_runner.lock` **之前**执行 `read_committed()` 与 `quarantine_uncommitted()`。若第二个 extract 恰在第一个进程已写 NPZ/metadata、尚未写 commit 时启动，第二个进程可以先移走第一个进程的文件，之后才因 GPU 锁存在而失败，导致第一个进程保存失败。

本轮只有一个 extract 进程，当前结果不受影响。下一版应先取得锁，然后在锁内重新扫描 committed/missing 并执行 quarantine。硬终止遗留的 stale lock 也应有基于 PID/所有权的明确恢复流程。

### 中优先级：缓存语义校验还可加强

`validate_arrays()` 目前没有把 `microclaim_indices`、sentence passage/id/hash、top-k indices/values 逐项与 layout 比较。正常写入路径直接从 layout 生成它们，且 commit SHA 能发现随机损坏，所以本轮抽查正确；正式冻结前仍建议补上这些精确断言。

### 中优先级：环境签名没有在 extract 时重算

签名保存了 Python/package/CUDA 版本，但 `load_prepared()` 没有断言当前 `_software_signature()` 与冻结签名一致。因此在 CPU 自检后若环境被更新，旧自检仍可被接受。当前运行环境未改变；下一版应在 extract 前复核环境签名。

### 方法解释边界

- 这是 attention × value-norm 启发式归因，不是完整 LRP，也不等价于删除该句后的因果效应。
- post-token 信号是在词元已经输出后定位风险，适合监测与复查，不应描述成输出前预测。
- 片段内取最大值会偏向较长片段；后续训练与消融应显式检查句长相关性。

## 资源判断

冻结数据最长序列为 1,232 tokens，最多 458 个查询词元、42 个 claims、48 个来源句、720 条 claim-sentence 边，均低于模型 4,096 token 上限。CSR 主体预计 341,286,912 bytes；含紧凑数组的未压缩粗估约 371 MB。运行快照显示 RTX 3070 使用约 5,196/8,192 MiB，尚余约 2,823 MiB，未见显存危险。提取仍应保持单进程。
