# Exact-subset × expanded-v4 complement：fit-only exploratory diagnosis

这是固定 256 答 fit cohort 上的互补性诊断，不是新候选、正式选择结果或独立验证。只计算预先指定的 v4、语义方向固定的 exact，以及 0.75/0.25 CDF-rank 融合；没有搜索权重、特征或阈值。

## 总体 4-BPE 窗口

| 固定分数 | F1 | AP | Precision | Recall | Predicted positive |
|---|---:|---:|---:|---:|---:|
| expanded-v4 OOF | 0.560446 | 0.479554 | 0.505616 | 0.628613 | 5,075 |
| exact: -mean(full-empty) | 0.159201 | 0.136713 | 0.144250 | 0.177609 | 5,026 |
| 0.75 v4-rank + 0.25 exact-rank | 0.402301 | 0.459707 | 0.550404 | 0.317001 | 2,351 |

F1 operating point固定如下：v4 使用既有 fit-OOF 阈值 `0.8225097060203552`。对 exact 与融合，在每个 frozen exact-v3 pilot held fold 内，仅用其余四折分数算右连续经验 CDF；把既有 v4 阈值在训练分布中的分位 `q_f` 原样迁移。标签不进入尺度化或阈值迁移。

## 类型召回

| 固定分数 | 冲突 recall | 无依据 recall |
|---|---:|---:|
| expanded-v4 OOF | 0.165079 | 0.667020 |
| exact: -mean(full-empty) | 0.136508 | 0.180950 |
| 0.75 v4-rank + 0.25 exact-rank | 0.041270 | 0.339878 |

冲突与无依据按原始四类 span 在 raw BPE 上重建；mixed windows 同时计入两个分母。

## 互补性

- v4 阈值以下共有 **40,583** 个窗口，其中 **1,516** 个为 v4 false negatives；exact 在这个完整 v4-rejected 区域的 AP 为 **0.051420**，区域正例率为 0.037356。
- 全体 raw score：Pearson **0.236322**，Spearman **0.271489**。
- 交叉拟合 CDF ranks：Pearson **0.272649**，Spearman **0.272405**。

固定融合没有显示可用互补：相对 v4，AP 下降 **0.019847**，固定 operating point 的 F1 下降 **0.158145**；冲突召回也从 0.165079 降至 0.041270。exact 在 v4-rejected 区域的 AP 仅比该区域 0.037356 的正例率高 0.014064。低相关在这里没有转化为有效补漏，因此该融合不升格为新候选。

## 映射核对与边界

从五个 `fold_*/predictions.npz` 独立组装 34,919 个 fit claim OOF，再按 expanded-v4 冻结 claim→window max 映射重建同一批 45,658 个窗口。response id、answer hash、token ids、offsets、window `(response_id, token_start)` 和二元标签全部逐项相等。独立审计结果见 `AUDIT.json`。

本诊断没有打开 combined v4 `scores.npz`，也没有读取 calibration 行/分数或 official test；没有加载模型或使用 GPU。结果只能回答这批 fit 样本上两个固定信号是否呈互补迹象，不能据此把融合登记为新候选。
