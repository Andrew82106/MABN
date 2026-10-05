# 同信息 LLM judge baseline

本基线只看到 `independent_mas_v3/traces_public.jsonl` 的公开轨迹，不读取标签、机制字段或 OOF 预测；每批 8 条，temperature=0，LANYUN `glm-5.3-flash`，固定 seed=20261005。该基线是 zero-shot 外部判别器，不是本文方法的组成部分。

## 完整运行

- 抽样：4,000 条公开轨迹，500 批。
- 返回：3,119 条有效预测（390 批完整返回）；110 批失败，881 条轨迹缺失。失败原因保存在原始 `errors.jsonl`，没有被当作安全或危险。
- 与本文 OOF 在相同 3,119 条 episode 对齐后的结果：

| 方法 | AUROC | AUPRC | F1@0.5 | Brier |
|---|---:|---:|---:|---:|
| LLM judge | 0.778 | 0.585 | 0.574 | 0.315 |
| 双层 BN（同 episode） | **0.825** | **0.681** | **0.645** | **0.158** |

这说明在相同公开信息下，当前双层 BN 的排序、分类和概率校准均优于这个 zero-shot judge；但由于 judge 有 22.0% 的轨迹缺失，结果只能作为严格的部分覆盖 baseline，不能写成“全量 API judge 已完成”或据此宣称普遍优越性。

原始运行目录（按提交包约定保持为 development 产物）：
`results/submission/development/llm_judge_baseline_lanyun_full_20261006_r1/`
