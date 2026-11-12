# 同信息 LLM judge baseline

本基线只看到 `independent_mas_v3/traces_public.jsonl` 的公开轨迹，不读取标签、机制字段或 OOF 预测；每批 8 条，temperature=0，LANYUN `glm-5.3-flash`，固定 seed=20261005。该基线是 zero-shot 外部判别器，不是本文方法的组成部分。

## 完整运行

- 抽样：4,000 条公开轨迹，500 批。
- 首次运行返回 3,119 条；定向重试 110 个失败批次，并对最后 3 个异常批次再次低并发重试后，最终装配为 4,000/4,000 条有效预测、0 个缺失或非法 episode。所有原始失败和重试目录均保留在 development 目录，未覆盖原始结果。
- 与本文 OOF 在全部 4,000 条 episode 对齐后的结果：

| 方法 | AUROC | AUPRC | F1@0.5 | Brier |
|---|---:|---:|---:|---:|
| LLM judge | 0.771 | 0.578 | 0.571 | 0.317 |
| 双层 BN（同 episode） | **0.820** | **0.675** | **0.633** | **0.160** |

这说明在相同公开信息下，当前双层 BN 的排序、分类和概率校准均优于这个 zero-shot judge。它是全量同信息辅助基线，但不是已发表方法的复现，也不证明对所有外部模型或数据集普遍优越。

原始运行目录（按提交包约定保持为 development 产物）：
`results/submission/development/llm_judge_baseline_lanyun_full_20261006_assembled_final_r1/`
