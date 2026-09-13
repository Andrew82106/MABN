# 正式 baseline 协议审计 v3

日期：2026-09-13。结论：**FAIL；两者继续 N/A，不得以数值进入正式主表。** 本次仅静态读取协议、源码和现有审计；未运行 GPU，未打开任何新 test/validation，也未修改 baseline 代码。

## 总结

| 对象 | 核心方法与统一计分 | 正式身份门禁 | 主表影响 |
|---|---|---|---|
| ReDeEP(Token) 论文公式迁移 | PASS：共同 fit/cal、4-BPE、`answer=max(window)`、双层同规则阈值、F1/AUROC/AP；模型外字符映射无参数且不读 calibration gold | FAIL：抽取未完成；runner/原始特征未绑定冻结 checkpoint、源码和候选头身份；公共冻结清单仍是旧身份 | 只能列 N/A，不能计入“最强正式 baseline”数值比较 |
| RAGognizer 官方 Llama-2 checkpoint 迁移 | PASS：LoRA、集成检测头、BF16、sigmoid 原生概率不改；字符映射无参数无标签；统一计分正确 | FAIL：当前入口只覆盖 calibration 159，而推理型 baseline 协议要求 fit+cal 全 3,839；公共冻结清单没有登记 | 只能列 N/A，不能把未来 cal-only 分数当正式主表结果 |

## 会影响正式身份的偏差

1. **RAGognizer 覆盖范围不足。** `BASELINE_PROTOCOL.md:5` 要求无需项目标签训练的推理型 baseline 对全 3,839 答产生冻结原生输出。现 `METHOD_FREEZE.json:101-105`、`prepare_cal_plan.py:202-226`、`run_cal_inference.py:121,135-136`、`adapter.py:142,184-185` 都固定为 calibration 159。该链路结构上无法满足全量覆盖。

2. **公共冻结清单失真。** `FORMAL_BASELINE_FREEZE.md:21` 仍称 ReDeEP 没有完整可执行参数身份，并引用旧协议哈希 `a398...`；当前 `METHOD_FREEZE.json` SHA256 已是 `0a03b59c...`，且 runner/score 已存在。该清单完全没有 RAGognizer 条目。与此同时，`BASELINE_PROTOCOL.md:17` 声称正式入口及哈希由该清单给出，因此当前公共身份链自相矛盾。

3. **ReDeEP 结果 provenance 未封闭。** `run_redeep_formal_baseline_v1.py:325-351` 运行前只核输入和候选头文件，不核 `METHOD_FREEZE`、模型 shard、tokenizer、上游 commit 或 runner/core 源码哈希；`save_features`（296-322）也不把这些身份写入每答产物。更早的 `--resume` 验证（279-291）只看四组 shape 与 answer hash，连 version、候选头、坐标和有限值都不查。`score_redeep_formal_baseline_v1.py:88-113` 虽然后验补查部分字段，但其 `candidate_heads` 在 223-231 仅要求各行一致，没有与发布的 32 头清单逐值核对。因此现有 scorer 不能证明输入分数确实来自冻结 checkpoint/代码。当前本机 shard、tokenizer、候选头及上游 commit 手工静态复核均匹配冻结值，独立抽样/全量现存 raw 审计也未发现实际错文件；问题是正式链不强制这些事实。

4. **RAGognizer 最终分数链尚未注册。** runner 对权重和上游源码的硬门禁是 PASS，关键 8 个本地文件也与 `ARTIFACT_HASHES.json` 全匹配；但 `evaluate_shared.py:105-149` 只冻结 adapted-score 哈希，并未校验 GPU run audit/adapter audit。正式登记时至少要把 runner、adapter、evaluator、raw output、GPU audit、adapted output 和 score-freeze 哈希串起来，避免脱离已审计 runner 的同格式分数进入评测。

5. **已有 ReDeEP raw 审计未进入正式 RUNBOOK。** `src/audit_redeep_raw_features_v1.py` 能核候选头并生成逐文件哈希清单，但 `RUNBOOK.md:11-16` 只调用 `score ... check/freeze/evaluate`，而 `freeze_scores` 不要求该 manifest/complete 标记。把这道审计设为 score freeze 的前置依赖后，已有正确缓存可以验收复用；否则一致但来源错误的一整套缓存仍可能通过。

6. **新独立 holdout 还没有可执行入口。** ReDeEP runner 只接受当前冻结的 fit/cal 输入，scorer 也没有 holdout stage。这不影响当前 calibration 开发表继续列 N/A，但会阻断以后的一次性最终结论；届时应另建只消费已冻结方法参数/阈值的封存入口，不能重新选择头、层、MinMax 或阈值。

## 已通过的实质规则

- ReDeEP：`METHOD_FREEZE.json:26-35,37-71` 与 score 入口一致；fit 标签只用于作者方法内的头/层选择，calibration 标签直到冻结分数后才读取；论文公式身份与官方代码诊断分名，禁止按结果择优。
- RAGognizer：`METHOD_FREEZE.json:27-58`、`run_cal_inference.py:176-267` 保留官方集成 `transformer_heads`、LoRA、检测头、BF16、sigmoid 和原生 response-token packing；无量化、替换头、重训、平滑、融合或概率校准。
- 两者的统一层都使用项目 4-BPE 合格窗口、窗口/整答分别按 `F1 → precision → higher threshold` 选阈值、判定 `score >= threshold`，并报告 F1/AUROC/AP。整答均为 `max(eligible windows)`。
- 两者的输出坐标映射均为预先固定的字符区间相交与算术聚合，不读取 calibration 标签，不新增学习参数。ReDeEP 回答 17592 的首字符无原生 target score已披露，未伪造分数；这是可用性边界，不应解释为完整原生覆盖。

## 判定

核心公式/模型与统一评测设计可继续执行，没有发现为了压低 baseline 而删模型、换 checkpoint、改原生分数或择优路线的行为。但在上述身份门禁修复、全量覆盖完成并独立复算前，ReDeEP 与 RAGognizer 都不能成为带数值的正式主表 baseline；因此也还不能用它们判断当前候选是否满足“不被最强正式 baseline 超过”。
