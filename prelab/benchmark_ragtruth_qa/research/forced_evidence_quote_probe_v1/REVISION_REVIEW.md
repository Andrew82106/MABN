# Forced-evidence quote probe v1：修订复审

## 结论

**B1–B5 全部 PASS，没有发现新的同范围阻断项。** B6（实际 sanitized manifest 尚未物化）单列为 pending；因此可以继续 CPU manifest/runner 实现，GPU 仍需等 B6 独立审计。

审查锁：

- PROTOCOL SHA256：`78e5558b61592238b0979c7c60aea23c79b8c7d3181e88113d3972b4d056f7aa`
- PLAN SHA256：`6d439ee257dde65d2ef2c9974cd3dd716f4f0d06890cfb7df9c8af6da562680c`

## B1：Nested CV、阈值和指标 — PASS

每个 inner 模型只用自己的三份 inner-train 重算行、每答基础权重、weighted scaler、类质量/类因子和 LR；outer refit 只用四份 outer-train。候选阈值、`score >= threshold`、F1/precision tie、unweighted window/answer、sklearn 1.6.1 AP、pooled 输入和单类失败均已冻结。

## B2：推进门 — PASS

`exact_rate` 明确为 `sum(parse_valid AND stop_close AND source_exact_substring)/3776`，invalid 计失败且分母不能缩小。Answer 门统一为 `P3_answer_AP >= P2_answer_AP - 0.01`。

## B3：KV-cache 与 hooks — PASS

Cached decode 只决定 token IDs/logprob/entropy/margin，不挂 Q/K hook；P3 hidden/attention 固定来自停止后的 batch=1、no-cache full replay。Smoke 要求逐位置 argmax token 完全相同，scalar `rtol=atol=5e-3`，重复 hidden64/attention `rtol=0, atol=1e-6`，失败即停止。

## B4：tokenization、停止和 invalid context — PASS

所有路径整段 tokenize，固定 `add_special_tokens=False` 并断言唯一 BOS。每步重解完整 generated prefix，禁止逐 token decode 拼接。无闭合时只在隔离 relation forward 中以 `logical_quote + </quote>` 合成上下文，不改变生成 validity/exactness。

## B5：P0 lineage — PASS

Raw `lb_prefix_pre_header.npy` 的 hash、float32 `[696220,1024]`、token/window/answer index 与生成代码均已锁定并核对；全 fit standardized cache、旧模型和旧分数被明确禁止。P0 固定 `C=1e-4`，使用相同的训练子集局部 scaler/权重/类别平衡规则。

## Pending：B6

实际 sanitized manifest、lineage 和 runner 尚未物化。本复审不把它算作 B1–B5 的失败，但在独立核对 allowlist、hash 和 fail-closed 门以前，不能启动 GPU。

本复审未读取 calibration/official test，未运行 GPU、模型或训练，也未修改 PROTOCOL/PLAN。
