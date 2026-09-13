# Forced-evidence quote probe v1：独立协议审查

## 结论：BLOCK

Native cohort、无标签 claim、549 维特征和资源统计已经修正并能独立复现；但当前仍不能启动 GPU 或评分。主要阻断是 nested CV 仍可能泄漏、推进门含义不一致、KV-cache 与现有 no-cache hooks 不兼容，以及 P0/实际 sanitized manifest 尚未锁定。

## 已通过

- 3,680 个 fit 回答、615 组；634 个原生 Llama2 回答覆盖 **615/615** 组。冻结样本为 256 组、256 个原生 Llama2 答、3,776 claims。
- response/group digest 分别为 `bf1fd595…` / `e24c2729…`；五折 40/56/62/45/53，均复现。
- label-blind 文件先命中 3,777 行，固定规则排除纯 `":` 后为 3,776；`claim_prompt_text` digest `a59b8b52…` 复现。
- prompt 不含“资料不足/拒答”暗示；template/suffix hash、30-token suffix、A/B/C token IDs 319/350/315 均正确。
- P1=21 维；P2/P3=`21+11+5+256+256=549` 维，列序已冻结。
- prompt token 总数 1,827,311、最大 799；机械 quote token 总数 124,455、最大 148，均复现。
- 4-BPE、stride 1、claim→window max、answer=max 的高层映射一致；gold 只应在特征冻结后的 evaluator 加入。

## 阻断项

1. **Nested CV/阈值未完全冻结。** 每个 inner 模型的 scaler、answer/claim 权重和类别平衡必须只用三份 inner-train，不能使用完整 outer-train 标签。还需冻结候选阈值、`>=`/`>`、指标权重、AP 实现和单类折处理。
2. **推进门有歧义。** PROTOCOL 的“answer AP 相对 P2 不低于 0.01”与 PLAN 的“最多下降 0.01”不同；exact≥95% 需明确分母为全部 3,776 claims，invalid 计为失败。
3. **现有 hooks 不能直接用于 KV-cache。** 既有 hook 要求 `past_key_value is None`；必须冻结 cache-aware hook 或生成后 full replay，并预注册数值容差和 oracle。
4. **token/停止口径未封死。** literal `<s>` 要明确 `add_special_tokens=False`；必须定义整段 token-prefix 字节重建，不能逐 token 字符串拼接；无闭合标签时 relation suffix 的上下文也需固定。
5. **P0 lineage 不完整。** 尚未锁定 raw 1,024 维矩阵/index 哈希，也未证明缓存没有全 fit 标准化；P0 必须 fold-local scaler/训练。
6. **sanitized manifest 目前只是协议。** 实际物化、哈希并确认 allowlist 前，GPU runner 不能直接读取含 labels 的 `fit.jsonl`。

## 非阻断限制

- Prompt 没有提示“资料不足”，但它强制寻找支持句。exact 只能证明引用来自资料，不能证明资料支持 claim。
- P2/P3 同维，但 teacher-force 与 greedy generation 同时改变文本和轨迹，只能作完整条件对比。
- 没有 P3 generated-surface-only 对照，因此 P3 提升不能单独归因于白盒信号；P1→P2 只检验机械 quote 下的白盒增益。
- PCA 维度和哈希正确，但原回答状态到 quote 轨迹存在分布偏移。

## 执行门

CPU sanitizer/runner 实现可以继续；**GPU smoke、全量 GPU 和评分暂时 BLOCK**，待六项解决并完成 manifest/runner 独立审计。

审查锁：PROTOCOL `1198d893515172def308927023d3c264cdfef960d9bdd050220cfd823749374f`；PLAN `28a94870a4d62c3f4ba2df1e909b7d319a34e6dc4a7f1ee3fd91bf095a185f0c`。

审计说明：未用 GPU、未加载模型、未训练、未读取 baseline 结果或 official test。早期边界检查曾误读第 34,920 行并仅查看 `partition=calibration`；未查看、保留或使用标签/内容，最终复算只用 fit 与 label-blind fit 文件。
