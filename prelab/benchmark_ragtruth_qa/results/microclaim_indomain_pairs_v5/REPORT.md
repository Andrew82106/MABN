# Expanded-v5 in-domain microclaim pairs

本阶段只用 expanded-v4 的 fit 数据构造训练对；没有读取 calibration/test、没有训练模型、没有改 v4 或 baseline。

## 结果

- 选出 **56** 对，覆盖 41 个错误微主张、52 个安全微主张。
- 五折 pair 数：{'0': 6, '1': 12, '2': 15, '3': 11, '4': 12}；每个 OOF 模型只用 held_fold 不等于自己的 pair。
- 主槽位：{'negation': 3, 'object': 3, 'quantity': 48, 'source': 1, 'temporal': 1}。
- 表面相似度：最小 0.539，中位 0.643，P90 0.836；阈值只放行内容合格候选的 1.39%。
- 所有 pair 的 source、问题和底层三篇资料完全一致；top-2 打包证据也完全相同的有 13 对。
- 只覆盖全部错误微主张的 1.18% 和 36/615 个 group。

## 歧义控制

剔除原因见 AUDIT.json。最关键的是剔除了 31 个“文本相同但投影标签相反”的候选，以及 19 个局部 gold 与实际差异槽位对不上的候选。这些剔除只判断配对结构，不重新判断事实真假。

## v5 训练接法

保留 v4 的全部 BCE；另加 `0.25 * softplus(risk_safe - risk_error)`。pair 按 source group 等权，OOF 训练严格排除本折 pair。这样直接惩罚“同资料、同关系骨架，只改一个关键槽位却仍判安全”的情况。

额外 pair pass 预计为 v4 训练 token 的 **0.3%**，六个模型合计约 **0.2 分钟**（规划值，尚未跑 GPU）。

## 限制

自然严格 pair 很少，因此它只能作为低权重定向修正，不能替代 34,919 条 BCE 主监督。RAGTruth 没有 gold 支持句或关系三元组。这里的“主体/谓词/槽位”由确定性词法规则识别；标签仍完全来自原 gold span 投影，所以这些 pair 是训练监督，不是新的事实证书。
