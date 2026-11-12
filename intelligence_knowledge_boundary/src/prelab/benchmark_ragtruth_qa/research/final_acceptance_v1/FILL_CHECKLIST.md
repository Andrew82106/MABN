# RAGognizer 正式结果回填清单（已完成）

最终回填见 `REPORT.md` 和 `ACCEPTANCE_STATE.json`；本文件保留为操作记录。

只在以下三项都有证据时，把 provisional 改成正式结果：

1. 原生推理覆盖为 `3839/3839`，且 full raw manifest 完整。
2. 无标签适配、score freeze、评分以及独立复算均通过，模型/代码/输入/输出 hash 链闭合。
3. 正式数值来自同一 cal159、42,241 个 4-BPE 窗口和 159 个整答，使用统一 F1Opt 规则。

回填步骤：

1. 在 `ACCEPTANCE_STATE.json` 的 RAGognizer 行写入正式 `status`、`formal_coverage`、`formal_independent_audit` 以及窗口/整答的 AUROC、AP、F1。
2. 在 `REPORT_TEMPLATE.md` 中替换全部 `{{...}}` 字段。
3. 分别计算：
   - `WINDOW_MARGIN = 0.6902813989031736 - max(全部正式基线 window F1)`
   - `ANSWER_MARGIN = 0.8910891089108911 - max(全部正式基线 answer F1)`
4. 两个差值均大于或等于 0，且 RAGognizer 全量审计通过，填 `开发集验收通过`；否则填 `未通过` 并指出超过候选的基线与粒度。
5. 无论结果如何，保留“反复使用的 calibration 开发证据、新 holdout 尚未测”的限制。

当前四项已完成基线的最大值都是 Lookback Lens：窗口 0.600882、整答 0.845455。RAGognizer provisional 的两项 F1 都低于候选，但不能提前用于最终判定。
