# Retrieved-evidence NLI citation v2

v2 不改变 v1 的检索、claim、NLI 模型或概率，只新增引用条件读出。它分别汇总被引用来源与未引用来源的 E/N/C、覆盖和 BM25，再加入支持缺口、冲突差和引用状态。单一 C=0.1，source-group 五折 OOF；fit 定阈值，calibration 只报告。

另预先冻结一个 0.5/0.5 的 logit 平均融合，不做权重网格。由于绑定的当前白盒候选有既往 calibration 选型和 fit 非完整 OOF 历史，融合只作开发诊断，citation v2 单模型才是主结果。CPU prepare/check 不需要 NLI 缓存；缓存完成后另行 score。
