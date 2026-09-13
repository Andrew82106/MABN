# Exact subset attribution v1：CPU 审计

通过。3,839 条回答都冻结了 8 个精确资料子集；full 视图逐条等于已有原始重放坐标，所有视图保持同一答案 token 和字符偏移。prepare/check 不调用标签加载器或模型加载器。

GPU 与 score 是独立命令。score 固定使用 source-group 五折 OOF LR，阈值只取 fit；统一 4-BPE 窗口取 lexical token 最大值，整答再取窗口最大值。基线目录和算法均未修改。

预计执行 30,712 次顺序前向，共 13,700,732 个输入 token；最长 1232 token。GPU smoke 尚未运行。
