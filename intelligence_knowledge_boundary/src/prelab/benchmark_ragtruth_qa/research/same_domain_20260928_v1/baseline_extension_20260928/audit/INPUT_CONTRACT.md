# 六基线共同评测输入契约

新方法输出 validation 271 条、test 279 条 JSONL，每条为 `{"response_id": "...", "token_scores": [0.1, ...]}`。风险分数越大越可疑，可为未校准的实数，不要求概率。数组必须包含原始回答的全部 Llama BPE，长度与 `data/{partition}/geometry.jsonl` 的 `lexical_mask` 完全一致；不预先去标点、不筛选无暗示子集、不遗漏样本。适配器应以原始回答字符偏移将其他分词器/句子分数映射回此坐标。MiniCheck 的句级分数广播只代表句级定位，不能称原生词元检测。

每个方法的配置包含：`name`、`validation_predictions`、`test_predictions`、`model_artifacts`（检查点/训练配置/适配说明/运行 receipt 的路径列表），以及可选 `score_field`（默认 token_scores）、`scores_are_windows`（默认 false）、`pooling_candidates`（默认 [mean,max]）、`native_granularity`、`training_data`、`adaptation`。路径建议绝对路径。窗口分数若直接提供，必须包含且只包含共同有效窗口，使用 `scores_are_windows=true`。

共同评测：4 BPE 窗口、步长 1，去除全非字母数字窗口。max 只取有效词元，mean 保留窗口内全部词元。窗口标签是窗口内任一有效错误词元；回答分数取窗口最高分。answer/window 阈值分别只在无暗示 validation 135 条上选择：最大 F1，并列精确率，再并列高阈值。窗口适配按 `[min(answer F1, window F1), window F1, window precision]` 选择，并列优先 mean。

主测试固定为无暗示 155 条/75 来源组，全测试 279 条/133 来源组作补充。当前测试标签此前已经解封；本扩充是固定协议的补充比较，不称新的盲测。我们的方法固定原 Fusion-window-HGB，不因新增基线结果重新选择。

命令（CPU）：

```
python -X utf8 evaluate_extensions.py audit-existing
python -X utf8 evaluate_extensions.py freeze-new --config new_methods.json
python -X utf8 evaluate_extensions.py evaluate
```

`new_methods.json` 格式是 `{"methods":[上述配置, ...]}`。freeze-new 不读取测试标签，先按验证选择适配和阈值，再检查、绑定全部预测、模型产物及数据哈希；evaluate 仅消费冻结文件，不重新挑参数。既有 Lookback-default/tuned、RAGognizer-NF4 和 HGB 直接复用原冻结预测并独立重评分，默认版只作补充，不计成第七个独立基线。
