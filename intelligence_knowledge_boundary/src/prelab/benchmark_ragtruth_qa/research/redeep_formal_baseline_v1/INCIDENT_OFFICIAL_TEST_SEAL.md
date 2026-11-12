# 作者附带测试材料访问事件

日期：2026-09-13。

在判断作者缓存能否与本项目数据精确复用时，我曾让一个本地 Python 进程整体反序列化作者仓库中的
`ReDeEP/log/test_llama2_7B/llama2_7B_response_chunk.json`。该进程的目的仅是检查文件结构、任务分布和文本哈希连接；终端只展示过一条 Summary 样例，没有展示任何 QA 回答、QA 标签或测试指标。进程确实在内存中载入了整个文件，也计算过它与当前 fit/cal 文本的精确连接数。因此，即使没有用其中任何 QA 标签、回答、分数或统计来选择方法、参数和协议，也不能再声称原有 official QA test 对本轮研究保持“从未打开”的严格封存状态。

发现后立即停止访问。此后没有再次打开该文件，也没有读取 `final_test_tools`、`final_test_tools_v2` 或任何 test 标签。当前 ReDeEP 的公式、候选头、层/头排序规则、固定 K/β、输入模板和统一映射均只来自论文、作者源码、作者固定参数文件以及本项目 fit/cal。

处置如下：

- 原 official QA test 的严格封存身份失效并退出最终无偏评测；它不能用于方法选择，也不能再用于最终结论。
- 后续最终结果必须使用重新建立、且本轮开发完全未触碰的独立 holdout。
- 本目录代码把允许输入硬限制为 `feature_inputs.jsonl` 中的 RAGTruth `official_split=train`、partition 为 fit/calibration 的 3,839 条回答。
- `FORMAL_BASELINE_RESULTS.md`、`BASELINE_PROTOCOL.md` 和 `CURRENT_STATUS.md` 中凡声称旧 official QA test “仍封存/未打开”的文字都需要由主任务统一更正。本研究目录不直接修改这些公共文件，以避免并发冲突。

