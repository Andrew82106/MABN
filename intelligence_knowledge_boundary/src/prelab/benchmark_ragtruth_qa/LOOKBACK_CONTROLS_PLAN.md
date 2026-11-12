本对照用于分清“资料范围”“取状态时刻”“答案头”三处实现差异。旧 QA 结果、原回答、人工标签和官方测试集均保持原样。当前只完成 CPU 准备；GPU 等主任务调度。

| 保存字段 | 注意力资料范围 | 查询位置 | 回答／new-token 池 |
|---|---|---|---|
| `lb_source_pre_header` | 检索资料 | 当前词元之前 i−1 | chat 结束头＋此前回答 |
| `lb_prefix_pre_header` | 完整任务／问题前缀 | 当前词元之前 i−1 | chat 结束头＋此前回答 |
| `lb_source_post_header` | 检索资料 | 当前词元 i | chat 结束头＋回答至当前词元 |
| `lb_prefix_post_header` | 完整任务／问题前缀 | 当前词元 i | chat 结束头＋回答至当前词元 |
| `lb_source_post_legacy` | 检索资料 | 当前词元 i | 仅回答至当前词元；旧版参照 |

四格控制共享同一答案头，所以范围和时刻可以分别比较。第五格与原缓存每个元素都须精确相等；它与 `source_post_header` 的差异只来自是否将答案头纳入 new-token 池。`prefix_pre_header` 最接近官方代码的范围／时刻定义，但不称原版实验的精确复现。

原提示和答案全部一起沿用已冻结分词，不重新分词、生成或截断。完整前缀包括 BOS、开头 chat 标记、任务要求、问题、检索资料和原提示中的 `output:`，止于 chat 结束头之前。模型仍接收全部原文本。这里的答案头是原文本 ` [/INST] ` 中尚未进入回答的实际词元，而不是另行分词的字面字符串。

793 条实际答案头都是四个词元 `[518, 29914, 25580, 29962]`。首个回答词元仍包含头后空格，原始坐标从 −1 起；793 条全部保留，不能把该词元误划入头部。预测首个回答词元前，new-token 池由四个头词元组成，不会出现空均值。

每个头的特征始终是 `mean(context attention) / (mean(context attention) + mean(new-token attention))`。查询点自身的注意力保留；pre-read 的查询点是前一个回答／头词元，post-read 的查询点才是当前回答词元。后续内容由因果遮罩排除。全文回放只执行一次 backbone；每层捕获 q/k 后读取两组查询行，得到五份 float32 数组，不保存大 hidden。

[官方固定代码](https://github.com/voidism/Lookback-Lens/blob/e0a1fa3a898fbf6512af7be5567dea8ffe7a6620/step01_extract_attns.py)使用 `#Answer#:` 头及预测当前词元前的注意力。我们的实际 chat 头和 `output:` 位置不同；为了保持原发布输入和人工答案，不替换它们。原生未量化生成 trace 也不可得；本轮仍是固定 Llama2-7B 家族权重的 NF4 教师强制重建。

范围固定为 fit 634、calibration 159，共 793 回答、213,159 个原始回答词元；官方 test 和关联 withheld 内容不读取。未来评分沿用 `development_v1` 同一人工 gold、4 raw BPE／stride1、按组和回答加权、fit-only scaler、LR 的 C 候选 `{.001,.01,.1}`，每个控制仅使用 LB1024；不混入 NLL 改变这次控制目的。每格分别用原校准规则定阈值及 C，全部结果公开；增加控制后的校准成绩仍偏乐观，不能当最终测试成绩。

CPU 已通过两种微型随机架构的直接注意力 oracle：五格误差最大 `5.96e−8`，旧参照精确一致，同长度未读后缀替换下已有前缀特征精确不变。全部 793 布局、旧缓存哈希和坐标检查通过。真实 GPU 后续先以冻结顺序首／末两例验证重复精确一致、旧参照逐值一致；生产每一条也必须通过旧参照检查，失败时停止，不放宽误差门槛。

每条 NPZ 保存五份 `float32[N,1024]`，加原 `token_ids`、`answer_token_positions`、原始／裁剪字符 offsets。JSON 绑定原计划、当前布局、提取签名和 NPZ 哈希；可逐条续跑。全部 LB 原始大小约 4.07 GiB，使用压缩 NPZ，预留 7 GiB 空间。预计 793 次生产 forward 加两次重复自检；GPU 15–30 分钟只是依据旧提取耗时的预算，尚未实测。

文件：[代码](D:/Projects/Multi_Agent_Graph_Analysis/prelab/benchmark_ragtruth_qa/src/feature_lookback_controls.py)、[冻结协议](D:/Projects/Multi_Agent_Graph_Analysis/prelab/benchmark_ragtruth_qa/lookback_controls_protocol.json)、[CPU 自检](D:/Projects/Multi_Agent_Graph_Analysis/prelab/benchmark_ragtruth_qa/data/lookback_controls_v1/cpu_selfcheck.json)、[793 条准备记录](D:/Projects/Multi_Agent_Graph_Analysis/prelab/benchmark_ragtruth_qa/data/lookback_controls_v1/preparation.json)。模型与输入签名、逐条布局也位于该新数据目录。

运行命令：`prelab\.venv\Scripts\python.exe -X utf8 prelab/benchmark_ragtruth_qa/src/feature_lookback_controls.py prepare` 或 `cpu-check` 均不启用 GPU；只有主任务排队授权后的 `run` 才加载模型。评分实现与实际拟合另行进行，本次没有重新训练探针。
