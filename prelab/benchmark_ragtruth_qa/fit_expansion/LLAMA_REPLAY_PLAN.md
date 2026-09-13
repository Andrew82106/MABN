# 新增训练回答的统一 Llama 重放

最终入口：`run_llama_expansion_v3.py`；输出：`llama_features_v3/`。目前仅 CPU 准备和自检，GPU 等 root 安排，排在 MiniCheck 扩充及补缓存之后。

新增 3,046 份公开回答仍来自原有 634 个训练来源、615 个分组，没有新增独立题目或改动校准/测试划分。回答来自另外五个生成器；特征全部由同一个固定 Llama-2-7B-Chat NF4 模型重放取得，不能称为各原生成器的原生内部状态。

每份回答只做一次 backbone forward，保存原 LB（1,024）、NLL（1）、最后层状态（4,096），以及四组上下文范围×读取时刻的 LB（各 1,024）；浮点字段全为 float32。原 token IDs、位置及 raw/clipped 字符坐标全部保留。`lb` 对应旧 `lb_source_post_legacy`，不重复保存同一数组。

全部冻结词元视图已用同一分词器重新核对一致。共 495,347 个答案词元，最长输入加答案 1,207 词元，未超过 4,096。3,044 条首词元 raw offset 为 -1；另外两条为 0，均不丢弃。

GPU 启动前比较两条原 fit 工程样本：旧 LB/NLL/最后层状态与四组新 LB 均须精确等于既有缓存。新增样本用两条输入长度极端加一条 raw offset=0 样本（去重）重复提取，所有数组必须精确一致。样本只由长度、坐标和 ID 选择，不使用标签或检测分数。

未压缩数组约 **17.03 GiB**，预留 **40 GiB**。基于刚完成的 793 条五组 LB 提取（约 26 分钟），按输入长度×答案长度估算新批次 **59–80 分钟**，已加 NLL 和状态导出的余量；这仍是估计。

```powershell
prelab\.venv\Scripts\python.exe -X utf8 prelab/benchmark_ragtruth_qa/fit_expansion/run_llama_expansion_v3.py run
```

只在 root 明确交还 GPU 后执行。支持按回答续跑、源计划和文件哈希校验；全 3,046 条完成后才供训练。此脚本不生成回答、不读取标注值、不拟合探针，也不打开封存测试。

v1 保留 CPU 预检中缺少 synthetic response_id 的记录；v2 CPU 已通过。v3 在不改特征公式的前提下加入 root 要求的 raw offset=0 生产自检样本。前两版文件没有覆盖，后续只使用 v3。
