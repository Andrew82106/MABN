本轮提供完整的额外核查编码器训练对照，尚未启动真实训练。原生成 Llama 的参数及 RAGTruth 人工标签均不改。

两种模型都先对每份资料单独处理完整回答，再把核查词元状态按非空白字符映回原 Llama 词元；跨资料取最小风险 logit，之后才使用原标签计算二元训练误差。不会给单个资料块编造标签。每个原词元可以由不同资料支持；需要联合多个块才能推导的事实仍是此聚合的限制。

| 对照 | 可训练部分 | 预算 |
|---|---|---|
| frozen0 | 相同的 1,025 参数词元分类头，使用各资料原末层状态 | CPU，3 轮 |
| tail2 | MiniCheck 最后两层和同一词元头，共 25,193,473 参数 | GPU，3 轮 |

共同设置：seed 20261004，积累 8 答更新一次，尾层学习率 2e-5、头学习率 1e-4，AdamW weight_decay .01、梯度裁剪 1。每个来源组原 634 答与新增 3,046 答各占一半基础训练质量，随后依原规则做正负类平衡及组等权；标签、4 个原始 BPE 的滑动窗口及整答最大值不变。原 159 答仅用于每轮选阈值及最终轮次，官方测试集保持封闭。第 0 轮只诊断，不能选为最终模型。

每轮保留完整 fit/cal 词元预测、窗口和整答分数、固定校准阈值下的两组指标、完整 fit 加权 BCE、checkpoint、优化器与 RNG。冻结编码层与微调两层都使用全部 3,680 答，二者比较才是同数据量的微调收益；与旧 634 答实验比较同时涉及数据扩充。微调尾层启用原 dropout，冻结缓存来自 eval，因此对比包含实际微调流程的 dropout 差异。

入口：`prelab/benchmark_ragtruth_qa/src/tail_finetune_all_docs.py`。在项目根目录依次运行：

```powershell
# CPU 检查与协议冻结；freeze 已存在时只核对，不覆盖
prelab/.venv/Scripts/python.exe -X utf8 prelab/benchmark_ragtruth_qa/src/tail_finetune_all_docs.py tiny-test
prelab/.venv/Scripts/python.exe -X utf8 prelab/benchmark_ragtruth_qa/src/tail_finetune_all_docs.py freeze

# 等三个缓存来源全部完成后，核全量身份、哈希、完整 C×D 和坐标，生成逐资料末层特征
prelab/.venv/Scripts/python.exe -X utf8 prelab/benchmark_ragtruth_qa/src/tail_finetune_all_docs.py prepare

# CPU 冻结编码层对照
prelab/.venv/Scripts/python.exe -X utf8 prelab/benchmark_ragtruth_qa/src/tail_finetune_all_docs.py train-frozen

# 以下两步仅由根代理安排 GPU 后运行
prelab/.venv/Scripts/python.exe -X utf8 prelab/benchmark_ragtruth_qa/src/tail_finetune_all_docs.py gpu-smoke
prelab/.venv/Scripts/python.exe -X utf8 prelab/benchmark_ragtruth_qa/src/tail_finetune_all_docs.py train-tail
```

正式 prepare 要求全 3,839 个开发回答、36,777 个资料/陈述对齐全，每个真实块完整覆盖回答。原 selected、补 selected 和未选块缓存分别绑定原计划、资料打分、答案与数组哈希；未选末层的 `token_claim_index/token_document_index` 与 selected 的逐词元键显式适配。缺块不以 padding 代替，也不允许拿现有子集开训。

算法与数据协议已冻结在 `protocol_freeze.json`，实际缓存清单由后续 prepare 再冻结。如果原 793 的补缓存另存恢复目录，prepare 及其后各命令可附 `--backfill-dir prelab/benchmark_ragtruth_qa/fit_expansion/实际目录`。此选项仅绑定物理路径，不改变任何答案、资料选择、坐标或数值门槛；prepare 完成后路径不可再变。

GPU 自检固定检查首末 fit/cal、最长输入回答、首个多资料回答，按同一文档的输入位置核对原末层，最大绝对差门槛 2e-4；重复差门槛 2e-6。然后用最长回答和人工构造的零目标检查 detached-cache、非重入 checkpoint、梯度与真实显存。自检模型丢弃；不会用自检结果选择学习率、结构、标签或阈值。失败保留独立报告并停止。

已通过 CPU tiny 24 层回放、两层梯度隔离、真实训练函数的两文档两陈述聚合、跨句词元映射、标点窗口及同头平均等价检查。真实新答 16021 的缓存键、同文档输入 ID/末层位置和字符覆盖检查通过；这只是接口样例，不表示全量缓存已通过。使用合成分数核过原校准 42,241 个窗口及各指标逐值不变；训练窗口数为 653,979。

缓存 encoder22 约 38.33 GiB，逐资料映射后的末层还需数 GiB 磁盘。训练仅加载最后两层；完整模型前 22 层和原生成模型不加载到 GPU。真实显存峰值、耗时由后续 gpu-smoke 记录。3 轮训练外，每轮另做完整 fit/cal 评估，因此不能直接用一次 forward 时间当总训练耗时。
