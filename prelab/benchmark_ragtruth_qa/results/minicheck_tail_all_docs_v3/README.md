v3 修复 v2 的批次形状问题，两个版本的源码与冻结记录分别保留。v2 没有进行真实训练。本轮仍是额外 MiniCheck 核查编码器基线，原 Llama 生成器不训练。

原核查过程逐回答按 claim-major/document-major 顺序组成每批 4 条输入。v2 曾改成每份资料内部的 1–2 条局部批次，自检又使用无 padding 的单条输入；这可能改变浮点计算，不能保证严格重现已保存末层。v3 统一恢复原行序、每批大小、原 padding 长度与不足 4 条的末批，训练及数值自检共用 `pack_original_batch`。先完成原批次 forward，再映回原回答 BPE、跨资料取最小风险 logit，最后计算原标签 BCE。

三批入口状态都来自原完整 all-doc batch4 forward，实际保存的子集分别是：

| 缓存 | 路径 | 保存部分 |
|---|---|---|
| 新 3,046 答 | `fit_expansion/minicheck` | 原 selected |
| 原 793 答恢复 | `fit_expansion/minicheck_backfill_v2` | 原 selected |
| 全开发答缺块 | `fit_expansion/minicheck_unselected_v2` | complement |

新恢复目录额外记录 `original_batch_index/row/size/padding_length`，v3 必须逐对核验；新 3,046 的原缓存通过原计划唯一还原这些信息，并核实际 `batch_padding_length`。不能把仅 complement 重新排批的旧缓存冒充原 all-doc 几何。

此次只改数值回放形状，不调整效果参数。`FAIRNESS_CHECK.json` 证明 v3 与冻结 v2 的全部训练权重数组、3 轮回答顺序、优化器、轮数、损失、聚合和选择规则完全相同。每个对照每轮 460 次更新，共 1,380 次。原/辅助答基础质量各 50%；正负类与组平衡后，原答实际 loss 质量约 52.635%，两个对照使用同一权重。

| 对照 | 训练 | 固定预算 |
|---|---|---|
| frozen0 | 各资料原末层状态上的相同 1,025 参数词元头 | CPU，3 轮 |
| tail2 | MiniCheck 层 22/23 与同一词元头，25,193,473 参数 | GPU，3 轮 |

seed 20261004，积累 8 答更新，AdamW：尾层 lr2e-5、头 lr1e-4、weight_decay .01、梯度裁剪 1。尾层启用原训练 dropout，冻结缓存来自 eval，比较包含实际微调流程的 dropout 差异。仍用同 3,680 fit/159 cal、615/154 分离来源组及原人工标签；4 个原始 BPE 窗口包含标点位置，仅对 lexical 位置取最大概率，整答取全部可评窗最大值。每轮保存完整 fit/cal 分数、固定校准阈值下的指标、完整 fit 加权 BCE、checkpoint 和优化器；第 0 轮仅诊断，不能选为最终模型。

从项目根目录运行：

```powershell
# 协议及 CPU 检查已冻结，此命令只验证
prelab/.venv/Scripts/python.exe -X utf8 prelab/benchmark_ragtruth_qa/src/tail_finetune_all_docs_v3.py verify

# 三批全部完成后，CPU 全量准备及冻结编码层对照
prelab/.venv/Scripts/python.exe -X utf8 prelab/benchmark_ragtruth_qa/src/tail_finetune_all_docs_v3.py prepare
prelab/.venv/Scripts/python.exe -X utf8 prelab/benchmark_ragtruth_qa/src/tail_finetune_all_docs_v3.py train-frozen

# 以下仅由根代理安排 GPU 后执行
prelab/.venv/Scripts/python.exe -X utf8 prelab/benchmark_ragtruth_qa/src/tail_finetune_all_docs_v3.py gpu-smoke
prelab/.venv/Scripts/python.exe -X utf8 prelab/benchmark_ragtruth_qa/src/tail_finetune_all_docs_v3.py train-tail
```

完整 prepare 要求全部 3,839 答、36,777 个文档/陈述对、原输入 ID/末层字符坐标一一对应，冻结实际目录和所有来源哈希；不会拿现有子集开训。GPU 自检固定包含首末 fit/cal、最长输入、首个多资料回答及已知批次数值问题答 13953；这些选择只检查数值回归，与模型成绩或金标无关。原最大绝对差 2e-4、重复差 2e-6 门槛不变。

已通过 CPU tiny 24 层回放、原 batch4 行序/padding、末批大小、两层非重入 checkpoint 梯度、跨句词元映射、权重和原窗口评测检查。新 3,046 答全部 19,699 个 selected 序列的实际 padding 与原 all-doc 计划匹配。随后全量 prepare、严格 GPU 自检及两个对照的 3 轮训练均已完成，保存分数复算通过；GPU 已释放。最终选中尾层第 2 轮：窗口 F1=0.650293、整答 F1=0.838384，详见 `RUN_REPORT.md`，不再重复运行训练命令。
