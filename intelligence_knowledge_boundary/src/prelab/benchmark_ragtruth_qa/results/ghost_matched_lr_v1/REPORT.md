# 固定参数 GHOST 四维增量：CPU 设计已完成

`src/run_ghost_matched_lr_v1.py prepare-design` 的 session69873 已实际 exit0。当前没有新 GHOST 特征、窗口矩阵或正式拟合；GPU 未使用，也未排队。

第一阶段固定 **C=1e-4，共5个新 LR**：旧 pre-header LB、legacy LB+NLL、post-header LB、HARP64+legacy LB/NLL 各追加相同四维，另报四维单独输入。维度分别1028、1029、1028、1093、4。原四家都已经在旧五档 C 中选中1e-4，因此主增量比较使用相同 C。新模型不再搜索 C；这不是25次匹配搜索。未冻结的25-fit草稿及收窄说明已单独保留，收窄发生在任何新特征/结果出现前。

完整几何和权重检查通过：3680 fit /159 calibration，615/154互斥材料组；653979/42241窗口，风险窗口58433/5984，风险回答1127/100。708506原始BPE对应精确；窗口中的437624次标点槽出现全部参与均值。短窗仅平均实际槽，原标签、拒答政策、整答max均不变。base/loss/class factors逐值等于原expanded权重，fit loss总质量168123。

旧20模型及分数哈希、全量两级指标、双阈值、answermax和原五C选择均 exact；不重新拟合旧模型。旧同C校准成绩为：

| 旧家族 | 窗口F1 | 整答F1 |
|---|---:|---:|
| prefix_pre_header | 0.592998 | 0.852018 |
| legacy_lb_nll | 0.594668 | 0.853333 |
| prefix_post_header | 0.593129 | 0.855814 |
| harp64_legacy_lb_nll | 0.596677 | 0.870370 |

每家保持fit-only、base加权StandardScaler（16384行分块），float32、liblinear L2、seed20260924、maxiter2000、4线程；只用原cal分别确定窗口/整答阈值，同一个模型服务两级。出现收敛问题停止并保留，不加迭代或改参数。

后续明确执行顺序：

```powershell
# 只有3839特征全部完成后才能执行；本轮未运行。
prelab/.venv/Scripts/python.exe prelab/benchmark_ragtruth_qa/src/run_ghost_matched_lr_v1.py prepare-features
# 另行获得实际拟合授权后执行；本轮未运行。
prelab/.venv/Scripts/python.exe prelab/benchmark_ragtruth_qa/src/run_ghost_matched_lr_v1.py fit
```

prepare-features核上游complete、逐答身份、token IDs/原坐标/顺序及NPZ哈希，然后生成float32 `[696220,4]`。缺complete或只有3838条的实际CPU门禁测试均在矩阵创建/拟合之前拒绝，测试中的拟合调用为0。合成检查另验证追加列不改原raw列、标点/短窗均值及小型weighted LR接口；不使用真实数据拟合。

磁盘复用旧8.14GiB raw矩阵，不复制或重扫未消费的旧巨大token缓存。新增四维窗口矩阵10.62MiB；最大临时标准化矩阵2.66GiB，一次只保留一家，完成该家后删除该新临时文件；五份未压缩分数约26.7MiB。预留5GiB。旧同C四次拟合实测合计443.45秒；新五次估计8–20分钟，受收敛和CPU占用影响，这不是新运行实测。

这是四维信号的局部线性适配，不能与原论文RF整答F1直接比较；本项目仍是反复开发的fit/cal，封存test未开。完整源码/输入绑定见 `source_snapshot.json`，旧控制见 `OLD_CONTROL_CHECK.json`，门禁见 `MISSING_FEATURE_GATE_CHECK.json`。
