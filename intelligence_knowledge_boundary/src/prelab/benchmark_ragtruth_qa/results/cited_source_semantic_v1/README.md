# 被引用来源的语义支持度：推理及匹配对照已完成

旧 MiniCheck 缓存完整保存原793答、8852个自动陈述、11160个“陈述×资料块”支持分数，但资料块不是独立来源：601答只有一个混合块，192答有两个块。985块中786含三来源、10含两来源、189只与一个来源正文相交（通常只是末段）。原 `support_by_claim_document`、selected encoder22和未选encoder22/末层缓存均沿用该块轴，不能直接当作 passage1/2/3 支持度。

本入口保持全部原答和自动陈述，按已有引用解析选出2084个含明确引用的陈述（fit1640、cal444，来自558答），每条对三个独立原来源分别核查。每份来源保留其原header/body及空白；原陈述包括引用文字，未编辑。6252个输入最长299个RoBERTa词元，总792250词元，零截断、无需分块。CPU仅做了分词和几何核验，没有加载模型、推理或训练。

输出为原210364个4原始BPE窗口的四列float32特征：

1. 有至少一个有效来源编号的陈述所占词元比例。
2. 任意一个来源的最大支持度；无有效引用的陈述取0。
3. 任意来源最大支持度减去被引有效来源最大支持度；无有效引用取0。
4. 第3列乘以原窗口引用字符重叠比例。

第1–3列按原lexical词元→陈述映射求窗均值，标点仍计入4BPE窗口长度。未知格式及无效编号不自动判错；多引用取max只是输入信号，不能当作多来源联合蕴含。原窗口顺序、全部答案和引用重叠值已精确核对，未访问标注字段用于构造；没有读取封存测试。

冻结文件：`preparation_freeze.json`、`protocol.json`；CPU准备结果：`CPU_SELFCHECK.json`、`preparation_statistics.json`。全量准备会话50941实际exit0；此前首轮仅synthetic样例的`passage1`不符合原parser的`passage 1`格式而停止，修正样例后重跑；未改变真实输入、旧parser或已有实验。随后根代理明确分配GPU，推理会话37196、PID45952实际exit0：6252对耗时83.15秒，原anchor logits/支持分数均差0。全部四列已导出，GPU已释放。

本次已执行命令如下；产物完整，不应重复覆盖：

```powershell
prelab/.venv/Scripts/python.exe prelab/benchmark_ragtruth_qa/src/build_cited_source_semantic.py all
```

`all`仅执行固定MiniCheck推理及CPU特征导出，不拟合探针。也可分别执行`infer`、`export`；`check`只核冻结哈希，不使用GPU。推理先以原第一条完整输入回放固定checkpoint数值门禁，然后保存每对logits/支持分数和精确输入hash。完成后才产生`features_complete.json`及`window_features.npy float32[210364,4]`。

按旧CPU24对计时简单外推约42.7分钟，按旧GPU全量每对均时外推约2.9分钟；新来源输入更短，二者只是运行前估计。此前完成的citation_alignment只使用词面覆盖；本方案增加额外语义核查，不能称为原生成模型的原生白盒特征。

配套CPU固定18LR会话56365实际exit0，拟合/评分计时7.86秒。结果在相邻 `cited_source_semantic_lr_v1/REPORT.md`，全部候选保留，保存分数的两级计数/阈值/选型/整答max复核通过。共同12列控制→追加来源差2列的cal窗口/整答F1为：Lookback .646873/.867925→.646272/.871287；HARP .679646/.870813→.679828/.868687；semantic .660664/.858639→.660241/.858639。未见来源对应信号带来实质定位增益，不能将HARP的微小涨幅当作突破。测试始终未开。
