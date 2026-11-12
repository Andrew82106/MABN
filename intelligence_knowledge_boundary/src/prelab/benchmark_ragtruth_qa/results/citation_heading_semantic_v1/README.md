# 标题继承正文的独立来源语义：已准备，未推理/训练

固定复用 `citation_heading_scope_v1` 的 **387 条正文 claim /37 答**，没有重跑或扩大标题解析。每条与原3来源分别配对，共 **1,161 对**：fit 347条/1,041对，calibration 40条/120对。与旧6,252对来源核查的 claim 交集为 **0**。

CPU `prepare` 和 `check` 均已实际 exit0。输入长69–275词元，均值147.17，总170,867词元；全部低于512，无分块、截断或修改原句。只加载本地 tokenizer，没有加载模型或使用GPU。

原210,364窗口的几何文件按字节复用；由该映射重算的继承范围比例与冻结scope列逐值相同。合成检查覆盖max支持度、来源差值、无scope零值和词元平均分母。**真实GPU anchor数值门禁尚未执行，不能把CPU自检称作模型数值回放通过。**

新两列输出契约为 float32 `[210364,2]`：

1. `inherited_any_source_support_mean`：继承正文在原3来源中的最大支持度，按原窗口的可评词元取均值。
2. `inherited_source_support_gap_mean`：上述最大值减去继承来源的支持度，再按相同词元平均。

没有继承scope的claim两列均为0。编号只用于选择对应支持值，不作为数值特征；原句与来源header/body逐字保持。既有8词面列、3scope列、答案、标签和窗口不改。

未来匹配LR设计已单独冻结于 `training_protocol.json`：三个peer×两模式×三个C，共18次。两边共同使用原2风险分数+8词面+3scope+新任意来源支持度（14列）；候选仅再加继承来源支持差（15列）。两模式使用相同新推理预算，沿用634fit/159cal、权重、阈值与选型规则。**当前没有进行这18次训练。**

待根代理单独授权后，GPU入口为：

```powershell
& 'prelab/.venv/Scripts/python.exe' 'prelab/benchmark_ragtruth_qa/src/build_citation_heading_semantic.py' all
```

该命令先核原MiniCheck checkpoint与16023原批次anchor（既有门槛不放宽），再新推理和导出两列；不会训练。当前没有自动等待、排队或占GPU。

这是额外语义核查信号，尚无成绩。之前词面scope失败不能证明这一语义来源差已失败；反过来，补足输入也不保证性能提高。原校准集反复用于开发，不是独立测试。
