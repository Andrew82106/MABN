# 标题继承语义：18个LR入口已准备

入口：`src/train_citation_heading_semantic.py`。`prepare`、`check` 已实际成功；语义特征未完整时试运行 `run` 正常返回等待，未创建 `started.json`、模型或分数文件。**当前真实拟合数0，GPU使用0。**

固定三对象 × 两模式 × 三个C，共18个LR。两模式共同14列：原2风险分数、8词面列、3标题scope列、新任意来源支持度；候选只追加继承来源支持差为第15列。两边都使用同一1,161对新语义推理。原13列scope全部9候选及选中模型/hash另外保留为历史参考，不当作这次14列匹配控制，也不重训。

合成自检通过：原13列未变、14列为15列精确前缀、相同any-source输入、无scope新语义归零、scaler只fit、整答取同一最终窗口向量的max。

真实拟合前必须通过生产器的完整标记、1,161对/387条、GPU原anchor门禁、所有缓存/几何/来源哈希、原窗口顺序与no-scope零值检查。之后沿原634fit/159cal、窗口权重、两级阈值与选型规则，全部18候选均保存。没有测试访问或额外调参。

待特征完成且根代理授权后：

```powershell
& 'prelab/.venv/Scripts/python.exe' 'prelab/benchmark_ragtruth_qa/src/train_citation_heading_semantic.py' run
```

不重新prepare，不启动GPU。来源语义与LR均属于重复开发对照，不是独立测试。
