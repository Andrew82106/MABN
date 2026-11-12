# QA 最终测试的数据与计分接口

当前只完成校准模拟，**没有读取、导出或解封官方测试回答、质量或标签**。真实测试的最终模型及阈值由主管代理另行冻结；本目录不训练或选择模型。

## 已完成验证

现有 159 条校准回答及 3 条已知质量排除记录完成往返验证。全部 token 坐标、人工标签和 42,241 个有效 4 BPE 窗口与原导出完全一致；固定旧 LR 的窗口指标、159 个整答最大值和整答指标完全一致。独立审查另核了 154 个组的 bootstrap。未使用 GPU。

`simulation_calibration/SELFTEST_COMPLETE.json` 保存自检结果。五个处理脚本与 v1 字节相同；v2 只补充正式解封前对原开发 manifest、特征 manifest 和 token plans 的直接哈希校验，详见 `REVISION.json`。

## 稳定接口

1. 主管代理在全部开发结束后提供完整、不可变的开发冻结文件：所有终模型及变换/代码哈希、固定窗口与整答阈值、选型记录、主比较身份及 bootstrap 参数。`development_freeze.example.json` 只是不可运行的空示例。
2. 主管代理明确解封后，另外写 `data/FINAL_TEST_RELEASE_AUTHORIZATION.json`，绑定开发冻结、固定 150 个测试身份和本工具协议哈希。工具没有授权命令；示例 `authorized=false`。
3. `export_qa.py release` 才读取指定官方测试记录，保存原记录和官方质量排除理由，生成无标签的模型输入、原 token 布局及独立金标文件。
4. 单独冻结的模型适配器输出每个方法的完整窗口分数及预测 manifest；该适配器尚未在这里编造或实现。每行分数为 `window_id/response_id/group_id/score`，完整 schema 见 `score_frozen.py` 和校准模拟 manifest。允许真实 token、窗口或 claim 模型，但一律映射到相同有效窗口；不能把窗口分数冒充原 token 概率。
5. `score_frozen.py` 只应用固定阈值，整答取全部有效窗口最大分数；输出全部冻结方法、P/R/F1、混淆计数、AUROC/AP 和按组抽样的区间，没有拟合、挑阈值或挑最好方法的逻辑。

正式调用（当前**未执行**，必须先满足以上授权与冻结）：

```powershell
prelab\.venv\Scripts\python.exe -X utf8 prelab/benchmark_ragtruth_qa/final_test_tools_v2/export_qa.py release --freeze <完整开发冻结文件> --out prelab/benchmark_ragtruth_qa/data/final_test
prelab\.venv\Scripts\python.exe -X utf8 prelab/benchmark_ragtruth_qa/final_test_tools_v2/score_frozen.py --freeze <同一开发冻结文件> --bundle prelab/benchmark_ragtruth_qa/data/final_test/bundle_manifest.json --predictions <冻结适配器的预测manifest> --out prelab/benchmark_ragtruth_qa/results/final_test
```

## 固定口径和限制

只按官方 `quality == good` 评测，人数在解封前未知。保留全部四类人工 span 及 `implicit_true/due_to_null`；与有字母或数字的字符相交才构成风险 token。窗口是原始 4 BPE、步长 1；短答保留唯一短窗，只有完全无词汇字符的窗口排除。无人工风险标记的合格拒答仍参与负例评测，这与 R16 的拒答定位规则不同，不能直接把跨数据 F1 差解释成优化效果。

缺失预测、无效坐标或合格回答没有有效窗口时停止完整计分，不删困难回答或补 0。已有结果禁止覆盖。来源分组不是穷尽所有现实事件关联的证明；真实推理是否使用了声明权重仍需要独立核查终模型适配器。
