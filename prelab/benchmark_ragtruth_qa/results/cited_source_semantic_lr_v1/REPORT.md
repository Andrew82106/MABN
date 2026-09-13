# 被引来源语义分数的匹配对照

两模式都使用相同的新三来源推理结果、原8词面特征和相同两分数；区别仅为是否加入被引来源支持度差及其引用位置交互。

| 底层对象 | 模式 | C | 窗口F1 | 整答F1 |
|---|---|---:|---:|---:|
| lookback | source_agnostic_control | 0.001 | 0.646873 | 0.867925 |
| lookback | cited_source_gap | 0.01 | 0.646272 | 0.871287 |
| harp_claim | source_agnostic_control | 0.1 | 0.679646 | 0.870813 |
| harp_claim | cited_source_gap | 0.1 | 0.679828 | 0.868687 |
| semantic_claim | source_agnostic_control | 0.001 | 0.660664 | 0.858639 |
| semantic_claim | cited_source_gap | 0.001 | 0.660241 | 0.858639 |

全部793答及原210364个4BPE窗口保留；634fit只训练、159cal选阈值与C。不是只评有引用的句子，不是新人工标签。
引用可能只是提及来源，单源支持也不等于多源联合支持；这些是输入信号，不直接判标签。
这是已有模型训练内分数上的组合，非交叉拟合；反复开发校准成绩不能作为独立测试或SOTA证据。
