# 整答风险对局部窗口的匹配校正

三个已完成的局部组合都比较相同六档权重。控制只用自身整答分数，候选共同加入已完成Lookback树的整答分数。

| 对象 | 整答来源 | alpha | 窗口F1 | 整答F1 |
|---|---|---:|---:|---:|
| lookback | own_answer_control | 0 | 0.647496 | 0.867580 |
| lookback | shared_lookback_tree_answer | 0 | 0.647496 | 0.867580 |
| harp_claim | own_answer_control | 0 | 0.679660 | 0.868687 |
| harp_claim | shared_lookback_tree_answer | 0 | 0.679660 | 0.868687 |
| semantic_claim | own_answer_control | 0 | 0.661283 | 0.858639 |
| semantic_claim | shared_lookback_tree_answer | 0 | 0.661283 | 0.858639 |

原159答校准集已反复使用，仅为开发结果；官方测试未读。最终整答仍取全部窗口最大值，未分开挑两套输出。
没有新训练或推理；所有旧选中模型固定，上游训练内分数没有交叉拟合。各窗口同加整答logit不改变答内定位顺序。
