# 引用来源对应特征的匹配对照

三个底层对象、两种输入，各相同三档C；原训练/校准窗口和标签全部保留。

| 对象 | 输入 | C | 定位F1 | 整答F1 |
|---|---|---:|---:|---:|
| lookback | two_scores_only | 0.001 | 0.6504 | 0.8664 |
| lookback | two_scores_and_citation | 0.001 | 0.6475 | 0.8676 |
| harp_claim | two_scores_only | 0.1 | 0.6783 | 0.8627 |
| harp_claim | two_scores_and_citation | 0.1 | 0.6797 | 0.8687 |
| semantic_claim | two_scores_only | 0.001 | 0.6679 | 0.8619 |
| semantic_claim | two_scores_and_citation | 0.001 | 0.6613 | 0.8586 |

来源覆盖仅为词面特征，不据此自动重打标签；有无引用的全部回答都进入评测。底层训练分数不是交叉拟合，因此仍可能过拟合。
本表仍是反复开发的159答，不能作为独立测试或SOTA结论。还须与此前更强的固定权重/单调树组合比较，不能只报弱两分数LR对照。
