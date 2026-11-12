# 共享随机事实：本轮核读来源

核验日期：2026-09-19。这里只保存官方来源索引与使用边界，不镜像全文，不宣称下载或读完 PDF。

1. [KU Leuven DTAI：Semantics of Rules with Probabilities](https://dtai.cs.kuleuven.be/problog/tutorial/basic/08_rule_probs.html)。核读网页说明及共享选择/独立实例化示例。应用：同一授权依据被多个动作引用时必须共享随机节点，不能把每次使用都重建为独立事实。不是 MAS 风险实验或新安全保证。
2. [KU Leuven DTAI：Bayesian networks](https://dtai.cs.kuleuven.be/problog/tutorial/basic/02_bayes.html)。核读网页的随机事实、条件规则及共同父节点示例。应用：确定性约束与不确定事实可以放在同一概率结构中。本项目使用小图枚举与已有 BN 状态引擎，没有复现完整 ProbLog 求解器。
3. [Koller / Pfeffer：Object-Oriented Bayesian Networks](https://arxiv.org/abs/1302.1554)。复核作者摘要与 UAI 1997 会议元数据。应用：片段与层次复用属于已有方法；不把“模板子 BN”当作新发明。2013 年是 arXiv 上传年份，不是原会议年份。

对应实现说明：`paperAlpha/research/共享授权节点BN_联合结构与可学习性_20260919.md`。本轮受控频数、梯度和结构算例是本项目实验，不是以上来源的结论或复现实验。
