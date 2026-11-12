# 知识模板与模块化 BN：补充来源

核验日期：2026-09-19。本目录保存本轮新增/复核的来源索引，不移动旧文献，不声称已下载PDF或全文通读。

补充[共享随机事实来源索引](shared_probabilistic_facts.md)：ProbLog 官方教程中共享选择与独立实例化的区别，支持共享授权节点设计；不将已有概率建模思想称为本项目原创。

1. **Daphne Koller, Avi Pfeffer. Object-Oriented Bayesian Networks. UAI 1997.**
   [作者提交的arXiv条目](https://arxiv.org/abs/1302.1554)、[Stanford作者PDF](https://ai.stanford.edu/~koller/Papers/Koller%2BPfeffer%3AUAI97.pdf)。本轮核读摘要及会议元数据：对象类复用概率片段、支持层次组合。arXiv上传时间2013不是论文提出时间。支持设计来源，不是MAS安全效果证据。
2. **Peiran Wang, Ying Li, Yuan Tian. Aligning Provenance with Authorization: A Dual-Graph Defense for LLM Agents. 2026.**
   [版本v1](https://arxiv.org/html/2605.26497v1)。本轮核读§3.2–3.4：运行来源描述与授权参照分离；不把模型声称当独立授权。不复用其实验数字作为本项目结果。
3. **Xabier Muruaga. Bounded Agents: Delegation Security for Multi-Agent AI Systems. 2026.**
   [版本v1](https://arxiv.org/html/2608.15888v1)。本轮核读§4.2–4.4：范围缩小、基础设施维护的累计预算和约束条件。我们的请求额监测不具有该工作执行边界下的保证。

应用位置：`paperAlpha/configs/action_mechanism_templates_v1.json`和`paperAlpha/research/知识子图BN_v7_授权与额度分离_20260919.md`。本轮还复核了[Prompt Infection §3.1–3.2](https://arxiv.org/html/2410.07283v1)的角色/工具分工；未据此新建或验证额外感染因子。
