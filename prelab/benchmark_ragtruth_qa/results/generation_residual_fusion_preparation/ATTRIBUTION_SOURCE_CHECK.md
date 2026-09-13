AT2 确实提供我们尚未保留的“回答位置 × 具体来源”依赖信号；但依赖不等于事实支持。仅查主来源，未运行模型或改变实验。

| 工作 | 本次核验的发表状态 | 官方资源 |
| --- | --- | --- |
| **Learning to Attribute with Attention（AT2）**，Benjamin Cohen-Wang、Yung-Sung Chuang、Aleksander Mądry | 2025 arXiv 预印本。作者主页与代码引文仍写 preprint；检索到 ICLR2026 under-review 稿，但本次未取得可验证录用信息，不能称已发表顶会。OpenReview 正文验证页面／API403使最终决定无法直接核实。 | [作者主页](https://bencw99.github.io/)、[论文](https://arxiv.org/abs/2504.13752)、[MadryLab/AT2](https://github.com/MadryLab/AT2)、[投稿页面](https://openreview.net/forum?id=KFSv2egats) |
| **Model Internals-based Answer Attribution for Trustworthy Retrieval-Augmented Generation（MIRAGE）**，Jirui Qi、Gabriele Sarti、Raquel Fernández、Arianna Bisazza | **EMNLP2024 主会**，6037–6053页 | [正式会议页](https://aclanthology.org/2024.emnlp-main.347/)、[Betswish/MIRAGE](https://github.com/Betswish/MIRAGE) |

AT2 的单个来源分数为各注意力头的加权和：`score(s,Y)=Σ(layer,head) θ × mean(目标token) sum(来源token) Attention`。训练先随机屏蔽来源，计算原输出概率的变化，再学习共享头系数；论文用 2,000 样例、每例 32 次屏蔽，目标是预测屏蔽效应的负 Pearson 相关损失。因此**训练标签是扰动效应，不是人标幻觉**。这部分训练很贵，不能把廉价推理误说成全流程免费。[论文§3–4](https://arxiv.org/html/2504.13752v1#S3)

对新回答，**一次完整白盒前向得到各层状态，再额外计算目标 query 的 QK 注意力和线性分数即可**；如生成时已保存所需状态，可直接复用。官方 `AttributionTask.get_hidden_states` 做一次 `output_hidden_states=True`；`get_layer_attention_weights` 从各层状态重算局部注意力，取的是目标位置**前一个位置**的 query（`attribution_start - 1`），对应预测该词元。实现支持 Llama／Qwen2 等，不需要逐来源再跑模型或反向传播。它仍有重算 QK 与保存各层状态的成本，不是“零额外计算”。[官方状态回放](https://github.com/MadryLab/AT2/blob/main/at2/tasks/base.py#L114)、[注意力实现](https://github.com/MadryLab/AT2/blob/main/at2/attribution/attention.py#L60)

官方输出先保留 `[来源数, 目标词元数]`，再按所选目标范围取均值；因此可以输出逐 passage，亦可细到来源句／词。头系数是单层线性读出，可选最终 L1 归一；不是各头直接平均。[官方聚合](https://github.com/MadryLab/AT2/blob/main/at2/attribution/attributors.py#L173)、[线性读出](https://github.com/MadryLab/AT2/blob/main/at2/attribution/score_estimators.py#L75)

MIRAGE 是另一条已正式发表的路线：先比较有／无资料的下词分布，用 KL 找依赖上下文的回答词元；再对实际词元与无资料替代词元的概率差做输入 embedding 梯度，汇总到资料编号。官方代码明确使用 `saliency`、`contrast_prob_diff` 和去除 context 的对照，**不是一次纯前向**。它能做细粒度来源归因，成本包括额外无资料前向及选中词元反向；不需要另一个裁判模型。[论文方法](https://arxiv.org/html/2406.13663v2#S3)、[官方调用](https://github.com/Betswish/MIRAGE/blob/main/mirage.py#L130)

对当前实验的直接含义（研究推断）：已有 LB1024 把各资料汇成总量，无法从缓存逆推出“用了 passage1 还是 passage3”；必须在新提取时保留来源轴。可把“自称引用来源 A、内部归因却集中 B”作为风险特征，再由原人标训练／评测。它不能单独证明错引：可能 A/B 都支持事实，也可能模型依赖 B 却误读内容；而“资料不足”这种陈述也可能高度依赖输入资料。AT2 的源码支持 Llama 不等于现成已训练系数适配我们的 Llama2-NF4；必须明确目标模型、量化、时刻与训练迁移条件。当前融合入口和 v2 基线均保持原方案。
