# 本轮算法依据与实现边界

- [PRISM：Prompt-Guided Internal States，ACL 2025 主会](https://aclanthology.org/2025.acl-long.1058/)：通过提示让内部状态更关注真伪，在不同领域迁移检测器。本轮“加核查提示但不生成”的对照借鉴这一思路；提示、证据任务、特征及训练方案不同，不称原论文完整复现。其存在说明“核查提示改变内部表示”本身不能作为我们的原创点。
- [Simple Factuality Probes，Findings of EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.880/)：长文本事实性可以用轻量内部状态探针检测。本轮固定数据比较线性与小网络，检验增加容量是否有实际收益；非原始实验复现。
- [SelfCheckGPT，EMNLP 2023 主会](https://aclanthology.org/2023.emnlp-main.557/)：多次采样的回答一致性提供检测信号。本轮控制追加生成预算，比较普通扩写与证据核查；并未实现原文所有相似度／NLI 方法。
- [INSIDE，ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/0d1986a61e30e5fa408c81216a616e20-Abstract-Conference.html)：在多次回答的内部状态空间衡量一致性。我们考察追加内容的状态是否带来增量，但并未实现其 EigenScore 或特征裁剪，不能列作已完成的原版基线。
- [FactSelfCheck，Findings of EACL 2026](https://aclanthology.org/2026.findings-eacl.296/)：事实粒度的采样式检测已有研究。目标片段更细本身也不构成创新证明。

本轮待检验的问题是：在同一个 Qwen、相同新闻标签及追加生成预算下，风险引导的证据核查能否优于随机核查／普通扩写；补充回答的内部状态是否比只加提示、只读模型口头结论更有用。所有结果先按探索性实验解释。
