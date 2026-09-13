# 2026-09-12 补充阅读与实验决策

本页区分文献结果、本地实验和待验证推断；不把不同数据划分或不同粒度的F1放在同一排行榜。PDF与实际SHA保存在 `doc/ref_paper/intelligence_knowledge_boundary/hallucination_detection/manifest.json`。

- [RAGTruth，ACL 2024](https://aclanthology.org/2024.acl-long.585.pdf)：原文表6按预测/人工片段的字符重叠计分；微调Llama2-13B在QA上的片段F1为0.582。它与我们固定4个原始BPE的窗口F1不同，不能直接断言本地0.60已超过论文。
- [TRIVIA+，ACL 2026](https://aclanthology.org/2026.acl-long.680/)：补充长资料、人工句级标注与受控标签噪声。适合未来长资料泛化和整答/句级检查；句标签不能直接充当人工词元边界。官方仓库现迁至 [amazon-science](https://github.com/amazon-science/hallucination-benchmark-trivialplus)。当前没有导入其具体测试回答或标签。原文A.2说明在测试集优化F1阈值，解释文献数值时须披露；本项目仍按已有方案在校准集锁定阈值。
- [RLSeek，ACL 2026](https://aclanthology.org/2026.acl-long.1492/)：核查步骤要求引用相关来源，再判断风险片段。这提示我们，追加一次泛泛的自我核查可能不够，必须让信号体现具体陈述与具体证据的关系。该方法尚未复现；不能把本地A/B局部核查称为RLSeek。
- [Temporal Multi-Signal Fusion，2026预印本](https://arxiv.org/html/2608.18115v1)：联合文本、NLI及语言模型信号，研究序列结构。其表5的BiGRU全信号AUC为0.845、词元F1为0.242，二者不能混称“约0.84的F1”；论文也报告与专门训练的片段检测器有差距。它不是已核实的顶会论文，本地时序实验亦非完整复现。
- [LettuceDetect，2025预印本](https://arxiv.org/abs/2502.17125)：0.7922是整答级检测F1；它对资料、问题、回答共同编码后做词元分类，且已用RAGTruth训练。若下载公开成品直接跑我们的官方train派生校准集，须考虑训练重叠，不能把高分当作独立泛化。因此当前优先补充训练来源另有披露的MiniCheck语义核查基线。

本地证据已经显示两点：保留完整注意力头能小幅改善定位；仅词元分数平滑也能提高基线，但提升不足以达到目标。下一步同时检查小容量网络是否缓解过拟合，以及直接比较资料与回答的信号是否补上“明确冲突”的漏报。所有参数只在fit/cal阶段确定，官方QA测试仍封存。

- [Luna，COLING 2025 Industry Track](https://aclanthology.org/2025.coling-industry.34.pdf)：注意与LUMINA区分，也不是COLING主会轨道。其第3.2节指出证据可能散落在多个资料块：推理先对每个回答词元取各资料块中的最大支持分，再汇总整答。论文训练有能对应具体块的精细标注；本项目没有这种块级支持标注，不能机械照搬。

新增本地检查：原634fit里157答有多个MiniCheck资料块，159cal里35答有多个块。按整句支持分只选一个块，不能保证这句话每个词的支持证据都在该块。因此，后续微调核查编码器若采用原始全资料标签，应保留所有块，先聚合每个词的跨块支持分，再计算原全资料标签损失；不臆造单块标签。这个多实例训练变体是待验证设计，不是Luna完整复现，也不推翻此前选块探针的实际计分，但揭示其输入信息限制。

补充资料PDF已下载，哈希记录在参考资料manifest。所有现有选块缓存和结果保持原样；未选块以补充阶段另存，避免重复计算已完成状态。

进一步核对 [LettuceDetect v1 第4节](https://arxiv.org/html/2502.17125v1)：它联合编码资料、问题和回答，只在回答词元计算分类误差；训练采用6轮、AdamW 1e-5、weight_decay .01。原文QA字符片段F1为0.6441，不能用其整答0.7922替代定位效果。基于这个输入方式，新增完整上下文小模型对照，使用通用ModernBERT-base重新训练，不加载RAGTruth已微调成品。当前仅完成下载、3839条输入和CPU检查，尚未得到训练成绩；它不属于纯生成模型白盒信号。具体改动及论文复现边界见 `results/full_context_encoder_v1/README.md`。
