# 公共 QA 的 LUMINA 补齐盘点

**结论：R25 已实现完整的两项信号公式，但没有在公共 QA3839 上提取或计分。缺口是特征前向与本地计分，原 LUMINA 不需要训练分类器。** 本次仅查既有代码、缓存字段和输入统计；未读封存测试或新 GHOST 分数，未建立实验队列。

**已做。** [固定作者源码](https://github.com/deeplearning-wisc/LUMINA/blob/c43ff41d872b05f659dcb3ad3a6dd78226954319/lumina.py)对应本地 `round7_evidence_grounding/src/lumina7.py`：①原资料与替换资料下的 top100、未重新归一化概率、输入嵌入余弦核平方 MMD；代数化简保留概率质量差项；②逐层 IPR，保留深度权重、熵分母、最终最可能词元的概率比及实际回答词元概率校正；③固定风险分数 `0.5×IPR−0.5×MMD`。读取位置均为当前词元之前，完整回答前向不让当前位置看到未来。IPR 按作者代码包含全部输出层及末层再次 norm；论文 Eq.8 写至 L−1，这个差别已披露。

R25 的602答、14,968词元确已完成；其新增联合 MMD 替换两段正文的等长 token，保留标题。它是 Qwen/R16 结果，不能复用为 Llama/公共 QA 特征，也不等于作者自然随机资料干预的原样复现。

**公共 QA 可复用与未做。** 原793的 `data/features`、新增3046的 `fit_expansion/llama_features_v3/features` 都完整；可复用本地 Llama2权重、冻结输入/字符坐标、窗口几何和原对照。已核 schema 及各一份 NPZ 字段：只有 LB、NLL、末层状态等，没有 IPR、全层概率、随机资料 MMD 或 top100缓存。`hidden_last` 读在当前词元之后；向前错一格仅能覆盖后续词元的最终状态，既缺首词元状态，也缺其余31层。NLL、HARP投影、GHOST四个摘要都不能还原 IPR。MiniCheck/ModernBERT 的状态属于另一模型，不能替用。可复用 R7 数学函数，但须另接 QA 冻结输入；R25入口绑定 Qwen/R19计划，不能直接搬跑。补缺项可免重建 LB/NLL/HARP，仍需原资料和替换资料两次骨干前向。

**最小可执行成本（3839全覆盖口径）。** 原输入合计2,408,466词元、回答708,506词元，最长输入1232。需3839次原前向＋3839次替换前向＝**7678次**；若沿等长正文替换则输入合计**4,816,932词元**。IPR另需32层×708,506＝**22,672,192个层词元词表投影**，不是“两次普通前向”就结束。若改为完整随机资料，精确词元数须先构造新计划，不能沿用等长总数。公式不需新拟合；本地 F1 仍需预先固定聚合、仅在开发 cal 选阈值。只落 IPR/MMD/合成三标量约8.1 MiB；若另存两路 top100 float32概率/int32编号约1.06 GiB。沿既有逐层/16词元分块可避免常驻全词表张量；8GB上的最长样例与耗时尚未实测，R25的395秒不能直接外推。

**替换资料可以不看金标。** 本地论文附录D用另一数据点的资料作替换；可只用已开放 fit 材料池按固定种子、材料组与哈希选不同资料，保持问题及原回答 token 不变；同材料的多个生成器回答共享替换。不得用错误范围或检测分数挑 donor。不同材料身份只能排除已知重复，不能证明语义完全无关。整段随机资料更接近论文；等长正文补丁则更易保持位置，但必须继续称本地干预适配。

此外，原作者返回逐词元分数、论文Eq.2整答取词元均值，与本地“4原始BPE均值→整答最大值”不同；NF4/BF16、冻结聊天模板、不沿作者额外空格/12,000字符截断，以及3046份其它生成器回答的统一 Llama 重放，都须披露。可称“官方公式的公共 QA 适配”，不能称原论文全复现或原生成器轨迹。计数与具体可复用路径见 [inventory.json](D:/Projects/Multi_Agent_Graph_Analysis/prelab/benchmark_ragtruth_qa/results/lumina_completion_inventory_v1/inventory.json)。

论文依据：[已存PDF](D:/Projects/Multi_Agent_Graph_Analysis/doc/ref_paper/intelligence_knowledge_boundary/hallucination_detection/LUMINA_ICLR2026.pdf)，Eq.2、Eq.8、附录D。
