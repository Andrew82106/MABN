# 本轮补充的方法证据

## 跨任务高分不等于事实判断

[EMNLP 2025 Findings 原论文](https://aclanthology.org/2025.findings-emnlp.952.pdf)发现：RAGTruth的Data2txt更常含幻觉，识别任务类别的朴素规则就能解释部分总体高分；其测试的探针跨数据迁移明显变差。这个结论有具体方法和数据范围，不能推出一切白盒方法都无效。我们的QA单任务评测和原场景迁移应继续分开；新增辅助任务只用于训练，不能混进QA评测提高总体数值。长度/位置朴素对照用于本地检查，论文没有证明我们的长度信号必然有问题。

## 多种注意力特征

[*SEM 2025 原论文](https://aclanthology.org/2025.starsem-1.31.pdf)将收到的平均注意力、收到的注意力分散程度、发出的注意力分散程度送入Transformer和CRF做序列标注。这与仅用Lookback比例不同；收到后续词元注意力需要完整回答，属于离线信号。论文主要优势在长上下文的摘要与Data2txt，不能直接推断可改善我们的QA。正式场次是*SEM，不应写成ACL/EMNLP主会。[作者代码](https://github.com/Ogamon958/mva_hal_det)

## RAGLens与定位的区别

[RAGLens作者论文v1](https://zhenghao-he.github.io/assets/pdf/iclr2026.pdf)用SAE特征的全回答最大值，按训练标签筛特征，再用加性模型判断整答。它把重要特征触发处当局部解释；这不自动等于训练、校准过的逐词元风险概率。若采用，必须另按原范围标签检验定位。[作者仓库](https://github.com/gzxiong/RAGLens)已标ICLR2026，示例采用Llama3.2-1B及对应SAE；本轮只核查和保存来源，没有下载权重或运行RAGLens。作者PDF实际仍是2025-12-09 arXiv v1，不能冒称已核对最终定稿全部细节。

以上三份PDF已保存原参考目录，并记录URL和SHA；既有RLSeek ACL2026参考文件保留。当前GPU继续固定人标辅助训练，CPU继续扩充基线；本次来源核查不改变现有数据、标签、阈值或任务预算。
