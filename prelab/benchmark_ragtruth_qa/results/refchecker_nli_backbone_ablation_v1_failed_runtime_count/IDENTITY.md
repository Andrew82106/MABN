# 身份说明：仅为官方 NLIChecker 骨干替换消融

本实验只把 retrieved-evidence NLI v1 的冻结 NLI checkpoint 换成 RefChecker NLIChecker 默认的 RoBERTa-large。v1 的 51,953 个 pair、检索、claim、训练标签、LR 和 4-BPE 评测全部保持。

它没有复现 RefChecker 的 triplet/subsentence 提取、reference 分段、argmax merge_ret、多 passage/整答聚合、定位器或官方 benchmark，所以不能叫 RefChecker baseline。`is_joint=False` 的核对范围仅是：官方逐 claim×reference 展开，且 reference 在前、claim 在后。
