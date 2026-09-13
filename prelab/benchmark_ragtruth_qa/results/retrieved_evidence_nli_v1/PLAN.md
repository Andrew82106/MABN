# Retrieved-evidence NLI v1（我们的方法候选）

每条原回答陈述在 passage 1/2/3 内分别用固定 BM25 选最多两句。每个选中证据句单独与陈述送入冻结 ModernBERT NLI，保留 E/N/C。证据编号、坐标、原文和哈希全保留。

prepare/check 只做 CPU、无标签预处理；审核后才可单独运行 gpu-smoke 和 extract。score 再另起 CPU 进程读取开发标签，以 source group 五折交叉预测训练固定逻辑回归，将 claim 风险映射回原 4-BPE 窗口。正式 baseline 不改，本目录只属于新候选。
