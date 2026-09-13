已下载并核验通用 ModernBERT-base，完成全部开发输入、映射及 CPU 检查。尚未运行 GPU 或训练。

本对照参考 [LettuceDetect 预印本](https://arxiv.org/html/2502.17125v1) 的完整资料—问题—回答词元分类方式。使用 [ModernBERT-base 官方通用权重](https://huggingface.co/answerdotai/ModernBERT-base)，版本固定为 8949b909ec900327062f0ebf497f51aef5e6f0c8；不加载已经在 RAGTruth 上微调的 LettuceDetect 成品，避免与本项目从官方 train 划出的校准来源重叠。通用预训练语料的完全独立性没有得到证明。

每条输入保留完整资料、问题、原回答，使用分隔符区分三个部分。3839 条开发输入共1857538个编码词元，最长979，不截断、不选句、不丢资料块。问题作为独立输入段，不被补写为来源事实。

模型为约149M参数的 ModernBERT-base 加原生双类别词元分类头，全部参数训练。每个编码词元的风险 logit 为“风险类logit−支持类logit”，再按非空白字符重叠均值映回原 Llama BPE。只对原回答的可评词元计算原人工风险标签损失；资料与问题没有目标标签。纯空白词元的损失为0。该映射保留原708506个BPE，评测仍为4个原始BPE滑窗和整答最大风险。

使用与其他扩充模型完全相同的3680训练回答、615来源组，以及159校准回答、154独立来源组。训练来源内原634答与辅助3046答各占一半基础权重，再做既有类别平衡和组等权。6轮、AdamW学习率1e-5、weight_decay .01、累积8答、每次实际输入1答、梯度裁剪1；float32，关闭TF32和AMP，启用非重入梯度检查点，SDPA，关闭编译。每轮保存完整fit/cal预测和指标，只有第1至6轮可入选；第0轮只是诊断。

模型轮次和两个阈值仅在原校准集选择，官方测试保持封存。保留所有轮次的参数、优化器及随机状态，不覆盖失败或自动改训练预算。真实GPU显存和耗时须由后续检查测量，目前没有凭CPU结果断言8GB显卡一定可运行。

这是额外语义核查基线，不是原生成模型的白盒探针。它与MiniCheck比较时同时改变了编码器、完整输入和微调层数，不能把差异单独归因于问题段或上下文长度。它也并非LettuceDetect完整复现：本轮仅QA数据，使用本项目的分组/权重、原BPE监督映射和校准选择规则。论文给出的跨任务/字符片段数值不直接与本地窗口F1排序。

项目根目录命令：

```powershell
prelab/.venv/Scripts/python.exe -X utf8 prelab/benchmark_ragtruth_qa/src/run_full_context_encoder.py prepare
prelab/.venv/Scripts/python.exe -X utf8 prelab/benchmark_ragtruth_qa/src/run_full_context_encoder.py cpu-test
# 前两项已完成。以下仅在根代理安排独占GPU后执行：
prelab/.venv/Scripts/python.exe -X utf8 prelab/benchmark_ragtruth_qa/src/run_full_context_encoder.py gpu-smoke
prelab/.venv/Scripts/python.exe -X utf8 prelab/benchmark_ragtruth_qa/src/run_full_context_encoder.py train
```

GPU检查只用最长输入验证重复前向、人工零目标反向、梯度和显存；不使用真实风险标签来选模型或调参数。若失败保留报告并停止。`protocol.json`、`preparation_complete.json`与`source_snapshot.json`记录实际代码、数据、模型文件哈希。
