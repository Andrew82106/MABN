本次只核对原文与官方源码，不修改 v2 协议、不读取我们的封存测试数据。核查版本：2025 年论文 v1；官方仓库论文发布前最近提交 `ef4bdd43fd32cb2a0548caded77ed028e1afe1ca`（2025-02-23）。仓库目前已升级到其他模型，不能把当前默认代码直接当成原论文实现。

1. **完整输入是一个回答对应的完整资料／问题／回答。** 不是把同题六个生成器的回答拼在一起，也不是逐句或逐资料块单独判。历史预处理直接保留官方 `source.prompt` 与单个 `response.response`；Dataset 用 tokenizer 的文本对输入 `prompt, answer`，最长 4096、允许截断。当前 v2 则明确拼为“原资料 + SEP + 问题 + SEP + 原回答”，不保留完整生成提示中的全部指令；本轮最长 979，不截断。这是实质提示模板差别。[官方预处理](https://github.com/KRLabsOrg/LettuceDetect/blob/ef4bdd43fd32cb2a0548caded77ed028e1afe1ca/lettucedetect/preprocess/preprocess_ragtruth.py#L63)、[官方 Dataset](https://github.com/KRLabsOrg/LettuceDetect/blob/ef4bdd43fd32cb2a0548caded77ed028e1afe1ca/lettucedetect/datasets/ragtruth.py#L27)

2. **损失主要只算回答 token。** 问题和资料标签设为 -100；回答 token 与任何人标错误范围相交即为 1。历史代码把回答后的最终分隔 token 也默认标为 0，且没有我们这种 lexical-only 排除标点的规则。Trainer 直接使用 Hugging Face `outputs.loss`，没有自定义类别权重、来源组权重或原/新增回答半权重。当前 v2 则在原 Llama BPE 上计算带类别／来源组权重的 BCE，并排除非 lexical 词元；二分类 CE 与 logit 差 BCE 本身等价，主要差别是**坐标、掩码及权重**。[Dataset 标签](https://github.com/KRLabsOrg/LettuceDetect/blob/ef4bdd43fd32cb2a0548caded77ed028e1afe1ca/lettucedetect/datasets/ragtruth.py#L80)、[Trainer 损失](https://github.com/KRLabsOrg/LettuceDetect/blob/ef4bdd43fd32cb2a0548caded77ed028e1afe1ca/lettucedetect/models/trainer.py#L66)

3. **原论文：AdamW，学习率 1e-5，weight decay 0.01，6 轮，A100，batch 8，动态 padding。** 我们保留学习率／衰减／六轮；受显存限制采用单答前向、累积 8 答、BF16、梯度裁剪。我们只训练 QA3680；论文在 RAGTruth 的 QA、摘要、数据转文本三类任务上训练。不能称为逐项复现。[论文 §3–4](https://arxiv.org/html/2502.17125v1#S4)

4. **选模型的口径不同。** 原论文按 token F1 选检查点，0.5 概率划定风险 token；历史训练入口把 `split == test` 传入 Trainer，而 Trainer 每轮据其 F1 存最佳模型。这证明该历史公开入口存在按 test 选轮次的路径，**不能仅凭代码断言发表结果一定按此运行**。当前 main 已改为从 train 随机抽 10% dev。我们仍坚持组隔离 cal159、同时照顾整答与窗口、test 封存；不照搬历史 test 选型。[历史入口](https://github.com/KRLabsOrg/LettuceDetect/blob/ef4bdd43fd32cb2a0548caded77ed028e1afe1ca/scripts/train.py#L50)、[历史保存规则](https://github.com/KRLabsOrg/LettuceDetect/blob/ef4bdd43fd32cb2a0548caded77ed028e1afe1ca/lettucedetect/models/trainer.py#L95)、[当前入口](https://github.com/KRLabsOrg/LettuceDetect/blob/main/scripts/train.py#L70)

5. **论文的 span F1 实际按字符重叠计数。** 官方评测把预测范围字符长度作为预测量，与人标范围相交的字符长度作为命中量。我们的定位指标则以重叠的“四个原 Llama BPE”窗口为判定单位，窗口内取 lexical 风险最大值；两者分母、边界容忍、重复覆盖次数均不同，不能直接比较高低。论文 v1 的 QA base 数字是整答 F1 65.52%、字符 F1 61.50%，并非“四词元 F1 61.50%”。[论文表2–3与§5](https://arxiv.org/html/2502.17125v1#S5)、[字符评测实现](https://github.com/KRLabsOrg/LettuceDetect/blob/ef4bdd43fd32cb2a0548caded77ed028e1afe1ca/lettucedetect/models/evaluator.py#L171)

因此当前 v2 应称“LettuceDetect 启发的完整输入检测基线适配”。完整输入、全层微调和六轮预算有主来源依据；我们自己的数据范围、组权重、坐标映射、校准规则、输入模板及精度必须明确列出。此核查不改变已冻结实验。
