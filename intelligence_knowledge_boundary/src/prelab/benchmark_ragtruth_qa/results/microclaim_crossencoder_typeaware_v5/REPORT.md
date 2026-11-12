# Expanded-v5 类型感知续训：CPU 准备报告

v4 的 binary BCE 只要求 neutral/contradiction 任一变高，因此不会教模型区分“无依据”和“证据冲突”。v5 保留原风险分数，同时把 NLI 三类头重新对齐到安全、无依据、冲突。

## Fit 标签

| 互斥训练类 | 数量 | 三类软标签 E/N/C |
|---|---:|---|
| 安全 | 31445 | 1 / 0 / 0 |
| 仅无依据（EBI/SBI） | 3052 | 0 / 1 / 0 |
| 仅冲突（EC/SC） | 405 | 0 / 0 / 1 |
| 同时无依据与冲突 | 17 | 0 / 0.5 / 0.5 |

多类型处理已经固定：同属无依据或同属冲突时合并为一个类；跨两类的17条使用各半软标签。5条只有字符跨度相交、但没有风险词元的样本按原 binary gold 保持安全。

原始正例类型覆盖：EBI 2153、SBI 920、EC 380、SC 42。五个 held fold 和五个训练补集都含四种互斥类，且来源组零交叉。

## 冻结损失

`BCE(logsumexp(N,C)-E, 是否错误) + 0.25 × 三分类软标签CE`。仍用 v4 的逐样本权重；三分类部分在每个训练折内按加权标签质量做逆频率平衡，使 E/N/C 三类的总监督质量相等。最终风险分数仍是 v4 的 `sigmoid(logsumexp(N,C)-E)`。

## 唯一 fold-0 GPU pilot

从 v4 fold-0 checkpoint 继续训练1轮；新 AdamW，学习率 3e-06，weight decay 0.01，无 scheduler，seed 20261023。训练 27936 条、验证 6983 条，预计 8.2 分钟，保守 10.4 分钟，显存约 3.21 GB。只允许这一套参数，不做网格或校准集选择。

## CPU 验证

12条真实 fit 输入覆盖四类标签，tiny ModernBERT 完成一次联合损失更新；encoder 与 classifier 都发生更新（18 个参数张量），风险公式和软标签加权CE均逐值核对通过。

本阶段读取 calibration 行数为0，没有访问 calibration/test 标签，没有使用 GPU，也没有修改 baseline。
