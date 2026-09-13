# MVA 来源与接入核查

**可作为新的事后定位基线，但不能直接复用现有LB/NLL/HARP缓存，也不能据论文保证QA收益。** 没有下载模型、实现提取器、使用GPU或读取本项目封存测试。

核查版本：作者仓库 `master` commit **`f8b871a06b6c18dabe5881bd02a68854b2940b81`**。下列实现公式依据固定版源码，避免把论文与实现混写。[gen_features.py](https://github.com/Ogamon958/mva_hal_det/blob/f8b871a06b6c18dabe5881bd02a68854b2940b81/features/gen_features.py)

设单层单头注意力为 $A_{ij}$，行是query、列是key，完整输入共 $N$ 个词元，包括提示、资料、回答；位置从1计。忽略代码的epsilon及FP16舍入，三类原始特征为：

| 特征 | 官方代码实际计算 | 时点 |
|---|---|---|
| `key_avg` | $\frac1N\sum_{i=j}^{N} iA_{ij}$ | 当前词元读入后，汇总自己及后续query对它的注意力；使用未来回答 |
| `key_entropy` | $p_{ij}=A_{ij}/\sum_r A_{rj}$，再算列熵 $-\sum_i p_{ij}\log p_{ij}/\log m_j$，$m_j$为该列非零项数 | 使用未来回答 |
| `query_entropy` | 当前行归一后算 $-\sum_j p_{ij}\log p_{ij}/\log m_i$ | 当前词元**读入后**的query行，包含自身及此前提示/资料/回答；不是预测当前词元的 $i-1$ 行 |

非零项只有一个时熵置0；官方先作行/列归一，再调用熵函数归一，存FP16，NaN置0。实际实现范围是完整输入矩阵，之后按回答位置取特征。每类保留全部层和头，Llama2-7B的32层×32头对应 **3×1024=3072维/词元**。[源码532–672行](https://github.com/Ogamon958/mva_hal_det/blob/f8b871a06b6c18dabe5881bd02a68854b2940b81/features/gen_features.py#L532-L672)

需要明确区分两套定义：论文式(2)的入向均值分母是有效后续数 $T-j+1$，代码却除完整 $N$；论文式(4)–(5)使用校正矩阵 $A'_{ij}=iA_{ij}$，式(5)还写成行归一，而代码入向熵实际使用**原A的列归一**。因此未来实现必须声明“复现作者代码”还是“按论文公式适配”，不能悄悄互换。[论文§3.2，PDF第3–4页](https://aclanthology.org/2025.starsem-1.31.pdf#page=3)

未来信息还有第二条路径：官方检测头是Transformer+CRF；`optimize.py`使用padding mask，未给因果mask。这属于完整回答可见后的序列标注。即便只取outgoing特征，照搬该检测头也不能称逐时刻在线预测。[检测头代码](https://github.com/Ogamon958/mva_hal_det/blob/f8b871a06b6c18dabe5881bd02a68854b2940b81/features/optimize.py#L657-L767)

论文QA是**检测器自身分词下的token micro-F1**，不是本项目4BPE窗口F1。Llama3-8B提取时raw **.563**、norm **.556**，其微调对照 **.597**；Qwen2.5-7B为 **.506/.487**。QA显式冲突的raw召回仅 **4.0%**。原论文Lookback使用固定0.5阈值，不能拿其较弱数字代替我们校准过的强对照。[论文表3、7、8及§4.2–4.4](https://aclanthology.org/2025.starsem-1.31.pdf#page=6)

本地接入的最小边界：

- `src/feature_qa.py`已能从Llama2各层query/key重建因果注意力，并保留原完整聊天模板与`answer_token_positions`。数学接口相容，但需要新增前向统计；现有LB比例、词表NLL和HARP状态均不能还原上述列分布。
- 先考虑raw三特征。官方 `get_features_no_generate` 同时堆叠全层注意力、norm中间量并调用多GPU路径；不能按原脚本直接承诺8GB可跑。沿现有逐层/分块接口累计列和、列熵及行熵在工程上可行，尚未实测资源。
- 不能复用其聊天边界解析：源码写死Llama3头标记，且`a or b`只返回第一个非空字符串；Llama2应使用本项目已验证的精确坐标。也不能无审查照抄norm变体：源码norm广播形状`[L,H,N,1]`乘在query轴，与论文右乘key范数的写法不同。[边界解析](https://github.com/Ogamon958/mva_hal_det/blob/f8b871a06b6c18dabe5881bd02a68854b2940b81/features/gen_features.py#L408-L476)、[norm广播](https://github.com/Ogamon958/mva_hal_det/blob/f8b871a06b6c18dabe5881bd02a68854b2940b81/features/gen_features.py#L147-L178)

本次只完成来源与接口判断。它是与LB不同的回答内部注意力信号，值得保留为候选；但未来方向、代码公式差异和较弱的QA冲突召回决定了它不是当前引用语义核查的直接替代品。
