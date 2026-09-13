# Round 7：Lookback Lens 与 ReDeEP-token 的提取记录

实现：`src/attention7.py`；CPU检查：`tests/test_attention7.py`。此文件描述已实现信号，不代表已完成7B运行或正式评测。

## 接口与数组

`extract_attention(tokenizer, frozen_model, row, generated)` 返回 `(values, metadata)`。

- `row` 只读 `row_id/system/prompt/passages[].text`，不读标准答案、资料缺失标记、分区或标签。
- `generated` 读取模型自己生成并冻结的精确输入和回答 token IDs、文本、自动解析的回答项范围。重新分词的文本不能代替精确生成 IDs。
- `item_ids[N]` 为可解析回答项；`lookback_features[N,L*H]`、`redeep_ecs[N,L*H]`、`redeep_pks[N,L]` 为未经标签拟合的特征。
- `token_lookback[R,L*H]`、`token_redeep_ecs[R,L*H]`、`token_redeep_pks[R,L]` 和 `response_token_offsets[R,2]` 保留逐 token 信号。它们不是逐词风险概率。
- 头维度按层优先、头其次排列；层号1起、头号0起。Qwen2.5-7B 的 `L=28,H=28`，因此头特征维数784。
- 每项对与该项文本范围重叠的 token 特征取算术平均。不使用后续回答项，不先选可疑句。解析失败项由调用方统计覆盖率，不能悄悄删掉评测标签。

## Lookback Lens 的忠实部分与适配

对第 `j` 个回答 token、每层每头：

`context_mean = sum(attention to source tokens) / number of source tokens`

`response_mean = sum(attention to generated tokens 0..j) / (j+1)`

`lookback = context_mean / (context_mean + response_mean)`

该公式保留官方 `step01_extract_attns.py` 的**分别取均值后相除**。把它改为两侧注意力总质量的比例，会改变长度校正，不能混用。官方 `step03_lookback_lens.py` 同样把范围内各头比例取平均后作为分类器输入。

明确适配：

1. 当前实验监测“这项刚写完时”的状态，query使用当前 token 已被读入的位置 `P+j`。官方生成代码在下一 token 预测时提取注意力，时点不同。我们不把该适配称为原版在线预测复现。
2. context只包括实际提示的 `Search results` 正文，含来源标题、编号与分隔符；不含问题、系统提示和助手模板。官方模板的context边界会包含部分任务文本。因此不能原样使用其Llama分类器权重。
3. 用Qwen本次自己生成的回答，采用统一回答项范围，重新在训练集拟合分类器，验证集选参数和阈值。

## ReDeEP-token 的忠实部分与适配

ECS：每层每头只在来源 token 内选注意力最高的10%，将这些 token **最终层状态**平均，再与当前回答 token 的最终层状态取余弦。实际使用 `max(1,floor(0.1*C))` 个来源 token；最少1个只处理极短测试材料。用选择矩阵乘法完成平均，避免巨大隐状态gather张量。

PKS：同一层 FFN 前后的**残差状态**分别经过模型最终 RMSNorm、词汇输出头和softmax，计算：

`JSD(P,Q)=0.5*KL(P||M)+0.5*KL(Q||M), M=(P+Q)/2`。

最后一层取block后的原始残差再归一化一次，没有重复归一化。每层均使用同一个最终RMSNorm与输出头，遵循官方自定义Llama的LogitLens方式。值使用自然对数，范围 `[0,ln(2)]`，没有原脚本的任意倍率。

原代码 `F.kl_div(logP,M)` 的方向是 `KL(M||P)`，不是论文标准JSD；官方代码也链接了Issue #2。当前实现明确修正为论文公式，不声称精确复制旧代码数值。结果采用“ReDeEP-token（Qwen适配，标准JSD）”名称。

输出全部层与头，提取端不依据标签挑Copying Heads。统一评测器可用训练集相关性排名，再在验证集选择保留数量及ECS/PKS组合权重。论文附录J在验证集排名及选参数，原网格top-K为1..32，alpha固定1，beta在(0,2)步0.1。本轮建议将排名及MinMax拟合放在训练集，验证集选：

- `K_heads={1,2,4,8,16,32}`；`K_layers={1,2,4,8,14,28}`。
- `beta={0.1,0.2,0.4,0.6,1.0,1.2,1.6,1.9}`，alpha=1。
- 风险连续值=`MinMax_train(sum selected PKS) - beta*MinMax_train(sum selected ECS)`。

这288种参数组合是缩小的适配网格，最终以统一评测器的冻结配置为准。不能使用Llama的头号、阈值，不能用测试集重新排序、标准化或定阈值。原论文主表最终是回答级聚合；本轮改变为回答项平均，需要单独报告。

## 8GB显存方案

1. 第一遍SDPA因果重放获得最终层全部 token 状态。
2. 第二遍保持SDPA，并hook实际q/k投影输出和RoPE位置；只对回答query每8个一批重算注意力行，正确展开Qwen GQA的KV头。mask严格禁止未来位置。
3. 每层立即从这些行计算Lookback与ECS，计算完即释放。不会保存所有层的完整 `S×S` 注意力。
4. FFN前后词汇分布每8个回答位置一批计算JSD，算完只保留标量，不保存全层全位置词汇分布。
5. 这是两遍离线重放，耗时计入检测特征提取，不称为在线零额外成本。CPU检查不保证实际NF4显存/耗时，7B烟测仍由主进程完成。

## 已完成的独立检查

`python -m unittest discover -s prelab/round7_evidence_grounding/tests -p test_attention7.py -v`

七项CPU检查通过：均值比例区别于质量比；标准JSD对称/同分布为零/上界；只选择资料top token；未来mask；真实tokenizer来源边界；随机tiny-Qwen各头各层与完整eager参照一致；改写后续回答及添加隐藏标签字段不改变前项特征。tiny模型为2层、16维，使用本地tokenizer，未加载7B权重、未用GPU。

## 原始来源

- Lookback Lens，EMNLP 2024：https://aclanthology.org/2024.emnlp-main.84/
- 官方Lookback代码：https://github.com/voidism/Lookback-Lens ，本地已有 `prelab/references/Lookback-Lens`。
- ReDeEP，ICLR 2025（公式3/5、§4.1、附录J）：https://proceedings.iclr.cc/paper_files/paper/2025/file/7daf60e805e596c3bd1e843e72ea5560-Paper-Conference.pdf
- 官方ReDeEP代码：https://github.com/Jeryi-Sun/ReDEeP-ICLR
- 原实现JSD问题：https://github.com/Jeryi-Sun/ReDEeP-ICLR/issues/2

官方回归示例还有筛选test行/截列等需要修复的工程问题。本轮不执行这些评测脚本，统一评测器严格按已冻结分区训练与测试；这不构成对论文实验流程的推断。
