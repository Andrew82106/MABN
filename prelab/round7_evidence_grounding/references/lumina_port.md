# LUMINA：Round 7 官方代码移植说明

## 版本与范围

- 官方仓库：https://github.com/deeplearning-wisc/LUMINA
- 固定 commit：`c43ff41d872b05f659dcb3ad3a6dd78226954319`。
- 原始 `lumina.py` SHA-256：`1b5ecd8c08982b40dae59f299e580219074ee83673768feb53cf15612b7d8a5f`。
- 原文副本为 `lumina_official.py`，保留 MIT 许可于 `lumina_LICENSE`；来源记录见 `lumina_source.json`。
- 本轮实现为 `../src/lumina7.py`，版本 `official-c43ff41-all-output-layers-top100-unnormalized-cosine-ipr-v1`。

本实现复用官方检测公式，适配本轮 Qwen 的实际生成 token、回答项单位及 8 GB 显存条件。它不是原论文所有数据集、模型、基线和成绩的完整复现。没有训练新权重，也不调用外部模型。

## 保留的计算

同一个实际回答的 token 序列，分别接在原问题和原资料、原问题和独立无关资料后；两次因果前向。对于实际回答 token t，取位置 `prefix_length + t - 1` 的状态，计算产生这个 token 时的预测；不能误用读入 token t 之后的下一个 token 预测。

外部信号保留官方 top-100 近似、未重新归一化的预测概率、输入 embedding、`(1 + cosine) / 2` 核及平方 MMD。内部 IPR 保留各层深度权重、熵倒数归一化、对最终 top-1 token 概率比的上限截断，以及实际 token 概率与最终 top-1 概率的校正。风险分数为 `0.5 * IPR - 0.5 * MMD`；它是排序分数，不是经过校准的幻觉概率。

每个实际回答 token 均计算三种分数。每项按既定字符范围所覆盖的 token 取算术均值。解析失败项仍留在 `item_ids`，分数为 NaN 并保留原因，不能默默删掉或记成低风险。标注与条件字段不参与计算。

## 原论文与官方代码的层差异

论文 Eq. 8 的层求和写为 `1..L-1`。固定官方代码使用 `outputs.hidden_states[1:]`，因此包括所有 L 个输出层。Hugging Face 的最后一个输出 hidden 已经过最终 norm，官方代码又对每个 hidden 调用最终 norm，因此最后层被再次归一化。

本移植按固定官方代码执行：第 1 到 L-1 个 decoder block 输出，加经过最终 norm 的第 L 层；所有这些状态再经过 norm 和 lm_head。元数据明确保存这一约定。没有偷偷删除最后一层或替换成论文公式。CPU 测试直接与官方私有 IPR 函数比较。

## 对接与内存变化

1. 原始输入重新渲染聊天模板后，必须与 `generated.input_token_ids` 完全一致。回答保持原始 `response_token_ids`；两种资料下使用相同回答 token，既不重新生成也不重新分词。
2. 只替换原用户提示 `Search results` 后的编号资料段落，问题、系统提示及措辞原样保留。拒绝混入另一题的问题。随机资料须来自调用方冻结的独立干扰池；代码检查问题标识、标题、段落文本不重复及段落数一致。语义上确实无关、没有泄漏答案，仍须由数据审查确认。
3. 没有沿用官方 `predict` 中的额外回答前空格、重建用户消息或 12,000 字符截断，这些会改变我们的实际生成序列。
4. 两次前向只调用 decoder backbone，不创建整段输入的词表 logits。仅把回答位置的各层 hidden 转到 CPU；每次最多处理 16 个回答位置的 lm_head，逐层累加 IPR，不在 GPU 堆叠 `层 × 回答 token × 词表` 概率。
5. MMD 使用余弦核的精确代数化简：`0.5 * ((sum(p)-sum(q))^2 + ||sum(p*normalize(Ep))-sum(q*normalize(Eq))||^2)`。这与官方的三个 k×k 核矩阵加权求和数学等价，并保留 top-k 概率质量不同产生的第一项；没有增加近似。原 top-100 截断是仍然保留的官方近似。
6. 模型激活及 norm/lm_head 使用模型当前 dtype；输出 logits、概率、embedding 核运算和累加使用 float32。官方代码未固定这些运算的统一 dtype，半精度 embedding 与 float32 top-k 权重可能不兼容。本移植明确记录 float32 数值策略，不宣称逐 bit 复现原生半精度计算。

## 接口

`extract_lumina(tok, model, row, generated, random_row, chunk_size=16)` 返回 `(values, metadata)`。

`values` 包含：`item_ids`、`lumina_mmd`、`lumina_ipr`、`lumina_score`；完整逐 token 数组为同名字段加 `token_` 前缀；另保留 `response_token_offsets`。每个 item 的分数可用时刻、实际 token 范围和不可用原因放在 metadata.items。

metadata 另含代码版本/hash、原与随机资料及 token hash、两次前向耗时、总耗时、GPU 峰值分配内存（若使用 GPU）、top-k、lambda、dtype、聚合规则、norm 约定。当前模板要求每段为 `[序号] 标题\n正文`，与 Round 6 一致；外部数据也需在进入生成前统一成此格式。

## 已验证与待验证

`../tests/test_lumina7.py` 只使用随机小 Qwen 和人工张量，在 CPU 上验证：官方核函数与代数化简、官方 IPR 与流式累加、完整输出逐 token 分数、不同 chunk 大小、将回答截断后的前项不变、原始 token 对齐、失败项保留，以及 bfloat16 激活下统计量有限。结果写入 `lumina_cpu_validation.json`。

这些 CPU 检查验证数学移植和因果位置。随后统一运行器已完成 2 条旧开发题工程样例的 7B NF4 提取：总计 2.525 秒，峰值 allocated 显存 5.402 GiB，没有 OOM；记录见 `../data/engineering/lumina/manifest.json`。该代码 hash 与 CPU 验证一致。它仅说明这两条短样例可运行，不能证明最长上下文显存、任务准确率或与原论文成绩相同；这些仍待本轮正式运行。原论文 H100 速度不能替代本机测量。
