# CORTEX 式3584维差分：有界提取方案

状态：只读调查与实现设计，未加载GPU、未提取或拟合。范围固定为R16实际train的602份原回答，301题、278事件组。此处只做最终层差分，不含上下文残差、平滑或新回答生成。

**论文已明确、未明确的部分**

已核对 [CORTEX 正文 §2、§3.1及附录C](https://arxiv.org/html/2606.31033v1)：输入抽象为 `q || r || a` 与 `q || a`，选同一答案词元的最终transformer层输出，直接相减；不做自回归生成。论文没有给出可逐字复现的chat提示文本，没有明确额外L2归一化或最终层归一化前后位置。本次检索没有找到作者官方实现。因此下述是透明的Qwen适配，不是作者源码复现。

**固定无资料提示**

本地原模板来自 [assemble16.py](D:/Projects/Multi_Agent_Graph_Analysis/prelab/round16_dataset_expansion/src/assemble16.py:18)。最小改动是保留原系统消息、问题、任务说明和空资料栏，只删除 `Search results:` 后的全部来源标题与正文；不改成要求自评，不添加“资料不足”“请拒答”等提示。

```text
system: You are a helpful assistant.

user:
Please answer the following questions using these search results. Write one short sentence for each numbered item.

Questions:
1. {原问题，逐字保持}

Search results:
```

这是有意固定的空资料对照，保留任务说明以少改一个因素；它不同于另写一个流畅的无资料问答提示，也不宣称是论文唯一实现。标题也删掉，因为标题可能携带目标事实。用原 [model7.chat_ids](D:/Projects/Multi_Agent_Graph_Analysis/prelab/round7_evidence_grounding/src/model7.py:73) 生成聊天前缀，仍设置 `add_generation_prompt=True`，然后直接拼原始 `response_token_ids`，不对答案文本重新分词。

```python
marker = "\n\nSearch results:\n"
assert row["prompt"].count(marker) == 1
no_ref_user = row["prompt"].split(marker, 1)[0] + marker
prefix_ref = generated["input_token_ids"]
assert model7.chat_ids(tok, row["prompt"], row["system"]) == prefix_ref
prefix_no_ref = model7.chat_ids(tok, no_ref_user, row["system"])
answer_ids = generated["response_token_ids"]
ids_no_ref = prefix_no_ref + answer_ids
```

**位置、归一化与差分**

设有资料前缀长P，无资料前缀长P0，答案长T，下标i从0开始。原 `hidden_28[i]` 是最终RMSNorm后的 `last_hidden_state[P+i]`。新回放使用同一NF4模型、BF16计算与SDPA：

```python
last = model.model(input_ids=ids_no_ref_tensor,
                   use_cache=False, output_attentions=False,
                   output_hidden_states=False).last_hidden_state[0]
h0 = last[P0:P0+T].float().cpu().numpy()       # float32[T,3584]
delta = cached_hidden_28.astype(np.float32) - h0
```

两项均为读入当前词元后的状态，**不是**预测当前词元前的P+i−1；不要与NLL/MMD时点混淆。不额外单位长度归一化，不分别归一化两状态，不取绝对值/范数，不先各做PCA再相减。缓存保存原3584维；4BPE窗口均值、中心化及PCA留到下游训练管线，全部使用fit-only base weights。

若下游采用128或384维投影，维数与C须先冻结；两种都试则两种都报告。随机化SVD只拟合当折训练窗口，保存均值、基、奇异值和解释方差，不能先在602条全量数据上降维。差分后投影与窗口均值可交换，但与不同条件分别拟合的PCA不等价。降低C是待验证的正则化选择，不能保证解决泛化差距。

**输入隔离和缓存接口**

- 读取元数据先筛实际 `split == train`，沿用R19的逐行预筛方式，避免调用会先解析全部800行的 `base.source_rows()`。只读取当条冻结生成、R18特征缓存和对应manifest，不读任何标签、参考答案、探针得分、旧val/test。已核对R16输入schema本身只有模型可见内容与分组元数据，无参考答案键。
- 对602条全提取，不能按回答正误、是否拒答、能否解析来过滤；提取不读 `generated['items']`。只保存训练row_id集合对应的新目录，旧输入、生成、缓存均不修改。
- 新NPZ建议键：`hidden_no_ref_28`、`hidden_delta_28`（均float32[T,3584]），`token_ids`、`response_token_offsets`、`token_start`、`token_end`（沿用原缓存）。另存JSON：原/新prompt哈希、原/新前缀ID哈希与长度、原生成/R18缓存哈希、模型和代码签名、时点/归一化定义、耗时与峰值显存。新prompt全文可保存在元数据以复核，答案不复制改写。
- 两个新float32数组合计约409 MiB未压缩（按现有14,968个答案词元估算）；已有有资料状态不复制。每答单独回放，模型本体直接返回最终状态，不计算全词表logits、所有层hidden或全部attention，降低成本。

**运行前后只需的检查**

1. 不用GPU先检查602条prefix重建完全一致、marker唯一、删除覆盖全部来源标题正文、原答案ID和字符坐标保持不变。前缀中的特殊token仍由同一chat模板生成，不能按字符串手拼token边界。
2. 最少首尾两个train ID作为固定集成检查，不按标签选例。其中一条用未改资料提示走新提取函数，要求与其原 `hidden_28` 一致；identity差分应为0。若不一致先解释模型/时点/数值路径，不通过放宽容差直接开始全量。
3. 相同无资料输入重复回放检查一致性；正式逐条校验形状、dtype、有限数值、answer IDs与坐标。不同前缀长度引起的真实差异不要求为0，不把它当实现错误。
4. 全量结束后冻结manifest，之后小探针才能读取跨度标签训练。GPU预算和准确耗时待实际记录；一次无资料回放通常比原长提示短，这是输入长度推断，不是性能保证。

**解释边界**

这是离线重放同一Qwen已经生成的答案。删除资料会改变答案的绝对位置、问题到答案的距离、注意力归一化和数值计算形状，差分混合了这些因素，不能解释成“纯证据因果贡献”。它还条件于已经固定的答案前缀，无资料模型自然生成的内容可能不同。低差分不能直接定为幻觉。

因果模型每个位置仅依赖当前及之前词元；本方案没有双向平滑。但它尚未测试在线缓存数值一致性和实时延迟，报告为离线词元/窗口定位。若日后加入CORTEX原式前向—后向平滑，必须明确读取后续词元；不能改称实时监测。完整/部分资料的同题回答保持同事件组，且绝不向部分条件补回隐藏的正确证据。
