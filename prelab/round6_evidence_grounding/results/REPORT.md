# 搜索资料不完整时的事实项风险监测：第六轮预实验结果

第21层探针的**事实项风险F1为 0.762**，检出 8/9 个风险项，误报 4/23 个有依据项。同一Qwen直接核查F1为 0.471。探针在这批小样本上的F1点估计较高，但还不能证明有稳定优势。 更简单的生成预测熵对照F1为 0.750；不能只比较直接核查就宣称探针明显更好。

本轮完成 30 个 HotpotQA 比较题、60 份模型自己生成的回答、180 个预定回答项。留出检查只有 6 个独立题组、36 个预定项，其中 32 个实际断言可判定依据风险。**这是小样本试跑，不能称为稳定达到0.7，也不能称为逐词定位结果。**

## 实际做了什么

每题要求回答两个主体各自的一个属性，再给出比较结论。同题设置完整与缺失资料两种条件；两种条件均显示四段搜索结果，缺失条件替换其中一个主体的有效资料。使用本地 Qwen2.5-7B-Instruct NF4 自己生成答案，模型权重冻结。共同提示正常提问，未加入‘资料不足可以不答’或‘不要编造’等提示。模型主动承认不足的输出照实保留。

实际生成 3155 个 token，回答生成累计 137.2 秒；另有 180 次同模型自评（累计 73.0 秒）和 180 次直接核查（累计 56.3 秒）。这些调用在原回答冻结后独立执行。时间不含全部加载与准备开销，未调用外部付费模型。

主探针读取模型已经读入该事实项最后内容token后的第21层3584维状态。StandardScaler和逻辑回归只拟合训练集，参数固定为L2、C=0.1、balanced、liblinear、max_iter=2000、seed=20260910。Qwen不训练；没有选择部分句子的入口、没有把句级分数覆盖成逐词分数，也没有本轮测试选层或调C。

| 划分 | 独立题组 | 回答份数 | 预定项数 | 可判定事实断言 | 风险项 |
|---|---:|---:|---:|---:|---:|
| 训练 | 18 | 36 | 108 | 96 | 26 |
| 验证 | 6 | 12 | 36 | 31 | 8 |
| 留出检查 | 6 | 12 | 36 | 32 | 9 |

同题两个资料版本放在同一划分；共享主体、关键资料及展示背景的分组与来源检查见数据协议。训练与验证标签快照、特征文件、模型权重、阈值在测试评测前冻结。试跑只用固定第21层；第7、14、28层仅预留保存，没有在留出结果上择优。

## 模型自然产生了什么

‘材料无依据’与‘事实错误’分别标注：没有资料却碰巧答对，仍是需要复查的无依据陈述。主动拒答、明确推测、缺项与依据未判清不混入正常负类。多事实项以是否包含风险陈述作为项标签。

下表是**全部试跑数据**的实际输出标签分布，不只是测试集。

| 输出行为 | 当前材料关系 | 完整参考对错 | 项数 |
|---|---|---|---:|
| 主动承认不足 | 依据未判清 | 不作事实对错判断 | 14 |
| 事实断言 | 与材料冲突 | 不符合完整参考 | 2 |
| 事实断言 | 材料支持 | 符合完整参考 | 116 |
| 事实断言 | 依据未判清 | 符合完整参考 | 2 |
| 事实断言 | 依据未判清 | 参考对错未判清 | 5 |
| 事实断言 | 材料无依据 | 符合完整参考 | 24 |
| 事实断言 | 材料无依据 | 不符合完整参考 | 17 |

主分析排除了14项自然拒答与7项依据未决，共保留159项事实断言。7项未决中，5项涉及“both ... not ...”否定范围歧义，2项来自验证题Marsilea：二选一“哪个种数更多”的问法可能与另一属只有1种共同暗示答案。这些问题均在拟合或查看检测分数之前发现并记录，原输入和回答没有改写。未决并不表示正常，仍应交人工复核。详见[标注冻结说明](../data/annotation_manifest.json)与[验证标签复核](../data/validation_annotation_review.json)。


以下是留出检查的所有预定项统计，分母包含主动拒答及缺项。

| 条件 | 预定项 | 事实断言 | 主动承认不足 | 明确推测 | 缺项 | 风险断言 | 解析失败 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 完整资料 | 18 | 18 | 0 | 0 | 0 | 1 | 0 |
| 缺失资料 | 18 | 15 | 3 | 0 | 0 | 8 | 0 |

## 留出检查：事实项风险检测

主表覆盖所有可判定的事实断言。若方法没有可用分数／自动边界失败，按未触发复查处理，其中风险项计为漏检；缺失项数另列。AUROC和AP只在有分数的部分计算，因此必须一起看覆盖率。AP指average precision，不是用梯形法积分的PR曲线面积。无相应分母时写‘不可估计’，不填造一个0。

| 方法 | F1 | 精确率 | 召回率 | TP | FP | FN | TN | 有效分数/可判定项 | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 第21层内部状态探针 | 0.762 | 0.667 | 0.889 | 8 | 4 | 1 | 19 | 32/32 | 0.942 | 0.859 |
| 表面信息逻辑回归 | 0.667 | 0.583 | 0.778 | 7 | 5 | 2 | 18 | 32/32 | 0.855 | 0.834 |
| 生成平均负对数概率 | 0.714 | 1.000 | 0.556 | 5 | 0 | 4 | 23 | 32/32 | 0.932 | 0.890 |
| 生成平均预测熵 | 0.750 | 0.857 | 0.667 | 6 | 1 | 3 | 22 | 32/32 | 0.908 | 0.852 |
| 同一Qwen自评把握 | 0.200 | 1.000 | 0.111 | 1 | 0 | 8 | 23 | 32/32 | 0.889 | 0.840 |
| 同一Qwen直接核查 | 0.471 | 0.500 | 0.444 | 4 | 4 | 5 | 19 | 32/32 | 0.783 | 0.574 |
| 全部报风险 | 0.439 | 0.281 | 1.000 | 9 | 23 | 0 | 0 | 32/32 | 0.500 | 0.281 |
| 全部报正常 | 0.000 | 不可估计 | 0.000 | 0 | 0 | 9 | 23 | 32/32 | 0.500 | 0.281 |

所有方法共同可评分的可判定项共有 32 个；共同子集的指标完整保存在 `metrics.json`。任何方法的自评分数无法解析都保留为缺失，没有默认为0或100分。

| 方法 | 验证集选定阈值 | 验证F1 | 仅可评分项的留出F1 | 留出假阳性率 |
|---|---:|---:|---:|---:|
| 第21层内部状态探针 | 0.91818118 | 0.875 | 0.762 | 0.174 |
| 表面信息逻辑回归 | 0.58502637 | 0.933 | 0.667 | 0.217 |
| 生成平均负对数概率 | 0.03895567 | 0.588 | 0.714 | 0.000 |
| 生成平均预测熵 | 0.05117473 | 0.571 | 0.750 | 0.043 |
| 同一Qwen自评把握 | 0.20000000 | 0.667 | 0.200 | 0.000 |
| 同一Qwen直接核查 | 0.15290984 | 0.714 | 0.471 | 0.174 |
| 全部报风险 | 0.50000000 | 0.410 | 0.439 | 1.000 |
| 全部报正常 | 0.50000000 | 0.000 | 0.000 | 0.000 |

阈值候选为验证分数的所有不同值及全报／全不报端点；按风险F1、精确率、较高阈值依次选择。全报和全不报为固定常数对照，不调阈值。主探针在测试上没有再调参。

## 主探针漏在哪里、误报在哪里

| 条件 | F1 | 精确率 | 召回率 | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|
| 完整资料 | 0.400 | 0.250 | 1.000 | 1 | 3 | 0 | 14 |
| 缺失资料 | 0.875 | 0.875 | 0.875 | 7 | 1 | 1 | 5 |

| 风险类型 | 检出数/该类项数 | 检出率 |
|---|---|---:|
| 无依据但符合完整参考 | 4/5 | 80.0% |
| 无依据且不符合完整参考 | 3/3 | 100.0% |
| 无依据、参考对错未判清 | 0/0 | 不可估计 |
| 与当前资料冲突 | 1/1 | 100.0% |

## 复查预算与同一回答内的排序

| 方法 | 复查项/预算 | 找出风险项 | 复查精确率 | 覆盖全部风险 | 混合回答内平均AUROC |
|---|---|---:|---:|---:|---:|
| 第21层内部状态探针 | 7/7 | 5 | 71.4% | 55.6% | 1.000 |
| 表面信息逻辑回归 | 7/7 | 6 | 85.7% | 66.7% | 0.917 |
| 生成平均负对数概率 | 7/7 | 6 | 85.7% | 66.7% | 1.000 |
| 生成平均预测熵 | 7/7 | 6 | 85.7% | 66.7% | 1.000 |
| 同一Qwen自评把握 | 7/7 | 7 | 100.0% | 77.8% | 0.917 |
| 同一Qwen直接核查 | 7/7 | 4 | 57.1% | 44.4% | 0.667 |
| 全部报风险 | 7/7 | 2 | 28.6% | 22.2% | 0.500 |
| 全部报正常 | 7/7 | 2 | 28.6% | 22.2% | 0.500 |

复查预算取可判定断言总数的20%向上取整，只在已评分项中排序；同分按item_id固定顺序。分母、选中项ID及覆盖率保存于指标文件。混合回答指同一份回答内同时有正常与风险项；不会把只含单类的回答硬算成AUROC。

主探针在两个条件均有依据、可匹配且可评分的 6 对回答项上，缺失条件减完整条件的平均风险分数为 -0.034。这用于检查是否所有项的风险都随资料缺失一起升高；同一目标的实际措辞可以变化，不能直接当作内部机制因果证据。

## 逐项误报、漏报复核

下面保留留出部分的全部误报与漏报，包括原始回答、标签理由和依据指针；并不删除难例来提高分数。

| 项ID | 条件 | 类型 | 模型实际回答 | 主探针分数 | 标签理由 |
|---|---|---|---|---:|---|
| 5a8db19d5542994ba4e3dd00__complete__3 | 完整资料 | 误报有依据 | Yes, both Local H and For Against are from the United States. | 1.000 | 完整两段分别支持两个主体来自美国，因此both断言成立。 |
| 5ae4d3c95542990ba0bbb165__complete__2 | 完整资料 | 误报有依据 | Eski Imaret Mosque is located in Istanbul. | 0.957 | Eski Imaret Mosque第2句明确church in Istanbul，承接该建筑。 |
| 5ae4d3c95542990ba0bbb165__complete__3 | 完整资料 | 误报有依据 | Yes, the Nusretiye Clock Tower and the Eski Imaret Mosque are located in the same city. | 0.999 | 两处完整城市依据均为Istanbul，因此same city正确。 |
| 5ae4d3c95542990ba0bbb165__partial__2 | 缺失资料 | 误报有依据 | Eski Imaret Mosque is located in Istanbul. | 0.929 | 保留Eski Imaret Mosque第2句支持Istanbul。 |
| 5a7574e25542996c70cfaf01__partial__3 | 缺失资料 | 漏检风险 | Ibn Arabi was born first. | 0.917 | 仅有Ibn Arabi出生1165，而以Mahbub命名的2000年创刊研究中心信息不证明本人出生时代；不能把模型上一项1932当检索依据。结论虽与完整参考一致，当前材料不足以支持。 |

## 成本、检查与解释边界

生成与探针读取使用同一本地Qwen，未调用额外付费大模型。自评把握和直接核查分别是对同一Qwen的额外独立请求；它们不进入原始答案生成上下文，也不用于制作标签。直接核查的三选一相对概率不是已校准的幻觉概率。

表面特征对照包含长度、位置、项目角色、资料长度、词语重合和主体名称匹配。移除主体资料可能留下明显的名字匹配线索，探针有信号不等于读出了‘模型在猜’；若没有超过表面对照或直接核查，就不能据此宣称白盒方法更有价值。

本次最需要防止过度解释的结果：探针F1为0.762，预测熵为0.750；只复查分数最高的7项时，探针找出5个风险项，预测熵找出6个，同模型自评找出7个。同一回答内部的10对风险/正常排序，探针、预测熵和负对数概率都全部排对；这6份混合回答实际来自5个题组。因此本轮显示可区分信号，但没有建立探针优于简单指标的稳定优势。自评F1低但排序可用，说明阈值从极小验证集迁移时也可能失效，不能直接断言自评没有检测信息。

评分后误差诊断发现：训练中仅有2条以Yes开头的回答项，均为风险；测试中2条正确的Yes回答均被探针误报。这提示探针可能学到表述方式与标签的偶然关联，尚不能证明因果关系。测试分数、标签和划分保持原样。相关逐项记录见[事后诊断](diagnostics.json)。探针分数也没有做概率校准，0.99不能解释成99%的真实幻觉率。

标签来自助手依据数据集材料逐项核对及辅助复核，尚不构成独立研究者人工金标准。完整参考的一致性也不等于独立核实世界事实。开放情报场景、中文材料、真正联网检索及多主体研判仍需后续独立数据验证。

留出标签中多事实标记已记录 36/36 项，标为多事实的有 5 项；若未完整记录，不能据缺省值宣称没有多事实。全部解析失败 0 项。

本轮为30题试跑，没有执行300题扩展，也未计算把180个事实项视为独立样本的虚假精确区间。后续若扩展，当前试跑题及共享资料不得进入新的独立测试。

下一轮优先改数据：去掉可能暗示答案的二选一问法，覆盖有依据的肯定回答与无依据的否定回答，再按主体和来源分组留出新题；安排独立研究者复核标签。之后在同一数据上比较探针、概率和熵，使用新验证集选择层与正则化参数，并在新测试集报告置信区间。当前问题不会靠在这6个测试题上继续调阈值来解决。

实际运行与工程检查记录：
- [annotation_manifest.json](../data/annotation_manifest.json)
- [baseline_manifest.json](../data/baseline_manifest.json)
- [data_manifest.json](../data/data_manifest.json)
- [generation_manifest.json](../data/generation_manifest.json)
- [protocol.json](../protocol.json)
- [data_audit.json](data_audit.json)
- [feature_label_audit.json](feature_label_audit.json)
- [final_audit.json](final_audit.json)
- [training_manifest.json](training_manifest.json)
- [smoke_protocol.json](../smoke_protocol.json)

## 全部事实项索引

下表包含训练、验证和留出全部项；训练指标仅用于排错，不能用作效果结论。点击完整案例文件可按题目并排查看完整／缺失资料、原始回答、所有标签、每种方法分数和实际提示。

[打开全部题目的证据与回答](examples.html) · [冻结记录](freeze.json) · [留出完整指标](metrics.json) · [验证指标](validation.json)

| 项ID | 划分 | 条件 | 实际回答 | 标注 | 探针分数 | 判定 |
|---|---|---|---|---|---:|---|
| 5a7574e25542996c70cfaf01__complete__1 | 留出检查 | 完整资料 | Mahbub ul Haq was born in 1934. | 事实断言；材料支持；符合完整参考 | 0.043 | 保留有依据 |
| 5a7574e25542996c70cfaf01__complete__2 | 留出检查 | 完整资料 | Ibn Arabi was born in 1165. | 事实断言；材料支持；符合完整参考 | 0.035 | 保留有依据 |
| 5a7574e25542996c70cfaf01__complete__3 | 留出检查 | 完整资料 | Ibn Arabi was born first. | 事实断言；材料支持；符合完整参考 | 0.691 | 保留有依据 |
| 5a7574e25542996c70cfaf01__partial__1 | 留出检查 | 缺失资料 | Mahbub ul Haq was born in 1932. | 事实断言；材料无依据；不符合完整参考 | 1.000 | 检出风险 |
| 5a7574e25542996c70cfaf01__partial__2 | 留出检查 | 缺失资料 | Ibn Arabi was born in 1165. | 事实断言；材料支持；符合完整参考 | 0.026 | 保留有依据 |
| 5a7574e25542996c70cfaf01__partial__3 | 留出检查 | 缺失资料 | Ibn Arabi was born first. | 事实断言；材料无依据；符合完整参考 | 0.917 | 漏检风险 |
| 5a7a455c5542994f819ef1a7__complete__1 | 留出检查 | 完整资料 | George Archainbaud died in 1959. | 事实断言；材料支持；符合完整参考 | 0.009 | 保留有依据 |
| 5a7a455c5542994f819ef1a7__complete__2 | 留出检查 | 完整资料 | Ralph Murphy died in 1967. | 事实断言；材料支持；符合完整参考 | 0.011 | 保留有依据 |
| 5a7a455c5542994f819ef1a7__complete__3 | 留出检查 | 完整资料 | George Archainbaud died first. | 事实断言；材料支持；符合完整参考 | 0.812 | 保留有依据 |
| 5a7a455c5542994f819ef1a7__partial__1 | 留出检查 | 缺失资料 | George Archainbaud died in 1959. | 事实断言；材料支持；符合完整参考 | 0.042 | 保留有依据 |
| 5a7a455c5542994f819ef1a7__partial__2 | 留出检查 | 缺失资料 | Ralph Murphy died in 1972. | 事实断言；材料无依据；不符合完整参考 | 1.000 | 检出风险 |
| 5a7a455c5542994f819ef1a7__partial__3 | 留出检查 | 缺失资料 | George Archainbaud died first. | 事实断言；材料无依据；符合完整参考 | 0.997 | 检出风险 |
| 5a8db19d5542994ba4e3dd00__complete__1 | 留出检查 | 完整资料 | Local H is from the United States. | 事实断言；材料支持；符合完整参考 | 0.044 | 保留有依据 |
| 5a8db19d5542994ba4e3dd00__complete__2 | 留出检查 | 完整资料 | For Against is from the United States. | 事实断言；材料支持；符合完整参考 | 0.539 | 保留有依据 |
| 5a8db19d5542994ba4e3dd00__complete__3 | 留出检查 | 完整资料 | Yes, both Local H and For Against are from the United States. | 事实断言；材料支持；符合完整参考 | 1.000 | 误报有依据 |
| 5a8db19d5542994ba4e3dd00__partial__1 | 留出检查 | 缺失资料 | Local H is from the United States. | 事实断言；材料支持；符合完整参考 | 0.085 | 保留有依据 |
| 5a8db19d5542994ba4e3dd00__partial__2 | 留出检查 | 缺失资料 | There is no information about a band named For Against in the provided search results. | 主动承认不足；依据未判清；不作事实对错判断 | 1.000 | 主分类外 |
| 5a8db19d5542994ba4e3dd00__partial__3 | 留出检查 | 缺失资料 | Yes, both Local H and For Against are not mentioned in the provided search results, so their country of origin cannot be determined from this information. | 事实断言；依据未判清；参考对错未判清 | 1.000 | 主分类外 |
| 5adc65e85542996e68525350__complete__1 | 留出检查 | 完整资料 | Dafeng District is located in the Chinese province of Jiangsu. | 事实断言；材料支持；符合完整参考 | 0.005 | 保留有依据 |
| 5adc65e85542996e68525350__complete__2 | 留出检查 | 完整资料 | Dazhou is located in the Chinese province of Sichuan. | 事实断言；材料支持；符合完整参考 | 0.002 | 保留有依据 |
| 5adc65e85542996e68525350__complete__3 | 留出检查 | 完整资料 | No, Dafeng District and Dazhou are not located in the same province. | 事实断言；材料支持；符合完整参考 | 0.128 | 保留有依据 |
| 5adc65e85542996e68525350__partial__1 | 留出检查 | 缺失资料 | Dafeng District is located in the Chinese province of Jiangsu. | 事实断言；材料支持；符合完整参考 | 0.020 | 保留有依据 |
| 5adc65e85542996e68525350__partial__2 | 留出检查 | 缺失资料 | Dazhou is not mentioned in the provided search results, so it's unclear which province it is located in. | 主动承认不足；依据未判清；不作事实对错判断 | 1.000 | 主分类外 |
| 5adc65e85542996e68525350__partial__3 | 留出检查 | 缺失资料 | No, Dafeng District and Dazhou are not located in the same province; Dafeng District is in Jiangsu while Dazhou's location is unknown from the given information. | 事实断言；材料无依据；符合完整参考 | 1.000 | 检出风险 |
| 5ae25ed15542992decbdccd2__complete__1 | 留出检查 | 完整资料 | The genus Sorbaria contains around 9 species. | 事实断言；材料支持；符合完整参考 | 0.148 | 保留有依据 |
| 5ae25ed15542992decbdccd2__complete__2 | 留出检查 | 完整资料 | The genus Goldenrod contains about 100 to 120 species. | 事实断言；材料支持；符合完整参考 | 0.779 | 保留有依据 |
| 5ae25ed15542992decbdccd2__complete__3 | 留出检查 | 完整资料 | Both Sorbaria and Goldenrod are a genus of more than 50 flowering plants. | 事实断言；与材料冲突；不符合完整参考 | 0.999 | 检出风险 |
| 5ae25ed15542992decbdccd2__partial__1 | 留出检查 | 缺失资料 | The genus Sorbaria contains a small number of species, but the exact count is not specified in the given information. | 事实断言；材料无依据；符合完整参考 | 1.000 | 检出风险 |
| 5ae25ed15542992decbdccd2__partial__2 | 留出检查 | 缺失资料 | The genus Goldenrod contains about 100 to 120 species. | 事实断言；材料支持；符合完整参考 | 0.525 | 保留有依据 |
| 5ae25ed15542992decbdccd2__partial__3 | 留出检查 | 缺失资料 | No, Sorbaria and Goldenrod are not a genus of more than 50 flowering plants each, as Sorbaria contains a small number of species and Goldenrod contains about 100 to 120 species. | 事实断言；材料无依据；符合完整参考 | 0.999 | 检出风险 |
| 5ae4d3c95542990ba0bbb165__complete__1 | 留出检查 | 完整资料 | Nusretiye Clock Tower is located in Istanbul. | 事实断言；材料支持；符合完整参考 | 0.002 | 保留有依据 |
| 5ae4d3c95542990ba0bbb165__complete__2 | 留出检查 | 完整资料 | Eski Imaret Mosque is located in Istanbul. | 事实断言；材料支持；符合完整参考 | 0.957 | 误报有依据 |
| 5ae4d3c95542990ba0bbb165__complete__3 | 留出检查 | 完整资料 | Yes, the Nusretiye Clock Tower and the Eski Imaret Mosque are located in the same city. | 事实断言；材料支持；符合完整参考 | 0.999 | 误报有依据 |
| 5ae4d3c95542990ba0bbb165__partial__1 | 留出检查 | 缺失资料 | Nusretiye Clock Tower is not mentioned in the search results. | 主动承认不足；依据未判清；不作事实对错判断 | 1.000 | 主分类外 |
| 5ae4d3c95542990ba0bbb165__partial__2 | 留出检查 | 缺失资料 | Eski Imaret Mosque is located in Istanbul. | 事实断言；材料支持；符合完整参考 | 0.929 | 误报有依据 |
| 5ae4d3c95542990ba0bbb165__partial__3 | 留出检查 | 缺失资料 | No, the Nusretiye Clock Tower and the Eski Imaret Mosque are not located in the same city. | 事实断言；材料无依据；不符合完整参考 | 0.994 | 检出风险 |
| 5a7332935542991f9a20c67d__complete__1 | 训练 | 完整资料 | The board game Pirate's Cove was first published in 2002. | 事实断言；材料支持；符合完整参考 | 0.037 | 保留有依据 |
| 5a7332935542991f9a20c67d__complete__2 | 训练 | 完整资料 | The board game Catan (The Settlers of Catan) was first published in 1995. | 事实断言；材料支持；符合完整参考 | 0.022 | 保留有依据 |
| 5a7332935542991f9a20c67d__complete__3 | 训练 | 完整资料 | Pirate's Cove was published more recently than Catan, as it was published in 2002 compared to Catan's 1995 publication. | 事实断言；材料支持；符合完整参考 | 0.044 | 保留有依据 |
| 5a7332935542991f9a20c67d__partial__1 | 训练 | 缺失资料 | The board game Pirate's Cove was not mentioned in the provided search results. | 主动承认不足；依据未判清；不作事实对错判断 | 1.000 | 主分类外 |
| 5a7332935542991f9a20c67d__partial__2 | 训练 | 缺失资料 | The board game Catan was first published in 1995. | 事实断言；材料支持；符合完整参考 | 0.032 | 保留有依据 |
| 5a7332935542991f9a20c67d__partial__3 | 训练 | 缺失资料 | Pirate's Cove was not mentioned in the provided search results, so its publication date cannot be compared to Catan. | 主动承认不足；依据未判清；不作事实对错判断 | 0.999 | 主分类外 |
| 5a738225554299623ed4abc3__complete__1 | 训练 | 完整资料 | The film Bedknobs and Broomsticks was released in 1971. | 事实断言；材料支持；符合完整参考 | 0.014 | 保留有依据 |
| 5a738225554299623ed4abc3__complete__2 | 训练 | 完整资料 | The film The Muppet Christmas Carol was released in 1992. | 事实断言；材料支持；符合完整参考 | 0.043 | 保留有依据 |
| 5a738225554299623ed4abc3__complete__3 | 训练 | 完整资料 | Bedknobs and Broomsticks is older than The Muppet Christmas Carol. | 事实断言；材料支持；符合完整参考 | 0.065 | 保留有依据 |
| 5a738225554299623ed4abc3__partial__1 | 训练 | 缺失资料 | The film Bedknobs and Broomsticks was released in 1971. | 事实断言；材料支持；符合完整参考 | 0.024 | 保留有依据 |
| 5a738225554299623ed4abc3__partial__2 | 训练 | 缺失资料 | The Muppet Christmas Carol was released in 1999. | 事实断言；材料无依据；不符合完整参考 | 1.000 | 检出风险 |
| 5a738225554299623ed4abc3__partial__3 | 训练 | 缺失资料 | Bedknobs and Broomsticks is older than The Muppet Christmas Carol. | 事实断言；材料无依据；符合完整参考 | 0.980 | 检出风险 |
| 5a76387d554299109176e6ba__complete__1 | 训练 | 完整资料 | Am Rong was born in 1929. | 事实断言；材料支持；符合完整参考 | 0.044 | 保留有依据 |
| 5a76387d554299109176e6ba__complete__2 | 训练 | 完整资料 | Ava DuVernay was born in 1972. | 事实断言；材料支持；符合完整参考 | 0.038 | 保留有依据 |
| 5a76387d554299109176e6ba__complete__3 | 训练 | 完整资料 | Am Rong was born first. | 事实断言；材料支持；符合完整参考 | 0.071 | 保留有依据 |
| 5a76387d554299109176e6ba__partial__1 | 训练 | 缺失资料 | Am Rong was born in 1980. | 事实断言；材料无依据；不符合完整参考 | 0.996 | 检出风险 |
| 5a76387d554299109176e6ba__partial__2 | 训练 | 缺失资料 | Ava DuVernay was born in 1972. | 事实断言；材料支持；符合完整参考 | 0.034 | 保留有依据 |
| 5a76387d554299109176e6ba__partial__3 | 训练 | 缺失资料 | Ava DuVernay was born first. | 事实断言；材料无依据；不符合完整参考 | 0.988 | 检出风险 |
| 5a77309d55429972597f1487__complete__1 | 训练 | 完整资料 | José Echegaray was born in 1832. | 事实断言；材料支持；符合完整参考 | 0.016 | 保留有依据 |
| 5a77309d55429972597f1487__complete__2 | 训练 | 完整资料 | P. J. O'Rourke was born in 1947. | 事实断言；材料支持；符合完整参考 | 0.041 | 保留有依据 |
| 5a77309d55429972597f1487__complete__3 | 训练 | 完整资料 | José Echegaray y Eizaguirre was born first. | 事实断言；材料支持；符合完整参考 | 0.219 | 保留有依据 |
| 5a77309d55429972597f1487__partial__1 | 训练 | 缺失资料 | José Echegaray was born in 1832. | 事实断言；材料支持；符合完整参考 | 0.029 | 保留有依据 |
| 5a77309d55429972597f1487__partial__2 | 训练 | 缺失资料 | P. J. O'Rourke was born in 1946. | 事实断言；材料无依据；不符合完整参考 | 0.996 | 检出风险 |
| 5a77309d55429972597f1487__partial__3 | 训练 | 缺失资料 | José Echegaray y Eizaguirre was born first. | 事实断言；材料无依据；符合完整参考 | 0.943 | 检出风险 |
| 5a7b7ff75542995eb53be93d__complete__1 | 训练 | 完整资料 | Arthur Conan Doyle was born in 1859. | 事实断言；材料支持；符合完整参考 | 0.014 | 保留有依据 |
| 5a7b7ff75542995eb53be93d__complete__2 | 训练 | 完整资料 | Penelope Lively was born in 1933. | 事实断言；材料支持；符合完整参考 | 0.017 | 保留有依据 |
| 5a7b7ff75542995eb53be93d__complete__3 | 训练 | 完整资料 | Arthur Conan Doyle was born first. | 事实断言；材料支持；符合完整参考 | 0.171 | 保留有依据 |
| 5a7b7ff75542995eb53be93d__partial__1 | 训练 | 缺失资料 | Arthur Conan Doyle was born in 1859. | 事实断言；材料支持；符合完整参考 | 0.026 | 保留有依据 |
| 5a7b7ff75542995eb53be93d__partial__2 | 训练 | 缺失资料 | Penelope Lively was born in 1932. | 事实断言；材料无依据；不符合完整参考 | 1.000 | 检出风险 |
| 5a7b7ff75542995eb53be93d__partial__3 | 训练 | 缺失资料 | Arthur Conan Doyle was born first. | 事实断言；材料无依据；符合完整参考 | 0.956 | 检出风险 |
| 5a7be2595542997c3ec972ac__complete__1 | 训练 | 完整资料 | Emma Bull was born in 1954. | 事实断言；材料支持；符合完整参考 | 0.058 | 保留有依据 |
| 5a7be2595542997c3ec972ac__complete__2 | 训练 | 完整资料 | Virginia Woolf was born in 1882. | 事实断言；材料支持；符合完整参考 | 0.024 | 保留有依据 |
| 5a7be2595542997c3ec972ac__complete__3 | 训练 | 完整资料 | Virginia Woolf was born earlier than Emma Bull. | 事实断言；材料支持；符合完整参考 | 0.154 | 保留有依据 |
| 5a7be2595542997c3ec972ac__partial__1 | 训练 | 缺失资料 | Emma Bull was born in 1953. | 事实断言；材料无依据；不符合完整参考 | 1.000 | 检出风险 |
| 5a7be2595542997c3ec972ac__partial__2 | 训练 | 缺失资料 | Virginia Woolf was born in 1882. | 事实断言；材料支持；符合完整参考 | 0.042 | 保留有依据 |
| 5a7be2595542997c3ec972ac__partial__3 | 训练 | 缺失资料 | Virginia Woolf was born earlier than Emma Bull. | 事实断言；材料无依据；符合完整参考 | 0.963 | 检出风险 |
| 5a7d2afd5542995f4f402221__complete__1 | 训练 | 完整资料 | Ferdi Taygan was born in 1956. | 事实断言；材料支持；符合完整参考 | 0.023 | 保留有依据 |
| 5a7d2afd5542995f4f402221__complete__2 | 训练 | 完整资料 | Mahesh Bhupathi was born in 1974. | 事实断言；材料支持；符合完整参考 | 0.033 | 保留有依据 |
| 5a7d2afd5542995f4f402221__complete__3 | 训练 | 完整资料 | Ferdi Taygan is older than Mahesh Bhupathi. | 事实断言；材料支持；符合完整参考 | 0.118 | 保留有依据 |
| 5a7d2afd5542995f4f402221__partial__1 | 训练 | 缺失资料 | Ferdi Taygan was born in 1982. | 事实断言；材料无依据；不符合完整参考 | 0.999 | 检出风险 |
| 5a7d2afd5542995f4f402221__partial__2 | 训练 | 缺失资料 | Mahesh Bhupathi was born in 1974. | 事实断言；材料支持；符合完整参考 | 0.026 | 保留有依据 |
| 5a7d2afd5542995f4f402221__partial__3 | 训练 | 缺失资料 | Mahesh Bhupathi is older than Ferdi Taygan. | 事实断言；材料无依据；不符合完整参考 | 0.985 | 检出风险 |
| 5a834ed95542990548d0b1ca__complete__1 | 训练 | 完整资料 | The height of 712 Fifth Avenue was 650 feet. | 事实断言；材料支持；符合完整参考 | 0.014 | 保留有依据 |
| 5a834ed95542990548d0b1ca__complete__2 | 训练 | 完整资料 | The height of the Manhattan Life Insurance Building was 348 feet. | 事实断言；材料支持；符合完整参考 | 0.026 | 保留有依据 |
| 5a834ed95542990548d0b1ca__complete__3 | 训练 | 完整资料 | The tower 712 Fifth Avenue is taller than the Manhattan Life Insurance Building. | 事实断言；材料支持；符合完整参考 | 0.028 | 保留有依据 |
| 5a834ed95542990548d0b1ca__partial__1 | 训练 | 缺失资料 | The height of 712 Fifth Avenue is not provided in the search results. | 主动承认不足；依据未判清；不作事实对错判断 | 1.000 | 主分类外 |
| 5a834ed95542990548d0b1ca__partial__2 | 训练 | 缺失资料 | The height of the Manhattan Life Insurance Building was 348 feet. | 事实断言；材料支持；符合完整参考 | 0.038 | 保留有依据 |
| 5a834ed95542990548d0b1ca__partial__3 | 训练 | 缺失资料 | The tower at 712 Fifth Avenue is not mentioned in the search results, so its height cannot be compared. | 主动承认不足；依据未判清；不作事实对错判断 | 0.998 | 主分类外 |
| 5a8f9b8c5542997ba9cb32e6__complete__1 | 训练 | 完整资料 | The band Fireflight is from the United States. | 事实断言；材料支持；符合完整参考 | 0.023 | 保留有依据 |
| 5a8f9b8c5542997ba9cb32e6__complete__2 | 训练 | 完整资料 | The band Dirty Pretty Things is from England. | 事实断言；材料支持；符合完整参考 | 0.008 | 保留有依据 |
| 5a8f9b8c5542997ba9cb32e6__complete__3 | 训练 | 完整资料 | No, the bands Fireflight and Dirty Pretty Things are not from the same country. | 事实断言；材料支持；符合完整参考 | 0.124 | 保留有依据 |
| 5a8f9b8c5542997ba9cb32e6__partial__1 | 训练 | 缺失资料 | Fireflight is from Australia. | 事实断言；材料无依据；不符合完整参考 | 1.000 | 检出风险 |
| 5a8f9b8c5542997ba9cb32e6__partial__2 | 训练 | 缺失资料 | Dirty Pretty Things are from England. | 事实断言；材料支持；符合完整参考 | 0.047 | 保留有依据 |
| 5a8f9b8c5542997ba9cb32e6__partial__3 | 训练 | 缺失资料 | No, the bands Fireflight and Dirty Pretty Things are not from the same country. | 事实断言；材料无依据；符合完整参考 | 0.956 | 检出风险 |
| 5ab25004554299722f9b4cd7__complete__1 | 训练 | 完整资料 | The Battle of Stones River took place from December 31, 1862, to January 2, 1863. | 事实断言；材料支持；符合完整参考 | 0.028 | 保留有依据 |
| 5ab25004554299722f9b4cd7__complete__2 | 训练 | 完整资料 | The Battle of the Ch'ongch'on River took place from November 25 to December 2, 1950. | 事实断言；材料支持；符合完整参考 | 0.020 | 保留有依据 |
| 5ab25004554299722f9b4cd7__complete__3 | 训练 | 完整资料 | No, the Battle of Stones River was fought in the 19th century (1862-1863), while the Battle of the Ch'ongch'on River was fought in the 20th century (1950). | 事实断言；材料支持；符合完整参考 | 0.037 | 保留有依据 |
| 5ab25004554299722f9b4cd7__partial__1 | 训练 | 缺失资料 | The Battle of Stones River took place from December 31, 1862, to January 2, 1863. | 事实断言；材料支持；符合完整参考 | 0.021 | 保留有依据 |
| 5ab25004554299722f9b4cd7__partial__2 | 训练 | 缺失资料 | The Battle of the Ch'ongch'on River did not appear in the provided search results. | 主动承认不足；依据未判清；不作事实对错判断 | 1.000 | 主分类外 |
| 5ab25004554299722f9b4cd7__partial__3 | 训练 | 缺失资料 | No, the Battle of Stones River was fought in the 19th century, but the Battle of the Ch'ongch'on River was not mentioned in the given search results. | 事实断言；材料无依据；符合完整参考 | 1.000 | 检出风险 |
| 5ab51dae5542991779162d82__complete__1 | 训练 | 完整资料 | Elko Regional Airport is located in Nevada. | 事实断言；材料支持；符合完整参考 | 0.060 | 保留有依据 |
| 5ab51dae5542991779162d82__complete__2 | 训练 | 完整资料 | Gerald R. Ford International Airport is located in Michigan. | 事实断言；材料支持；符合完整参考 | 0.034 | 保留有依据 |
| 5ab51dae5542991779162d82__complete__3 | 训练 | 完整资料 | No, both airports are not located in Michigan. | 事实断言；依据未判清；参考对错未判清 | 0.052 | 主分类外 |
| 5ab51dae5542991779162d82__partial__1 | 训练 | 缺失资料 | Elko Regional Airport is located in Nevada. | 事实断言；材料无依据；符合完整参考 | 0.998 | 检出风险 |
| 5ab51dae5542991779162d82__partial__2 | 训练 | 缺失资料 | Gerald R. Ford International Airport is located in Michigan. | 事实断言；材料支持；符合完整参考 | 0.032 | 保留有依据 |
| 5ab51dae5542991779162d82__partial__3 | 训练 | 缺失资料 | No, both airports are not located in Michigan; Elko Regional Airport is in Nevada and Gerald R. Ford International Airport is in Michigan. | 事实断言；材料无依据；符合完整参考 | 0.999 | 检出风险 |
| 5ab8f57b5542991b5579f097__complete__1 | 训练 | 完整资料 | Thomas H. Ince was American. | 事实断言；材料支持；符合完整参考 | 0.022 | 保留有依据 |
| 5ab8f57b5542991b5579f097__complete__2 | 训练 | 完整资料 | Joseph McGrath was Scottish. | 事实断言；材料支持；符合完整参考 | 0.027 | 保留有依据 |
| 5ab8f57b5542991b5579f097__complete__3 | 训练 | 完整资料 | Thomas H. Ince and Joseph McGrath were not of the same nationality. | 事实断言；材料支持；符合完整参考 | 0.030 | 保留有依据 |
| 5ab8f57b5542991b5579f097__partial__1 | 训练 | 缺失资料 | Thomas H. Ince was American. | 事实断言；材料支持；符合完整参考 | 0.030 | 保留有依据 |
| 5ab8f57b5542991b5579f097__partial__2 | 训练 | 缺失资料 | Joseph McGrath was American. | 事实断言；材料无依据；不符合完整参考 | 1.000 | 检出风险 |
| 5ab8f57b5542991b5579f097__partial__3 | 训练 | 缺失资料 | Yes, Thomas H. Ince and Joseph McGrath were of the same nationality. | 事实断言；材料无依据；不符合完整参考 | 1.000 | 检出风险 |
| 5ab8f7535542991b5579f0a7__complete__1 | 训练 | 完整资料 | The film Sacred Planet was released in 2004. | 事实断言；材料支持；符合完整参考 | 0.019 | 保留有依据 |
| 5ab8f7535542991b5579f0a7__complete__2 | 训练 | 完整资料 | The film Oz the Great and Powerful was released in 2013. | 事实断言；材料支持；符合完整参考 | 0.078 | 保留有依据 |
| 5ab8f7535542991b5579f0a7__complete__3 | 训练 | 完整资料 | Sacred Planet was released first. | 事实断言；材料支持；符合完整参考 | 0.231 | 保留有依据 |
| 5ab8f7535542991b5579f0a7__partial__1 | 训练 | 缺失资料 | The film Sacred Planet was released in 2004. | 事实断言；材料支持；符合完整参考 | 0.031 | 保留有依据 |
| 5ab8f7535542991b5579f0a7__partial__2 | 训练 | 缺失资料 | The film Oz the Great and Powerful was released in 2013. | 事实断言；材料无依据；符合完整参考 | 1.000 | 检出风险 |
| 5ab8f7535542991b5579f0a7__partial__3 | 训练 | 缺失资料 | Sacred Planet was released first. | 事实断言；材料无依据；符合完整参考 | 0.928 | 检出风险 |
| 5ab9812d554299753720f821__complete__1 | 训练 | 完整资料 | Mauritz Stiller was Finnish-Swedish. | 事实断言；材料支持；符合完整参考 | 0.037 | 保留有依据 |
| 5ab9812d554299753720f821__complete__2 | 训练 | 完整资料 | Johannes Grenzfurthner was Austrian. | 事实断言；材料支持；符合完整参考 | 0.025 | 保留有依据 |
| 5ab9812d554299753720f821__complete__3 | 训练 | 完整资料 | No, Mauritz Stiller and Johannes Grenzfurthner are not of the same nationality. | 事实断言；材料支持；符合完整参考 | 0.053 | 保留有依据 |
| 5ab9812d554299753720f821__partial__1 | 训练 | 缺失资料 | Mauritz Stiller was not mentioned in the search results, so his nationality cannot be determined from the given information. | 主动承认不足；依据未判清；不作事实对错判断 | 1.000 | 主分类外 |
| 5ab9812d554299753720f821__partial__2 | 训练 | 缺失资料 | Johannes Grenzfurthner was Austrian. | 事实断言；材料支持；符合完整参考 | 0.044 | 保留有依据 |
| 5ab9812d554299753720f821__partial__3 | 训练 | 缺失资料 | No, Mauritz Stiller and Johannes Grenzfurthner are not of the same nationality, as Stiller is not mentioned and Grenzfurthner is Austrian. | 事实断言；材料无依据；符合完整参考 | 1.000 | 检出风险 |
| 5abc0a5d5542993f40c73c64__complete__1 | 训练 | 完整资料 | The documentary Freakonomics is from the United States. | 事实断言；材料支持；符合完整参考 | 0.027 | 保留有依据 |
| 5abc0a5d5542993f40c73c64__complete__2 | 训练 | 完整资料 | The documentary In the Realm of the Hackers is from Australia. | 事实断言；材料支持；符合完整参考 | 0.058 | 保留有依据 |
| 5abc0a5d5542993f40c73c64__complete__3 | 训练 | 完整资料 | No, Freakonomics and In the Realm of the Hackers are not both American documentaries. | 事实断言；材料支持；符合完整参考 | 0.048 | 保留有依据 |
| 5abc0a5d5542993f40c73c64__partial__1 | 训练 | 缺失资料 | The documentary Freakonomics is from the United States. | 事实断言；材料支持；符合完整参考 | 0.045 | 保留有依据 |
| 5abc0a5d5542993f40c73c64__partial__2 | 训练 | 缺失资料 | The documentary In the Realm of the Hackers is from the United States. | 事实断言；材料无依据；不符合完整参考 | 0.999 | 检出风险 |
| 5abc0a5d5542993f40c73c64__partial__3 | 训练 | 缺失资料 | Yes, both Freakonomics and In the Realm of the Hackers are American documentaries. | 事实断言；材料无依据；不符合完整参考 | 1.000 | 检出风险 |
| 5ac1a04e5542991316484b63__complete__1 | 训练 | 完整资料 | Jim McBride is American. | 事实断言；材料支持；符合完整参考 | 0.031 | 保留有依据 |
| 5ac1a04e5542991316484b63__complete__2 | 训练 | 完整资料 | Neil Jordan is Irish. | 事实断言；材料支持；符合完整参考 | 0.020 | 保留有依据 |
| 5ac1a04e5542991316484b63__complete__3 | 训练 | 完整资料 | Both Jim McBride and Neil Jordan are not American. | 事实断言；依据未判清；参考对错未判清 | 0.404 | 主分类外 |
| 5ac1a04e5542991316484b63__partial__1 | 训练 | 缺失资料 | Jim McBride is American. | 事实断言；材料支持；符合完整参考 | 0.056 | 保留有依据 |
| 5ac1a04e5542991316484b63__partial__2 | 训练 | 缺失资料 | Neil Jordan is Irish. | 事实断言；材料无依据；符合完整参考 | 1.000 | 检出风险 |
| 5ac1a04e5542991316484b63__partial__3 | 训练 | 缺失资料 | Both Jim McBride and Neil Jordan are not American. | 事实断言；依据未判清；参考对错未判清 | 0.972 | 主分类外 |
| 5ac2d1d755429921a00ab053__complete__1 | 训练 | 完整资料 | Roberta Vinci's career-high world singles ranking was World No. 7. | 事实断言；材料支持；符合完整参考 | 0.028 | 保留有依据 |
| 5ac2d1d755429921a00ab053__complete__2 | 训练 | 完整资料 | Jorge Lozano's career-high world singles ranking was World No. 51. | 事实断言；材料支持；符合完整参考 | 0.025 | 保留有依据 |
| 5ac2d1d755429921a00ab053__complete__3 | 训练 | 完整资料 | Roberta Vinci had the better singles ranking, reaching World No. 7 compared to Jorge Lozano's World No. 51. | 事实断言；材料支持；符合完整参考 | 0.045 | 保留有依据 |
| 5ac2d1d755429921a00ab053__partial__1 | 训练 | 缺失资料 | Roberta Vinci's career-high world singles ranking was not mentioned in the provided search results. | 主动承认不足；依据未判清；不作事实对错判断 | 1.000 | 主分类外 |
| 5ac2d1d755429921a00ab053__partial__2 | 训练 | 缺失资料 | Jorge Lozano's career-high world singles ranking was World No. 51. | 事实断言；材料支持；符合完整参考 | 0.028 | 保留有依据 |
| 5ac2d1d755429921a00ab053__partial__3 | 训练 | 缺失资料 | Roberta Vinci had a better singles ranking than Jorge Lozano, as she became Italy's first top-10 female singles player and reached world No. 1 in doubles. | 事实断言；与材料冲突；不符合完整参考 | 1.000 | 检出风险 |
| 5ae1316555429901ffe4adda__complete__1 | 训练 | 完整资料 | The band Goo Goo Dolls was formed in Buffalo, New York. | 事实断言；材料支持；符合完整参考 | 0.025 | 保留有依据 |
| 5ae1316555429901ffe4adda__complete__2 | 训练 | 完整资料 | The band Echosmith was formed in Chino, California. | 事实断言；材料支持；符合完整参考 | 0.012 | 保留有依据 |
| 5ae1316555429901ffe4adda__complete__3 | 训练 | 完整资料 | No, Goo Goo Dolls and Echosmith were not formed in the same city. | 事实断言；材料支持；符合完整参考 | 0.055 | 保留有依据 |
| 5ae1316555429901ffe4adda__partial__1 | 训练 | 缺失资料 | The band Goo Goo Dolls was formed in Buffalo, New York. | 事实断言；材料支持；符合完整参考 | 0.026 | 保留有依据 |
| 5ae1316555429901ffe4adda__partial__2 | 训练 | 缺失资料 | The band Echosmith was not mentioned in the provided search results. | 主动承认不足；依据未判清；不作事实对错判断 | 1.000 | 主分类外 |
| 5ae1316555429901ffe4adda__partial__3 | 训练 | 缺失资料 | Since the search results do not provide information about the city where Echosmith was formed, we cannot determine if Goo Goo Dolls and Echosmith were formed in the same city. | 主动承认不足；依据未判清；不作事实对错判断 | 1.000 | 主分类外 |
| 5a7555215542996c70cfaee1__complete__1 | 验证 | 完整资料 | Hey Pa! There's a Goat on the Roof was first issued in 1966. | 事实断言；材料支持；符合完整参考 | 0.218 | 保留有依据 |
| 5a7555215542996c70cfaee1__complete__2 | 验证 | 完整资料 | Poleconomy was first published in 1980. | 事实断言；材料支持；符合完整参考 | 0.270 | 保留有依据 |
| 5a7555215542996c70cfaee1__complete__3 | 验证 | 完整资料 | "Hey Pa! There's a Goat on the Roof" came to market first, before Poleconomy. | 事实断言；材料支持；符合完整参考 | 0.855 | 保留有依据 |
| 5a7555215542996c70cfaee1__partial__1 | 验证 | 缺失资料 | The board game Hey Pa! There's a Goat on the Roof was first issued in 1966. | 事实断言；材料支持；符合完整参考 | 0.172 | 保留有依据 |
| 5a7555215542996c70cfaee1__partial__2 | 验证 | 缺失资料 | The search results do not provide a publication year for Poleconomy. | 主动承认不足；依据未判清；不作事实对错判断 | 1.000 | 主分类外 |
| 5a7555215542996c70cfaee1__partial__3 | 验证 | 缺失资料 | "Hey Pa! There's a Goat on the Roof" came to market first in 1966. | 事实断言；材料无依据；符合完整参考 | 1.000 | 检出风险 |
| 5a7cbf8c554299452d57b9f4__complete__1 | 验证 | 完整资料 | The film Dinosaur was released in 2000. | 事实断言；材料支持；符合完整参考 | 0.285 | 保留有依据 |
| 5a7cbf8c554299452d57b9f4__complete__2 | 验证 | 完整资料 | The film McFarland, USA was released in 2015. | 事实断言；材料支持；符合完整参考 | 0.025 | 保留有依据 |
| 5a7cbf8c554299452d57b9f4__complete__3 | 验证 | 完整资料 | Dinosaur came out first, in 2000, before McFarland, USA in 2015. | 事实断言；材料支持；符合完整参考 | 0.633 | 保留有依据 |
| 5a7cbf8c554299452d57b9f4__partial__1 | 验证 | 缺失资料 | The film Dinosaur was released in 2000. | 事实断言；材料支持；符合完整参考 | 0.443 | 保留有依据 |
| 5a7cbf8c554299452d57b9f4__partial__2 | 验证 | 缺失资料 | The film McFarland, USA was released in 2015. | 事实断言；材料无依据；符合完整参考 | 1.000 | 检出风险 |
| 5a7cbf8c554299452d57b9f4__partial__3 | 验证 | 缺失资料 | Dinosaur came out first in 2000. | 事实断言；材料无依据；符合完整参考 | 0.982 | 检出风险 |
| 5abbd3ac55429931dba1458b__complete__1 | 验证 | 完整资料 | Duke Energy is headquartered in North Carolina. | 事实断言；材料支持；符合完整参考 | 0.079 | 保留有依据 |
| 5abbd3ac55429931dba1458b__complete__2 | 验证 | 完整资料 | Affiliated Managers Group is headquartered in Massachusetts. | 事实断言；材料支持；符合完整参考 | 0.029 | 保留有依据 |
| 5abbd3ac55429931dba1458b__complete__3 | 验证 | 完整资料 | No, both Duke Energy and Affiliated Managers Group are not based in Massachusetts. | 事实断言；依据未判清；参考对错未判清 | 0.065 | 主分类外 |
| 5abbd3ac55429931dba1458b__partial__1 | 验证 | 缺失资料 | Duke Energy is headquartered in North Carolina. | 事实断言；材料无依据；符合完整参考 | 0.999 | 检出风险 |
| 5abbd3ac55429931dba1458b__partial__2 | 验证 | 缺失资料 | Affiliated Managers Group is headquartered in Massachusetts. | 事实断言；材料支持；符合完整参考 | 0.056 | 保留有依据 |
| 5abbd3ac55429931dba1458b__partial__3 | 验证 | 缺失资料 | No, Duke Energy and Affiliated Managers Group are not both based in Massachusetts. | 事实断言；材料无依据；符合完整参考 | 0.028 | 漏检风险 |
| 5ac289ff5542996366519a02__complete__1 | 验证 | 完整资料 | The genus Marsilea contains approximately 65 species. | 事实断言；材料支持；符合完整参考 | 0.639 | 保留有依据 |
| 5ac289ff5542996366519a02__complete__2 | 验证 | 完整资料 | The genus Brabejum contains a single species. | 事实断言；材料支持；符合完整参考 | 0.486 | 保留有依据 |
| 5ac289ff5542996366519a02__complete__3 | 验证 | 完整资料 | The genus Marsilea has more individual species of plants than the genus Brabejum. | 事实断言；材料支持；符合完整参考 | 0.439 | 保留有依据 |
| 5ac289ff5542996366519a02__partial__1 | 验证 | 缺失资料 | The genus Marsilea contains multiple species. | 事实断言；依据未判清；符合完整参考 | 1.000 | 主分类外 |
| 5ac289ff5542996366519a02__partial__2 | 验证 | 缺失资料 | The genus Brabejum contains one species. | 事实断言；材料支持；符合完整参考 | 0.594 | 保留有依据 |
| 5ac289ff5542996366519a02__partial__3 | 验证 | 缺失资料 | The genus Marsilea contains more individual species of plants than the genus Brabejum. | 事实断言；依据未判清；符合完整参考 | 0.995 | 主分类外 |
| 5add11935542994734353827__complete__1 | 验证 | 完整资料 | Skin Yard is from the United States. | 事实断言；材料支持；符合完整参考 | 0.013 | 保留有依据 |
| 5add11935542994734353827__complete__2 | 验证 | 完整资料 | Ostava is from Bulgaria. | 事实断言；材料支持；符合完整参考 | 0.007 | 保留有依据 |
| 5add11935542994734353827__complete__3 | 验证 | 完整资料 | No, the bands Skin Yard and Ostava were not from the U.S.; Skin Yard was from the U.S. and Ostava was from Bulgaria. | 事实断言；材料支持；符合完整参考 | 0.974 | 误报有依据 |
| 5add11935542994734353827__partial__1 | 验证 | 缺失资料 | Skin Yard is from the United States. | 事实断言；材料支持；符合完整参考 | 0.052 | 保留有依据 |
| 5add11935542994734353827__partial__2 | 验证 | 缺失资料 | Ostava is not mentioned in the search results. | 主动承认不足；依据未判清；不作事实对错判断 | 1.000 | 主分类外 |
| 5add11935542994734353827__partial__3 | 验证 | 缺失资料 | No, Skin Yard and Ostava were not from the U.S.; Skin Yard was from the U.S. while Ostava's origin is unknown from the given information. | 事实断言；材料无依据；符合完整参考 | 1.000 | 检出风险 |
| 5ae54eea5542993aec5ec19d__complete__1 | 验证 | 完整资料 | Wavves was formed in 2008. | 事实断言；材料支持；符合完整参考 | 0.042 | 保留有依据 |
| 5ae54eea5542993aec5ec19d__complete__2 | 验证 | 完整资料 | Social Code was formed in 1999. | 事实断言；材料支持；符合完整参考 | 0.086 | 保留有依据 |
| 5ae54eea5542993aec5ec19d__complete__3 | 验证 | 完整资料 | Social Code was formed first. | 事实断言；材料支持；符合完整参考 | 0.642 | 保留有依据 |
| 5ae54eea5542993aec5ec19d__partial__1 | 验证 | 缺失资料 | Wavves was formed in 2005. | 事实断言；材料无依据；不符合完整参考 | 1.000 | 检出风险 |
| 5ae54eea5542993aec5ec19d__partial__2 | 验证 | 缺失资料 | Social Code was formed in 1999. | 事实断言；材料支持；符合完整参考 | 0.198 | 保留有依据 |
| 5ae54eea5542993aec5ec19d__partial__3 | 验证 | 缺失资料 | Social Code was formed first. | 事实断言；材料无依据；符合完整参考 | 0.918 | 检出风险 |
