# Lookback 的局部失败诊断与 Round10 改进依据

本分析只使用已结束的 Round7/9 实验和固定原论文，未读取 Round10 新测试检测分数。它描述本地适配的失败及新方法动机，不预设提升幅度、F1 0.7 或 SOTA。

## 原论文已经做了什么

Lookback Lens 使用上下文注意力均值与已生成文字注意力均值的比例，拼接所有层/头，并对片段向量取平均训练逻辑回归；原论文 §2.1 已区分预标注片段和滑动窗口，不能把“细粒度片段监督”本身说成这里的新发明。[EMNLP 2024 原论文](https://aclanthology.org/2024.emnlp-main.84.pdf)

本地保持比例公式，改用 Qwen2.5-7B-Instruct NF4、真实 Search results 区和读入当前 token 后的状态；数据、标注口径及部分聚合方式不同。因此下面的成绩是本地适配结果，不是与原论文公开表格直接可比的全面复现。

## 本地证据：三个问题不能混在一起

**第一，读到资料不等于对应了正确事实。** Round7 外部例 `ragognize_test_2337__partial__1` 问 Jetour Freedom 销售年月，模型将 Leapmotor B10 的交付月份回答成“Sales of the Jetour Freedom commenced in China in April 2025”；Lookback 分数 0.095820，低于当轮阈值 0.746901226，漏报了主体错配。当前资料确有 April 2025，但不是被问车型的销售日期；这个例子证明局部漏检存在，不能单独证明模型通过哪条注意力路径造成错误。[原资料与冻结预测复核](../../round7_evidence_grounding/results/ERROR_CASES_REVIEW.md)

**第二，整项风险与内部错误位置是不同任务。** Round9 全局粗监督的词元 F1 为 0.387，细监督为 0.578；18 份混合正负词元回答的句内宏 AUROC 从 0.566 升到 0.881。整项广播对照在每句话内部的 AUROC 为 0.5，表明整句话可疑不等于定位了其中哪几个词；细监督也在原论文中已有先例。[冻结指标](../../round9_evidence_binding/results/metrics_test.json)、[评测口径](../../round9_evidence_binding/EVALUATION_DRAFT.md)

**第三，正确拒答可能占据复查预算。** Round7 `ragognize_test_0042__partial__1` 正确说明资料未给 First Bus London 初始车队规模，但 Lookback 风险 0.992608 并进入前 20% 复查名单；其事实断言 F1 分母本来排除拒答。Round10 问题级明确审查后的正常拒答计无风险，词元级仍单独描述拒答报警，需公开这个分母变化，不能把它包装成算法提升。[旧案例及分母说明](../../round7_evidence_grounding/results/ERROR_CASES_REVIEW.md)、[Round10 协议](../PLAN.md)

## 为什么不继续补主体与属性正则

Round9 原规则要求同一句同时满足主体关联、属性线索及局部角色规则；完整资料的测试 40 条中仍有 18 条候选为空，局部缺失资料为 29/40，说明“没有候选”常常也只是规则没有识别，而非真实无证据。[冻结覆盖统计](../../round9_evidence_binding/results/runtime_and_candidate_coverage.json)

已亲读以下三个完整资料例及其匹配 trace：

| 输入行标识 | 真正可见的内容 | 原规则失效点 |
|---|---|---|
| `ragognize_train_0158__complete` | Jaylan Pearman 的背景、奖学金合同和“His deal … June 2024” | 将问句中的 `Jaylan Pearman's Perth Glory` 抽成不可分主体，实际来源没有该连续锚 |
| `ragognize_train_1681__complete` | 首句给 2025 Belgian Darts Open，后句“The seedings were confirmed on 7 February” | 主体在前句，属性在后句；标题与抽取主体不完全相等，后句未被承接 |
| `ragognize_test_0275__complete` | FireAid 背景后，organizers 预计收益超过 $100 million | 属性句省略 FireAid，organizers 未被识别为标题主体关联 |

来源为 Round9 `data/inputs.jsonl` 及对应 `data/features/<row_id>.json` 的 `candidate_plan`；没有将另一条件或参考答案当成当前展示证据。

这些是规则门槛的具体失败，不证明替换成 embedding 就能理解关系。Round9 `binding_fine` 相比全局细监督仅从 0.578 到 0.590，主 F1 增量 95% 配对区间为 [-0.027, 0.065]，尚无稳定整体增益；“匹配失败造成该弱增益”目前也只是待检验解释。[独立结果审查](../../round9_evidence_binding/results/INDEPENDENT_FINDINGS.md)

## 固定的针对性改动

Round10 对所有可见句子计算问题内容词的冻结输入 embedding MaxSim 软相关度，再检查当前 token 的注意力和 embedding 匹配落在哪里，不要求实体与属性在同一句中同时词面命中。新增仅 4 层段×4 通道，温度 0.1 固定；另加 8 维纯文本控制，将长度、词面覆盖与白盒信号分开对照。[精确公式与接口](../FEATURES.md)

这是“模型表示和注意力的弱对齐信号”，不是蕴涵模型：问题的静态 embedding 没有读取之后的资料；标题可能误承接其他主体，MaxSim 不理解关系顺序，数字或姓名的 BPE 碎片也可能匹配得很高。全部候选权重非空不等于找到了正确证据，也不能将注意力重叠解释为因果上的证据使用。

新词元特征在当前 token 读入后计算，不看后续文字；用整项均值检测问题级风险则需要等待这一短回答完成。两级训练和评测分别固定，主比较是 `LB+白盒+文本` 对 `LB`，白盒额外贡献比较是它对 `LB+文本`；结果有效、无明显提升或变差均应照实报告。

## 运行脚本只读审查备注

审查对象为实现初版 `src/run10.py`；未修改脚本。已确认开发复用仅取 Round9 train/validation 320 条、逐文件复制哈希，生成包装用固定 unlabelled 占位，提取输入使用可见白名单，生成/数组缓存分别绑定输入行、生成文件和源码签名；未发现将标签或隐藏主体传入新特征的路径。

已向根任务反馈冻结前需核对的三处：stage 签名应包含实际使用的 `attention7.py` 依赖；保存每条缓存前应复验输入/协议冻结清单，而不只在入口验证；新数组校验应锁 16/8 维、float32 和特征名而不只检查行数。此处是审查当时的建议记录，不能代替根任务后续修正、GPU核对及最终冻结检查，也不是已发生实验污染的断言。
