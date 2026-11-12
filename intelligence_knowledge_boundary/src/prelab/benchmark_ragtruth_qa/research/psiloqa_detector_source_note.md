# PsiloQA：辅助数据使用前的训练源码核查

仅阅读官方论文、源码和模型元数据；未运行作者脚本、下载模型权重或打开数据测试集。PsiloQA锁定仓库提交 `95233aefcfbdfc2b0876ffc7d45729d7fc69bd2d`。它发表于 **Findings of EMNLP 2025**，不是EMNLP主会。[正式论文](https://aclanthology.org/2025.findings-emnlp.626/)

**1. 公开训练入口没有传资料，也没有传标准答案。** `train_script.py::build_samples`只将 `question` 赋给 `prompt`、`llm_answer`赋给 `answer`；未读取 `wiki_passage`、`golden_answer`或 `annotated_span`。其默认初始化是通用 `answerdotai/ModernBERT-base` 加两类token分类头；读取完整发布train和validation，不按语言过滤、不混入RAGTruth。默认seed123、batch4、6轮、lr1e-5。论文§4.3.2却说输入context/question/answer三元组，附录batch8；因此当前入口不能直接当作论文实验的完整可复现配方。[锁定入口](https://github.com/s-nlp/PsiloQA/blob/95233aefcfbdfc2b0876ffc7d45729d7fc69bd2d/train_script.py#L59)

**2. 作者入口直接消费 `labels` 列，不会用HAL标记纠正它。** 每个 `[start,end]`直接转为字符区间；缺失或空列表成为无正标注样本。因此，根任务观察到的“HAL标后一个1975、列坐标落前一个”和“有HAL但labels为空”，若输入此入口，就会直接进入其训练标签。该判断是源码条件推论；本轮没有复读这些数据行。[build_samples](https://github.com/s-nlp/PsiloQA/blob/95233aefcfbdfc2b0876ffc7d45729d7fc69bd2d/train_script.py#L59)

**3. 仓库确实存在会错定位重复片段的另一路解析。** `row_to_psiloqa_record`先抽取HAL内部文字，再用 `llm_answer.find(chunk,cursor)`，失败时从头 `find(chunk)`；它没有利用标记前的未标注文字。首次标注一个重复年份时，可命中原回答更早的同值。这个函数供UQ评估使用，并构造含问题、wiki资料及“无法回答”提示的prompt；不能把它的prompt当成上述训练入口的输入，也未找到它生成HF发布 `labels` 列的导出调用链。因此只能确认同类错误机制存在，不能宣称发布错误的来源已完全证明。[解析代码](https://github.com/s-nlp/PsiloQA/blob/95233aefcfbdfc2b0876ffc7d45729d7fc69bd2d/psilo/methods/uncertainty/evaluation.py#L12)

人工一致性分析notebook的 `parse_annotation_text`则用正则匹配的原位置减去累计标签长度，未按片段内容回搜。两条公开坐标路径不一致；本轮仅读取notebook代码单元，没有读取其输出或CSV。[notebook](https://github.com/s-nlp/PsiloQA/blob/95233aefcfbdfc2b0876ffc7d45729d7fc69bd2d/evaluation/consistency_tests.ipynb)

**4. Loss落在检测器自己的token上，与我们的原生成器4BPE口径不同。** 当前官方LettuceDetect依赖将 `(sample.prompt, sample.answer)`作为tokenizer双序列，`only_first`截断第一段、上限8192；prompt及批次padding标成-100，回答token按与正区间相交与否赋1/0。源码循环还包含末尾SEP，通常赋0；没有我们的lexical遮罩。Trainer将labels交给HF默认token分类loss，没有显式类别、材料组或回答等权；直接AdamW，按验证集token argmax分类F1选最优轮。故不能直接对照我们的4原始BPE窗口F1/整答max双指标。[编码与标签](https://github.com/KRLabsOrg/LettuceDetect/blob/2096ed28f3b662a62b4406795da3d6fbc5490063/lettucedetect/datasets/hallucination_dataset.py#L136) · [Trainer](https://github.com/KRLabsOrg/LettuceDetect/blob/2096ed28f3b662a62b4406795da3d6fbc5490063/lettucedetect/models/trainer.py#L52) · [验证口径](https://github.com/KRLabsOrg/LettuceDetect/blob/2096ed28f3b662a62b4406795da3d6fbc5490063/lettucedetect/models/evaluator.py#L14)

**依赖版本限制：** 上段核的是LettuceDetect提交 `2096ed28f3b662a62b4406795da3d6fbc5490063`。PsiloQA的pyproject和uv.lock没有列出/锁定lettucedetect，不能断言当前依赖的所有边界处理与论文当年一致。[依赖声明](https://github.com/s-nlp/PsiloQA/blob/95233aefcfbdfc2b0876ffc7d45729d7fc69bd2d/pyproject.toml)

**5. 标准答案用于产生银标；RAGTruth混训变体确实出现在论文中。** 标注器读取passage、question、gold_answer、hypothesis，调用GPT判断并插HAL；这不是训练检测器输入标准答案的证据。论文§5.3用mmBERT分别训练RAGTruthQA、PsiloQA英文、两者合并；论文主要编码器实验还包含多语言PsiloQA训练。因此，“PsiloQA模型”这个泛称不足以判定它是否见过RAGTruth。[标注器](https://github.com/s-nlp/PsiloQA/blob/95233aefcfbdfc2b0876ffc7d45729d7fc69bd2d/psilo/dataset/annotator.py#L24) · [论文§5.3/表4](https://aclanthology.org/2025.findings-emnlp.626.pdf#page=8)

**发布权重仍有来源缺口。** 本次官方s-nlp公开列表找到 `modernbert-base-psiloqa`（revision `d81599e6894dd6bf5e571e3fae0cdea545dacbdc`）及 `modernbert-large-psiloqa`（`7beffcecd4d5bd6839fb6b95d6e7600ef6856a94`）；两者均无README/model card，文件列表未给出训练数据配方。没有核实到与论文RAGTruthQA-only/Both明确对应的发布检查点，亦不能仅凭这两个名称保证没有RAGTruth训练。若使用过官方完整RAGTruth train，会触及我们从中拆出的cal来源，不能直接作为该cal上的公平新基线。[base元数据](https://huggingface.co/api/models/s-nlp/modernbert-base-psiloqa) · [large元数据](https://huggingface.co/api/models/s-nlp/modernbert-large-psiloqa)

对当前辅助数据工作的直接结论：有用的是其原wiki资料、问题、自然生成回答和可核对的银标位置；不能盲信发布labels与HAL一致，也不能把“问题＋回答”的公开训练入口照搬后声称做了资料条件核查。上述发现没有修改任何辅助数据、标签或训练计划。
