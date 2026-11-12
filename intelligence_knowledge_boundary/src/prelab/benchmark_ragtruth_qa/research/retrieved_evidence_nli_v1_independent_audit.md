# Retrieved-evidence NLI v1 独立静态审计

审计对象：`src/run_retrieved_evidence_nli_v1.py`，SHA256 `353129f81e194b87542ec60aa74a6db012cb437108384805ceaa1dd13d1bd64a`。本次只读代码、开发侧 label-free 产物和本地 NLI 配置；未启动 GPU、未载入模型、未打开官方 test。

## 结论

**没有发现阻断 GPU 提取的实现错误。** 数据预处理没有让标签值参与 claim 划分、BM25 检索、NLI 配对或特征计算；NLI 类别与前提/假设顺序正确；逻辑回归、标准化和阈值都只使用 fit；当前冻结数据上的 4-BPE 投影与统一评测等价。

## 逐项核对

- **标签流**：`prepare` 经 `feature_qa.development_rows()` 只返回问题、三份资料、回答、分区与组等白名单字段。自动 claim 来自既有 NLTK 句子切分产物，其生成代码没有读取 hallucination span。`score` 到全部 NLI cache 冻结后才调用 `q.metadata()` 打开开发金标。没有发现标签进入 BM25、NLI 文本或特征列。
- **claim 与证据**：每个 claim 在 passage 1/2/3 内分别取 BM25 top-2；pair 顺序和 `(claim, passage, rank, sentence)` owner 一一绑定，缓存校验会逐项复核。现有 label-free 产物为 793 答、8,845 个有效 claim、51,953 对 NLI 输入；句子坐标、claim 坐标和哈希都有闭环。
- **NLI 语义**：代码用 `tokenizer(premise, hypothesis)`；本地模型卡规定第一段为 premise、`text_pair` 为 hypothesis。代码按列 0/1/2 读取 E/N/C，且载模时强制断言 `id2label={0: entailment, 1: neutral, 2: contradiction}`；本地 `config.json` 完全一致。
- **训练与阈值**：五折只切 `fit_claim_indices`，并按 source-connected `group_id` 隔离；每折 scaler 和 LR 只拟合该折训练组。最终 scaler/LR 只拟合全部 fit，再预测 calibration。窗口和整答阈值均由 fit 决定；学习分数使用 fit OOF。calibration 只在最后 `q.metrics` 中报告，没有进入训练或阈值。
- **4-BPE 投影**：claim 分数先给其 lexical BPE，窗口取其 4 个 raw BPE 中 lexical claim 分数最大值，整答再取全部有效窗口最大值；窗口宽 4、步长 1、标点占宽的口径正确。独立遍历当前 174,518 个 lexical BPE，跨多个 claim 的词元为 0；重建得到 210,364 个有效窗口。因此脚本的“单 owner”写法在当前冻结数据上与“所有相交 claim 取最大值”完全等价。

## 非阻断风险

1. “label-free 阶段从未访问 annotation”这一表述过强：`development_rows()` 会反序列化含标签的开发 JSONL 后再取白名单，`source_files()` 也会哈希整份含标签文件。没有标签依赖或统计泄漏，但若论文要主张物理隔离，应另存纯文本输入，或把表述改成“标签未被引用、传递或用于决策”。
2. 6,419/51,953（12.4%）个入选证据句与 claim 的 BM25 词项交集为 0，此时规则固定取每个 passage 最前两句。这是已披露的检索噪声，会限制效果，但不是接线错误。
3. claim 实际是句级片段；一句中只要一个 lexical BPE 为风险，整句 claim 就是正类并广播全句。它会牺牲边界精度，结果应称“句级候选映射到 4-BPE”，不能称原生 token 探针。
4. `raw_or_risk=max(maxC,1-maxE)` 会把“某来源支持、另一来源冲突”也判高风险；它只能作为固定诊断分数。主要候选应看完整 E/N/C 与覆盖特征训练出的 `claim_lr`，并另报冲突案例。
5. 单 owner 投影没有一般性处理一个 lexical BPE 同时跨两个 claim 的情形。当前数据中该情形为 0，所以不阻断本轮；换 claim 切分器时应改为所有相交 claim 取最大值或加唯一性断言。

此外，cal159 在整个项目中已被多轮查看。此脚本没有使用 cal 选模型或阈值，但本轮 cal 成绩仍只能称开发结果，不能当独立泛化证据。
