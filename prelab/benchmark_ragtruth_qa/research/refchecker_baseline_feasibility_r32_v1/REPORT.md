# RefChecker 正式基线可行性审计（r32 v1）

审计日期：2026-09-12  
论文：[Knowledge-Centric Hallucination Detection（EMNLP 2024）](https://aclanthology.org/2024.emnlp-main.395/)  
官方源码：[amazon-science/RefChecker](https://github.com/amazon-science/RefChecker/tree/1df1b25cee792ba2b171302e31ca4f768bd67703)，固定为 `v0.2.17` / `1df1b25cee792ba2b171302e31ca4f768bd67703` / tree `9f2e810cc91abaeeceb105cf4b8027eb463580f8`。

## 结论

**完整RefChecker正式基线当前不可运行。** 官方开放的 Mistral claim-triplet 抽取器，未量化权重本身为 **13.489 GiB**，已经超过本机 **8 GiB** 显存；完整运行还需要激活、KV cache 和 vLLM 运行空间。RepC 分支另需 13.489 GiB 的 OpenHermes 主干、约 1.000 GiB 的压缩分类器文件，并存在官方源码的设备与调用接口问题。

NLIChecker 和官方 localizer 各自约 1.33 GiB，缩小 batch 后可以在 8 GiB 显存上单独运行。但没有官方抽取器生成的 claim-triplet，就不是完整 RefChecker。因此本轮只能给出**可执行方案和阻塞结论，不能给 RefChecker 报分**。

本轮只做 CPU 静态审计和数据规模统计：没有运行 GPU、没有下载权重、没有读取封存的 official test，也没有修改正式 baseline。

## RefChecker 原生输出是什么

RefChecker 在 Noisy Context 场景中的流程与本项目很接近：

1. 抽取器读取“问题＋模型回答”，把回答拆成若干 `[subject, predicate, object]` claim-triplet。
2. checker 将每个 triplet 与检索资料比较，输出一个硬标签：`Entailment`、`Neutral` 或 `Contradiction`。
3. 可用 `strict`、`soft` 或 `major` 聚合为整答结果。`strict` 中只要有 Contradiction 就判 Contradiction；否则只要有 Neutral 就判 Neutral；全为 Entailment 才判 Entailment；没有 triplet 时为 Abstain。

所以它的原生最细粒度是**事实三元组级**，不是 token、字符或 4-BPE 窗口级。官方 triplet 对象没有回答中的字符位置；NLI 和 RepC 公开接口最终也只返回硬标签。`NaiveEmbedLocalizer` 是一个额外的粗定位器，返回带颜色的 HTML token 文本，而不是数值位置或风险概率。

这意味着 RefChecker 可以原生回答“哪条事实可疑”，但要参加本项目的统一窗口评测，必须先把官方`NaiveEmbedLocalizer`作为冻结的作者定位组件运行，再做无参数坐标映射。localizer带sup-SimCSE权重和官方阈值，本身不是无参数适配器。

## 接入全部 3,839 条回答的冻结方式

RefChecker是无需本项目标签拟合的推理型基线，必须覆盖fit 3,680条和calibration 159条，共3,839条回答；每条都有3篇检索资料，共11,517篇。既定4原始BPE、stride 1几何共有696,220个合格窗口：fit 653,979个、cal 42,241个。另有772个候选窗口因对应字符不含任何字母数字字符而固定排除（fit 692、cal 80）；这属于所有方法共享的评测几何，不是RefChecker自行丢弃样本。当前主开发计分仍只报告cal159。

每条样本按下面方式送入官方流程：

- `question` 送给官方 Mistral-SFT triplet 抽取器。
- `original_response` 作为待检测回答。
- 将现有 `Passage 1/2/3` 的正文按顺序解析为 3 个独立 reference，再按官方 Noisy Context driver 的格式加上 `Passage 0/1/2:` 前缀。
- 抽取器固定为 `dongyru/Mistral-7B-Claim-Extractor@fbc414e...`，`claim_format=triplet`，`temperature=1e-5`，`max_new_tokens=1000`。
- NLI 分支固定官方 NLI 模型，调用公开接口时显式设 `is_joint=False`、`max_reference_segment_length=200`、`merge_psg=True`。`CheckerBase` 已明确 joint 只供 LLM checker；设为 false 是使用官方选项，不改模型。
- RepC 分支保留官方 OpenHermes 主干和默认 `nn_ensemble`，不替换其结构或参数。

RAGTruth 人工幻觉 span **只能作为最后评分的金标**。它们不能替代抽取器产出的 claim，也不能输入 checker 或 localizer。否则测到的是“给定正确错误位置后的核查能力”，不是 RefChecker 的端到端能力。

## 冻结作者定位组件与无参数4-BPE映射

统一映射规则现由 v3 冻结；v2 仅保留历史身份：

1. `Entailment → risk 0`；`Neutral/Contradiction → risk 1`。这是论文的严格事实性定义，不训练阈值。
2. **冻结作者方法层：**对每个 risk=1 的 triplet，原样调用官方 `NaiveEmbedLocalizer.locate(response, triplet)`；模型固定为`princeton-nlp/sup-simcse-roberta-large@96d164d9950b72f4ce179cb1eb3414de0910953f`，阈值固定为`[0.65, 0.60, 0.65]`。该步骤有学习权重，不计入零参数适配器。
3. **无参数统一适配从这里开始：**解析冻结localizer返回的有序 HTML span。红、蓝、绿 token 是选中的 subject、predicate、object；黑色 token 不计入。
4. 将 HTML 解码后的可见 token 流与原回答都做 Unicode NFC、转小写、连续空白归一为一个空格；随后做单位代价的全局 Levenshtein 对齐，平局顺序固定为“对角、消耗 localizer 字符、消耗回答字符”。把选中 token 中精确匹配的非空白字符投回原回答，合并相邻字符为区间。出现多个官方选中位置时全部保留。
5. 一个 4-BPE 窗口的冻结字符区间只要与任一 risk triplet 的定位区间有非空交集，就输出 risk=1；否则为 0。多个 triplet 取并集。
6. 只要某个 risk triplet 的 HTML 无法完整解析、选中 token 没有任何非空白字符能精确投回原回答，或投回字符没有覆盖任何合格窗口，就把该 triplet 的 risk=1 **广播到整条回答的全部合格窗口**，并记录 `localization_status=whole_answer_fallback`。这会如实暴露定位失败，而不会悄悄丢掉错误。
7. 如果抽取器没有生成 triplet，则保留原生 `Abstain`，统一窗口输出全 0，并单独报告 abstain 数量和比例。
8. 统一整答分数严格取该回答全部合格 4-BPE 窗口的 max。官方 `strict` 聚合只另列原生复核，不替代统一回答级聚合。

步骤3–8只是模型外的确定性坐标转换，没有训练、校准或附加分类器；步骤2是冻结的作者学习型组件。人工 span 只在映射完成后计算窗口 F1、整答 F1、AUROC 和 AP。只要有 risk triplet，成功定位会令至少一个合格窗口为1，无法落到合格窗口则触发整答广播；没有 risk triplet 时全部窗口为0。因此统一 `answer=max(window)` 与另列的 `strict` 风险值在冻结规则下必然相等。该等价性是规则证明，不是运行结果。

早期`FEASIBILITY_CHECKLIST.json`把步骤2也放在`unified_evaluation_adapter`对象内，层级命名不严谨；该清单现已指向v3并把输入边界移到冻结HTML之后。`FORMAL_BOUNDARY_V2.json`仍原样保留其“直接用strict作整答结果”的历史身份；当前机器可读边界见[FORMAL_BOUNDARY_V3.json](./FORMAL_BOUNDARY_V3.json)。

## 8 GiB 显存与计算规模

| 组件 | 固定模型 | 权重/下载体积 | 8 GiB 判断 |
|---|---|---:|---|
| triplet 抽取器 | Mistral-7B-Claim-Extractor | 13.489 GiB | 不可按官方未量化方式运行；还未计运行空间 |
| NLIChecker | RoBERTa-large NLI | 1.328 GiB | 可单独小 batch 运行 |
| Localizer | sup-SimCSE RoBERTa-large | 1.324 GiB | 可单独小 batch 运行，但调用量很大 |
| RepC 主干 | OpenHermes-2.5-Mistral-7B | 13.489 GiB | 不可运行 |
| RepC 默认分类器 | `zthang/repe` nn ensemble | 1.000 GiB 压缩包 | 与主干合计至少 14.489 GiB 下载量，尚未计解压和运行空间 |

官方 README 给 Mistral 服务示例使用 vLLM `--tensor-parallel-size 8`。这不等于必须有 8 张卡，但进一步说明官方路径并非面向 8 GiB 单卡。若要忠实运行，保守方案是提供一张 24 GiB 级显卡，或采用官方文档的多卡 vLLM 路径；不能用 4-bit、GGUF 或别的抽取器冒充正式 RefChecker。

实际 claim 数必须在官方抽取器跑完后才知道。仅用于容量规划：按论文 Noisy Context 平均4.9个claim/回答估算，全3,839答约有18,811个claim、至少56,433个claim-reference pair和约75,244次localizer前向；若把旧的长回答线性上界按当前全量范围缩放，则约82,488个claim、249,855个pair和329,946次localizer前向。这些都只是容量估算，不是实验结果。

## 源码层面的忠实性问题

这些问题都需要保留或明确披露，不能暗改后仍称“官方原样 baseline”：

- **NLI 的 question 实际未使用。** 发布代码只把 reference 作为 premise、claim 作为 hypothesis。正式运行应保留这一行为，不能自行把问题拼入 premise。
- **跨资料合并时 Entailment 优先。** 任一分段给出 Entailment 就覆盖其他分段的 Contradiction；没有 Entailment 才看 Contradiction。这会漏掉“不同来源互相冲突”的情况，但属于官方规则。
- **官方 benchmark driver 没有显式关闭 joint。** 非 LLM checker 应通过公开参数 `is_joint=False` 调用；否则 NLI 接收到嵌套 claim 列表，不能忠实完成逐 claim 检查。
- **RepC 公开路径不能原样执行。** 主干被硬编码到 `cuda:1`，输入默认送到 `cuda:0`；`CheckerBase.check` 还会传入 RepC `_check` 不接受的关键字。在非 joint 展平中，base 又把整组 questions 列表传给每个 pair。单张 8 GiB 卡既不满足设备要求，也不满足显存要求。
- **Localizer 很粗。** 它使用 `hidden_states[0]`，并把最后一个回答分段的 token 数误作三个 triplet 元素的搜索长度，最终只返回 HTML。正式 baseline 应保留它；若修复，只能另列“RefChecker 适配版”。

RefChecker 确实是 source-conditioned：它直接把每条抽取事实与检索资料比较，因而能针对“回答有没有依据”以及 Evident Conflict 类问题。但官方的 Entailment 优先合并规则可能掩盖多来源冲突，这应作为之后误差分析的一项，而不是先修改 baseline。

## 当前冻结决定

- 正式候选保留两条：`RefChecker-Mistral-SFT+NLI` 和 `RefChecker-Mistral-SFT+RepC`。
- 当前两条均为 `blocked`，没有正式分数。
- NLI 和 localizer 的“可单独运行”不等于完整 baseline 已运行。
- 不使用 ModernBERT、人工 span、替代 LLM 或量化 Mistral 生成 claim。
- 如以后为了修复 RepC 的设备/参数分发问题而改源码，结果必须标为“执行适配版”，并同时保留官方原样不可运行的审计记录。

## 可复核材料

- CPU 数据/源码审计：[CPU_AUDIT.json](./CPU_AUDIT.json) 与 [audit_feasibility_cpu.py](./audit_feasibility_cpu.py)
- 固定模型 revision 与文件体积：[MODEL_METADATA_SNAPSHOT.json](./MODEL_METADATA_SNAPSHOT.json)
- 原可行性记录：[FEASIBILITY_CHECKLIST.json](./FEASIBILITY_CHECKLIST.json)；当前方法/适配边界：[FORMAL_BOUNDARY_V3.json](./FORMAL_BOUNDARY_V3.json)（v2 仅保留历史）
- 官方实现证据：[extractor](https://github.com/amazon-science/RefChecker/blob/1df1b25cee792ba2b171302e31ca4f768bd67703/refchecker/extractor/llm_extractor.py)、[checker base](https://github.com/amazon-science/RefChecker/blob/1df1b25cee792ba2b171302e31ca4f768bd67703/refchecker/checker/checker_base.py)、[NLI](https://github.com/amazon-science/RefChecker/blob/1df1b25cee792ba2b171302e31ca4f768bd67703/refchecker/checker/nli_checker.py)、[RepC](https://github.com/amazon-science/RefChecker/blob/1df1b25cee792ba2b171302e31ca4f768bd67703/refchecker/checker/repc/repc_checker.py)、[localizer](https://github.com/amazon-science/RefChecker/blob/1df1b25cee792ba2b171302e31ca4f768bd67703/refchecker/localizer/embed_localizer.py)、[Noisy Context driver](https://github.com/amazon-science/RefChecker/blob/1df1b25cee792ba2b171302e31ca4f768bd67703/benchmark/evaluation/autocheck.py)
