# 当前基线与候选对照

> 口径修正（2026-09-12）：本文件含大量历史“匹配融合对照”，它们不是论文原样baseline。正式基线按[BASELINE_PROTOCOL.md](./BASELINE_PROTOCOL.md)单列：冻结作者模型/公式，主比较统一使用本项目4-BPE窗口与整答评测；作者原生尺度另作复核。作者只支持整答时，将其冻结整答分数原值广播到本项目窗口后评分，并明确标注“原生无定位输出”。不会故意削弱基线，也不会给基线追加我们的结构。

当前统一开发评测的正式对照如下。F1越高越好；基线低于本文候选是实际运行结果，不是实现目标。

| 方法 | 统一4-BPE窗口F1 | 统一整答F1 |
|---|---:|---:|
| Lookback Lens冻结结构迁移 | 0.600882 | 0.845455 |
| LUMINA固定作者公式迁移 | 0.331299 | 0.785047 |
| GHOST论文结构重实现＋零参数整答广播 | 0.320361 | 0.772201 |
| 当前本文候选 | **0.690281** | **0.891089** |

Lookback与LUMINA的**统一评测适配器本身**均未训练参数，也未改变冻结基线分数；独立复算通过。LUMINA正式v2还把无标签映射与gold评分物理分开，逐值复放旧结果。Lookback默认逻辑回归在适配前仍按正式基线流程用fit数据训练过，不能把这句话理解成Lookback基线从未训练。完整口径与作者原生复核见[正式结果表](./FORMAL_BASELINE_RESULTS.md)。这些仍是反复使用的cal159开发成绩，不是封存测试结论。

> Lookback专项审核：现有 `results/lookback_regularization_v2`、`fit_expansion/llama_baselines_v1` 中Lookback族均为本地适配；Lookback＋tail2／large／NLI／FAVA及交叉拟合组合属于自研消融。下文这些历史分数与模型名保留，均不能作为原版Lookback Lens结果。正式论文结构迁移已completed：固定8原生BPE、1024维跨度均值、默认LR（仅max_iter=1000），无缩放、权重或调参；材料组隔离、fit-only阈值及整答max另列为数据/评测适配。

正式Lookback的prepare、check、fit均实际exit0；334次迭代，无收敛警告。以下仅比较同一cal159答、41685个完整8-BPE窗及同标签，不与旧4-BPE窗口F1直接排位：

| 方法身份 | 8-BPE span AUROC（主指标） | AP | 额外8窗F1 | 阈值来源 |
|---|---:|---:|---:|---|
| Lookback Lens论文结构迁移，completed | 0.855601 | 0.624460 | 0.587706 | 仅fit确定 |
| 当前自研候选的8窗评测适配v2 | 0.911669 | 0.731856 | 0.693463 | 固定此前cal选中的旧阈值 |

正式Lookback的额外整答max F1为0.817734，阈值同样仅fit确定。当前候选v2对已有4窗分数聚合的规则在v1覆盖失败后确定，候选/阈值也已使用cal；F1差异不能作无偏胜出证据，AUROC/AP仍只属反复开发集对照。v2不写入基线结构，官方test继续封存。见 [正式Lookback报告](./results/lookback_official_span_v1/REPORT.md)、[自研同尺度适配](./results/current_on_lookback8_v2/REPORT.md)。

正式GHOST论文结构迁移已实际完成：四维全答均值＋论文最终750树RF，整答F1 **0.772201**、AUROC 0.588983。按本项目评测要求，把冻结整答分数原值广播到该答全部4-BPE窗口后，统一窗口F1为 **0.320361**；独立复算通过，没有修改RF或重新训练。整答最优开发阈值把159答全部报风险，所以不能把该F1单独解释成有效区分。详见 results/ghost_official_answer_rf_v1/REPORT.md 与 results/ghost_k4_evaluation_adapter_v1/REPORT.md。

2026-09-12补充：固定NLL+GHOST浅树只用原native634答训练，为0.333022/0.796537；对应expanded训练为0.331200/0.796610。同五维/100轮/原cal159/4BPE规则，保留native权重比例并统一总质量，没有重算类平衡。一次数据消融实际exit0，无实质收益、未替换当前候选；不能唯一归因跨生成器重放。详见 results/ghost_native_only_v1/REPORT.md。

截至 2026-09-12 10:38（本地记录时间），当前同一候选 **semantic_claim＋tail2树＋large权重0.4** 为 **窗口0.690281／整答0.891089**。增强HARP＋FAVA为0.687696／0.870466，同预算Lookback＋large LR为0.679293／0.873096、树为0.677619／0.879227。large/NLI/FAVA共18个已选组合对照已汇总；当前候选两项点估计均不低于其中Lookback/HARP对照，但既有配对区间跨0且未计重复选型偏差，尚未证明稳健优势。定位目标0.75仍未达到。

PsiloQA后续比较已实际CPU准备完成：六个原底座各六档同权重，共36项；等待同一个完整三轮QA训练选中结果后才能计分。不会把large/NLI/FAVA融合输出继续当底座。当前没有PsiloQA新成绩。防过拟合三折中Lookback全部完成、large第0折完整结束、第1折运行；GHOST提取与5个LR均已完成。新增10036对局部银标训练入口已准备，两个固定模式共享全部数据/模型/顺序/预算，仅排序项权重0或1不同；GPU训练未开始。详见 results/psiloqa_fixed_convex_v1/preparation_complete.json、results/local_pair_transfer_v1/REPORT.md。

最新HARP对照的固定阈值配对检查已actualexit0：相对HARP＋FAVA，窗口差+0.002586，154材料组5000次抽样的95%条件区间[-0.024476, +0.026243]；整答差+0.020623，区间[-0.023000, +0.066670]。对HARP＋NLI的两项区间也跨0。这些区间还未包括反复选型偏差，不能据此证明稳定胜出；没有新拟合或重选阈值。见 results/current_harp_group_intervals_v1/REPORT.md。

2026-09-12 10:38：GHOST全3839答、696220×4窗口矩阵、5-LR与固定诊断均完整actualexit0。HARP＋GHOST为0.598131／0.867580，原同C对照0.596677／0.870370；窗口仅增0.001454、整答降0.002790，显性冲突检出177→173/997。单GHOST四维0.300257／0.782609。四组增量的两项资料组条件区间均覆盖0，没有更新当前最佳模型。完整表见 results/ghost_matched_lr_v1/RUN_REPORT.md；它是局部LR适配，不是原论文RF整答复现。

LUMINA公共QA原/随机整段资料全3839答、708506回答BPE已actual exit0。正式gold隔离v2逐值复放后，固定作者公式统一4-BPE/整答F1仍为0.331299/0.785047；论文式全答raw均值复核为AUROC 0.743390、AUPRC 0.838979、F1Opt 0.791489。原公式/IPR/负MMD分别保存，没有训练、调权或融合。见 results/lumina_k4_evaluation_adapter_v2、results/lumina_qa_scoring_v1、results/lumina_official_answer_v1。

2026-09-12 10:49补充：三个同预算浅树已actual exit0。NLL单维0.291465/0.772201，GHOST四维0.330848/0.786611，合并五维0.331200/0.796610。该固定非线性组合没有实质定位收益，当前最佳不变。NLL整答为全部报风险。见 results/ghost_nll_nonlinear_v1/RUN_REPORT.md。

以下均为同一 **159 条校准回答、154 个资料组、42,241 个四原始 BPE 窗口**的开发成绩，正常回答和拒答均保留。整答取全部可评窗口最大分数，两级分别选阈值；窗口 F1 不是逐词元精确定位 F1。此校准集已反复用于选型，封存测试未打开。训练规模和信号来源不同，表格不是纯算法排名。

## 方法身份与训练规模

| 方法身份 | 实际输入与适配 | 本项目训练回答 | 窗口 / 整答 F1 |
|---|---|---:|---:|
| Lookback 本地适配 `lb_prefix_pre_header` | 完整前缀、预读取、真实聊天头；4BPE的1024维均值，加权标准化及加权LR、五档C。不是原版基线 | 634 | 0.587831 / 0.858447 |
| `HARP_claim` | 生成模型 LB + NLL + 固定输出头底部64方向投影，TCN 后做同陈述风险传播；只借用 HARP 投影，不是完整 HARP 方法 | 634 | 0.638995 / 0.854271 |
| MiniCheck 官方分数适配 | **额外语义核查器**读取资料和回答；固定分句/分块，支持分数映射到原窗口，校准集选阈值；无本项目梯度训练 | 0 | 0.5622 / 0.8103 |
| `semantic_claim` | MiniCheck 核查隐状态 PCA64、句风险，与生成模型 LB/NLL 做 TCN 融合，再做同陈述传播；**含额外核查器** | 634 | 0.650293 / 0.860215 |
| MiniCheck `frozen0` | 核查编码器冻结，只训练词元风险头；全部资料块映射后聚合 | 3680 | 0.598540 / 0.819512 |
| MiniCheck `tail2` | 相同数据、映射及3轮预算，训练核查器最后两层和同一风险头；固定选第2轮 | 3680 | 0.650293 / 0.838384 |
| 扩充Lookback本地适配，前缀/预读取 | 加权标准化、加权LR及五档C；统一Llama回放，不是原版基线 | 3680 | 0.592998 / 0.852018 |
| 扩充LB＋NLL自研对照 | 同五档C，保留source/post旧定义并追加NLL，不是原版Lookback | 3680 | 0.594668 / 0.853333 |
| 扩充HARP64＋LB/NLL线性 | 固定HARP64方向，同五档C | 3680 | 0.596677 / 0.870370 |
| 扩充HARP序列基线 | 固定30轮，13800更新，选第5轮；同旧模型轮数，非同计算量 | 3680 | 0.617962 / 0.869159 |
| ModernBERT 完整上下文 | 额外编码器读取全部资料/问题/回答，固定六轮，选第3轮 | 3680 | 0.641796 / 0.829493 |
| ModernBERT NLI初始化 | 相同六轮训练，固定选第5轮；仅替换编码器初始化，上游独立性有限可核实 | 3680 | 0.638658 / 0.847291 |
| ModernBERT 人工辅助迁移 | 新9678人工答一轮后原QA三轮，选QA第2轮；不同任务及更多输入词元 | 9678辅助＋3680 QA | 0.648763 / 0.832432 |
| 冻结 ModernBERT＋语义小头 | 固定上游模型，52944参数，同六轮，选第4轮 | 3680 | 0.646541 / 0.831050 |
| 同小头＋LB/NLL | 同样结构与预算，生成信号实际输入，选第6轮 | 3680 | 0.646979 / 0.820276 |

前两项不调用额外语义模型，但仍是本地固定 Llama 回放所得白盒信号，不等于持有数据发布时的历史原生轨迹；TCN 与同陈述传播也是离线处理。3680 答由原634加同来源的3046条其他生成器人工回答组成，仍只有615个训练资料组；不是新增3680个独立问题。其他生成器回答的 Llama 回放尤其不能称为其原生内部状态。

来源：[Lookback](D:/Projects/Multi_Agent_Graph_Analysis/prelab/benchmark_ragtruth_qa/results/lookback_regularization_v2/REPORT.md)、[HARP/semantic 陈述传播](D:/Projects/Multi_Agent_Graph_Analysis/prelab/benchmark_ragtruth_qa/results/claim_pooling_v1/REPORT.md)、[MiniCheck 分数](D:/Projects/Multi_Agent_Graph_Analysis/prelab/benchmark_ragtruth_qa/semantic_baseline/cuda_variant/results/REPORT.md)、[冻结头与尾层微调](D:/Projects/Multi_Agent_Graph_Analysis/prelab/benchmark_ragtruth_qa/results/minicheck_tail_all_docs_v3/completed_diagnostics/REPORT.md)。

## 旧模型共同特征对照（不作为论文原样基线）

每格为“窗口 / 整答 F1”。三个对象都加入**同一个已固定的 tail2**；未只强化 HARP。

| 另一底层对象 | 固定权重组合：同5档权重 | 单调树：同深度2、100轮 | 两分数 LR：同3档 C | 两分数+8维：同3档 C |
|---|---:|---:|---:|---:|
| Lookback | 0.656221 / 0.860104 | 0.651954 / **0.878505** | 0.650364 / 0.866359 | 0.647496 / 0.867580 |
| HARP_claim | 0.674405 / 0.859903 | 0.677223 / 0.852632 | 0.678266 / 0.862745 | **0.679660 / 0.868687** |
| semantic_claim | 0.667370 / 0.858696 | 0.660850 / 0.861702 | 0.667853 / 0.861878 | 0.661283 / 0.858639 |

固定权重只在校准集选五档之一，不重训底层；树和 LR 用原634答拟合组合头。三者都使用训练过3680答的 tail2，因此这些组合属于**额外语义核查融合**，不是纯生成白盒方法。组合头的训练输入是底层训练内分数，没有交叉拟合，仍可能过拟合。

引用特征的第一列在无引用窗口也提供普通资料词面覆盖，故最后两列不是“仅引用编号”单因素比较。HARP 加8维只增窗口 **0.001394**、整答 **0.005942**；Lookback 与 semantic 的窗口反而下降。全部结果保留，不能将小幅开发提升写成稳定胜出。

来源：[固定权重15候选](D:/Projects/Multi_Agent_Graph_Analysis/prelab/benchmark_ragtruth_qa/results/completed_score_fusion_v1/REPORT.md)、[3个单调树](D:/Projects/Multi_Agent_Graph_Analysis/prelab/benchmark_ragtruth_qa/results/completed_score_combiner_v1/REPORT.md)、[匹配18个 LR](D:/Projects/Multi_Agent_Graph_Analysis/prelab/benchmark_ragtruth_qa/results/citation_alignment_lr_v1/REPORT.md)。保存分数与模型已有独立复核，本汇总未重跑审计。

## 新完成来源语义匹配对照

三对象各使用同一批6252个独立来源—陈述推理，固定三档C；两边共同有两底层分数、原8词面特征和2个来源无关语义特征，仅候选再加2个被引来源支持差特征。全部18次拟合均已完成。

| 底层对象 | 共有语义信息的控制：窗口 / 整答 | 加入来源对应差：窗口 / 整答 |
|---|---:|---:|
| Lookback | 0.646873 / 0.867925 | 0.646272 / 0.871287 |
| HARP_claim | 0.679646 / 0.870813 | 0.679828 / 0.868687 |
| semantic_claim | 0.660664 / 0.858639 | 0.660241 / 0.858639 |

没有实质定位收益。HARP相对旧8维组合只高0.000168，整答不变；增加推理成本后仍未两项超过全部强基线。完整报告：results/cited_source_semantic_lr_v1/REPORT.md。

## 尚未完成，不计入成绩

| 计划 | 状态快照 | 将比较什么 |
|---|---|---|
| FAVA合成辅助迁移 | 已完整exit0；选QA2，0.661103/0.833333；36匹配组合亦完成 | 固定合成一轮→原QA三轮，额外数据及训练日程对照 |

状态更新至2026-09-12 07:52本地时间。large六轮、12个加large拟合组合、标题来源18LR及36固定凸组合均已actual exit0；未完成项不计入成绩。旧小头融合相对零生成输入控制只增窗口0.000438、整答下降0.010774，不能把最新多模型融合收益归于那个小头。

原QA选型模型迁移到原R16训练开发数据，原五折仅校准阈值后为0.348525/0.539589；人工辅助版本为0.379225/0.600551，低于原场景本地方法0.709014/0.800000。不是同训练流程对照，也不是独立测试。R16使用助手标注human_gold=false；本轮不读其旧val/test，但历史已有暴露记录，不宣称从未打开。

补充：全答报风险的无模型基线整答 F1 已有0.7722，因此仅整答超过0.75不足以说明可靠。当前窗口目标0.75仍未达到；后续用同一候选的两项成绩比较，不能预先保证新方法胜出。

回答长度/绝对窗口末端两个固定朴素分数也已完成：窗口F1分别0.294055/0.277985，整答均0.772201且全部判风险。它们无神经训练，只在同cal选阈值，不理解事实。详见results/length_position_naive_v1/REPORT.md。

原R16五折适配已完成（与公共QA分开）：单独语义0.429950/0.590000；Lookback+语义0.614789/0.692810；原R26+语义0.710713/0.808511，原R26为0.709014/0.800000。小幅开发收益不代表稳定胜出，标签仍human_gold=false。

公共QA新增整答校正36候选已完成：三对象同六权重、各含自身整答控制及共同Lookback树整答输入；全部仍选alpha0，未采用。保持原窗口、回答max及两个阈值定义，报告results/answer_conditioned_scores_v1/REPORT.md。

新增NLI初始化对照仅完成CPU准备，未GPU/训练；完整资料head和二分类初值与原base exact，仅换编码器且2048长度cap，现输入最大979。入口src/run_full_context_nli_initialization_v2.py，前版构造接口失败保留，不计为训练结果。保守标题scope补充对照已完成；当前候选0.679609/0.868687，未超过原候选。全三对象均同三C，旧控制原模型回放通过，没有降低基线。

R29原场景完整768维状态对照：同C=.01、原五折，R26融合0.722039/0.804054；对应PCA32融合0.710713/0.808511，原R26 .709014/.800000。定位改善但整答比PCA融合下降，维数/正则几何也变化；不混入公共人工QA表。公共QA局部8次幂门控18候选同预算全部仍选alpha0，没有新胜出模型。

下一组同步组合已完成CPU准备：run_large_matched_combination_v2.py，三对象各相同三C的LR与一次固定单调树，均获得同一large已选轮次信号。当前large仍训练中，组合未拟合。旧12控制回放exact；v2明确处理不收敛，不改变预算。原R16整答监督对照R30已完成且审核通过：R26+窗口监督 .707156/.787234，+窗口及整答监督 .703499/.801394，定位未提升，不替代R29 .722039/.804054。

FAVA合成辅助分支当前只有数据及token准备：最终v2为7482答/7408组，19729原合成span，非人标；输入映射全量通过、最长1481。拟固定aux一轮→原QA三轮，同generic base初始化，未运行GPU训练。该新增语义模型若完成，将给各对照同等使用机会，不能仅强化候选；它不改变官方QA测试。

## 最新：固定旧模型后加入同一个large信号（自研组合）

本节全部属于自研组合/消融，尤其Lookback＋large不能标为原版Lookback基线。直接重新拟合融合头的12个候选均未超过旧最佳；其上游训练内分数使融合明显偏重large，但仅凭系数不能确定因果。为检验限制权重是否有帮助，六个旧模型各同六alpha `[0,.2,.4,.6,.8,1]`，按原规则选单个候选，先逐窗组合、再取整答max。没有新增模型拟合，端点全部核对，保留完整36候选。

| 底座 | large权重 | 窗口F1 | 整答F1 |
|---|---:|---:|---:|
| Lookback＋tail2＋引用LR | 0.4 | 0.679293 | 0.873096 |
| Lookback＋tail2树 | 0.6 | 0.677619 | 0.879227 |
| HARP_claim＋tail2＋引用LR | 0.4 | 0.689809 | 0.883495 |
| HARP_claim＋tail2树 | 0.4 | 0.685120 | 0.890995 |
| semantic_claim＋tail2＋引用LR | 0.4 | 0.690159 | 0.878788 |
| semantic_claim＋tail2树 | 0.4 | **0.690281** | **0.891089** |

当前两项最好来自最后一行的同一候选。其逐窗分数为 `0.6 × 原semantic/tail2树分数 + 0.4 × large分数`，阈值定位0.65746985、整答0.45667378。该方法包含额外语义模型并读取完整回答，不能称为纯白盒或实时逐词元检测器；资料组校准仍反复开发。详见results/large_fixed_convex_v1/REPORT.md。

R31另在原R16训练开发场景完成：R26＋large为.731707/.829431，原R29为.722039/.804054。独立代码/数值审核已过，但数据为助手标注human_gold=false、反复开发，且骨干/维度/上游监督同时变动；不混入上表或称独立测试。

当前36凸组合已独立复算通过。按154资料组固定阈值重采样5000次，相对同预算Lookback树，定位差+.012662的95%区间为[-.008045,.031403]，整答差+.011862的区间为[-.019894,.044423]。与Lookback LR及旧HARP引用LR的区间也跨0；点估计领先尚不等于稳定胜出，且该区间未计反复选型偏差。详见results/large_convex_group_bootstrap_v1/REPORT.md。

## 2026-09-12 08:24 类型监督对照收尾

公共QA当前仍保留同一候选 `semantic_claim＋tail2树＋large权重0.4`：窗口 **0.690281**、整答 **0.891089**。同预算增强Lookback LR为0.679293/0.873096，Lookback树为0.677619/0.879227；此前HARP＋tail＋8特征LR为0.679660/0.868687。154资料组、5000次固定阈值配对区间对三个比较的双F1均跨0，且未计反复校准选型偏差；点估计领先尚不是稳健优势或独立测试证据，窗口目标0.75仍未达到。

原3680 fit类型清点已通过：634个source_id、615组；560300个lexical词元中，无风险512902、仅无依据42078、仅冲突5284、两类重叠36（3答）。另105408个非lexical不作训练负例。两类OR与全部665708 raw BPE的原binary风险完全一致，未改标签。82答同时包含两类，不等于同词元重叠。

| 本轮对照 | 训练范围 | 原控制 窗口/整答F1 | 新读出或双类型头 窗口/整答F1 |
|---|---|---:|---:|
| 旧TCN两个辅助头直接sigmoid-max | 原634fit，固定旧已选epoch3，零新训练 | 0.640745 / 0.857143 | 0.645510 / 0.861244 |
| hidden64两类直接LR | 3680fit，同原三C和binary权重 | 0.615621 / 0.842640 | 0.610490 / 0.844444 |
| hidden64_risk两类直接LR | 3680fit，同原三C和binary权重 | 0.620005 / 0.843137 | 0.618731 / 0.849558 |

每行两项指标来自同一候选。旧辅助头只改固定读出，不新选轮；其小收益没有超过当前强组合，也不是3680训练的同条件结果。直接双LR共12次新拟合、6候选（fit session16093 actualexit0），两头共享同一个C，固定窗max再答max；两个旧控制输出只是同一binary LR的确定性复制，不冒称增强容量控制。两种输入定位均略降、整答略升，不扩C、不增权、不改max。此前类型加权binary LR亦为负结果，三种做法不可混为同一个实验。

独立核验均已通过：旧辅助读出原213159词元/210364窗/793答、旧主头、两阈值与fit/cal计数精确复现；直接双LR的12头696220窗概率、scaler手算、3839答max及选择/计数全exact（审计session15119 actualexit0）。没有新增GPU操作或打开测试。

根代理fit-only实际词元权重盘点显示：冲突占原风险词元11.22%，却占风险损失系数质量16.57%；所有风险词元占lexical的8.46%，实际系数质量35.25%。因此不能简单解释为冲突从未获权重或被数量完全淹没；系数质量不是实际梯度或因果证据，不据此新增训练。

当前队列按根代理状态记录：NLI第5轮已完成、正运行第6轮，最终选型待定；FAVA仍按原GPU串行队列待运行，不写成已完成或提前引用中途成绩。本地时间只作此文档记录。原R16 R32已独立审核通过，但没有同时改善两项，继续保留R31定位较高的固定C=.01版本；这些human_gold=false的反复开发成绩不混入公共QA表。

记录：[fit类型清点](D:/Projects/Multi_Agent_Graph_Analysis/prelab/benchmark_ragtruth_qa/results/fit_type_partition_inventory_v1/REPORT.md)、[旧辅助读出独立核验](D:/Projects/Multi_Agent_Graph_Analysis/prelab/benchmark_ragtruth_qa/results/existing_auxiliary_readout_v1/INDEPENDENT_AUDIT.md)、[直接双LR完整结果](D:/Projects/Multi_Agent_Graph_Analysis/prelab/benchmark_ragtruth_qa/fit_expansion/direct_type_heads_v1/REPORT.md)、[直接双LR审计](D:/Projects/Multi_Agent_Graph_Analysis/prelab/benchmark_ragtruth_qa/fit_expansion/direct_type_heads_v1/INDEPENDENT_AUDIT.json)、[原词元损失系数盘点](D:/Projects/Multi_Agent_Graph_Analysis/prelab/benchmark_ragtruth_qa/results/fit_loss_exposure_v1/REPORT.md)。

## 2026-09-12 08:32 NLI完整结果与36组合收尾

NLI初始化的完整六轮训练 session97326 已 actual exit0，原规则固定选第5轮：公共QA窗口 **0.638658**、整答 **0.847291**。这属于同3680fit/159cal的额外语义检测器初始化对照；上游任务来源仍有未能完全核实的部分，不宣称已证明预训练样本独立。未打开官方测试。

根端固定36凸组合 session35625 已 actual exit0。只把同一已选NLI分数加入原六个LR/树底座，各给同六alpha，不使用large组合的输出当新底座，不新训练组合模型。所有六家均选0.2；每行两项来自同一个候选。

| 原底座 | NLI权重 | 窗口F1 | 整答F1 |
|---|---:|---:|---:|
| lookback__old_lr | 0.2 | 0.665229 | 0.872549 |
| lookback__old_tree | 0.2 | 0.670381 | 0.874419 |
| harp_claim__old_lr | 0.2 | 0.685063 | 0.873096 |
| harp_claim__old_tree | 0.2 | 0.683599 | 0.876289 |
| semantic_claim__old_lr | 0.2 | 0.680312 | 0.864078 |
| semantic_claim__old_tree | 0.2 | 0.682716 | 0.870466 |

按既定双指标规则，本轮最好是HARP旧LR＋NLI0.2的 **0.685063 / 0.873096**。未超过已保留的semantic旧树＋large0.4 **0.690281 / 0.891089**，当前最佳不换。此前相同large预算的Lookback LR为0.679293/0.873096、Lookback树为0.677619/0.879227；既有资料组区间跨0且未计反复选型偏差，仍不声称稳健领先或独立测试胜出，窗口目标0.75未达到。

独立审计已 actual exit0：统一e5及哈希、36概率公式、12个端点、原210364窗口／793回答max、原双阈值／fit-cal计数、六家选择均精确一致。没有重训、GPU或新增候选。审计保留原所有36输出，不按类型挑权重。

根端固定类型诊断另已完成：单NLI显性冲突178/997；HARP旧LR＋NLI0.2为266/997，正常回答内误报162窗。各使用既有阈值，类型可重叠；不同模型阈值下的召回只能描述，不能据此认定NLI专门解决了冲突，更不追加组合搜索。

根代理已确认FAVA正式启动：session79363、实际PID63224，按原已授权aux一轮→QA三轮顺序独占GPU，尚无完整结果或选中模型。本地时间仅为文档快照，不是训练完成时间；不得把排队/部分轮次写成最终成绩。

来源：[NLI36结果](D:/Projects/Multi_Agent_Graph_Analysis/prelab/benchmark_ragtruth_qa/results/nli_fixed_convex_v1/REPORT.md)、[本次独立审核](D:/Projects/Multi_Agent_Graph_Analysis/prelab/benchmark_ragtruth_qa/results/nli_fixed_convex_v1/INDEPENDENT_AUDIT.md)、[固定类型描述](D:/Projects/Multi_Agent_Graph_Analysis/prelab/benchmark_ragtruth_qa/results/nli_fixed_type_diagnosis_v1/REPORT.md)。

## 2026-09-12 08:43：真实分组留出训练已落实

- 新交叉拟合入口 `src/run_group_crossfit_lb_large.py` 已CPU准备、真实小模型训练循环检查并经根代理完整读码。原615材料组固定3折，所有生成器回答同组；原3680fit、634原生回答、159cal及4BPE口径不变。
- 三个固定C的Lookback已actualexit0（session16074），均6次迭代；原形状重放actualexit0（session42546），全部168123留出窗逐值一致、组隔离与一次覆盖通过。首次分块浮点差≤6.66e-16已另存，未改模型/分数。
- 三个large折各固定3轮，从通用初值训练；仅用折外组、过滤原前三轮顺序、重算折内权重，最后实际小批大小正确缩放。两组合器将只在原634fit窗口训练，cal两列复用旧全fit模型预测；没有在cal拟合树。large与两组合器目前未训练，无新F1。
- 根代理已授权唯一GPU所有者data_build在FAVA实际退出后，依次check与large fold0→1→2，每折实际退出后再接下一折。预计GPU75–95分钟为估算；旧large同资源门禁复用。磁盘本次实测约131.9GB可用，三最终检查点预计14.3GB。
- FAVA原session79363/PID63224已完成7482条辅助数据一轮（659.5秒），同一优化器进入QA第1轮；原QA三轮计划不变，完整结果待定。

当前最佳仍为同一候选 .690281/.891089；定位尚未达到.75，稳定领先尚未获证。交叉拟合能检验训练内分数偏差，但2/3材料训练量及历史cal选型仍有局限。依据：`results/group_crossfit_lb_large_v1/PLAN.md`、`ROOT_REVIEW.json`、`LOOKBACK_RESULT.md`、`LB_REPLAY.json`。



## FAVA完成、匹配基线更新与下一轮训练

记录时间：2026-09-12T09:02:39。FAVA原session79363实际exit0，固定合成辅助一轮加QA三轮全部结束，30.4分钟。统一选QA2，单模型窗口0.661103、整答0.833333；QA3训练定位0.818319但cal定位0.642369，增加轮数仍未解决泛化。不得拼接QA1或QA3的整答最高值。

六组原底座各获得相同FAVA信号及六档权重，36候选均完成，12端点和两级计数重放通过。该轮最佳是HARP线性组合，窗口0.687696、整答0.870466；Lookback最佳定位0.672805，整答0.871795；semantic线性0.683675/0.858586，semantic树0.680976/0.859903。这批结果没有超过旧候选，不替换。

累计large/NLI/FAVA三轮的18个已选同预算组合已汇总，未新增拟合、权重搜索或阈值。当前semantic树+large仍为0.690281/0.891089，两项点估计均不低于这些已选Lookback/HARP对照；最接近的HARP+FAVA定位已到0.687696，不能宣称优势稳定或SOTA。所有结果仍为反复使用的cal159；独立QA test150保持封存。完整表：results/matched_detector_trials_v1/REPORT.md。

诊断：FAVA单模型在原997个显性冲突窗口只检出164个，最佳FAVA组合202个。新增自动数据没有自动解决关系冲突；标签及类型分母原样保留。完整记录：results/fava_fixed_convex_v1/REPORT.md、ARITHMETIC_REPLAY.json、results/fava_fixed_type_diagnosis_v1/REPORT.md。

防过拟合训练已接续：FAVA释放GPU后，分组留出large fold0正式启动session2476、实际PID14496；fold1/2按实际退出顺序串行。三折Lookback已全部完成；该分支尚无最终组合成绩。

另已准备PsiloQA官方train辅助候选：16115英文行中，263条字符/标记失败和3组6条标签冲突去重后共隔离268条，保留15847答、6862材料组、27996自动风险span；其中15212风险答、635无标记答。4条仅空complexity保留。原文及失败记录全保存，参考答案与模型身份只在单独出处文件，输入仅资料、问题、原回答。CPU词元映射已全部完成，32.27秒、异常0、无截断，最长2761个编码词元；共3372272输入词元、1101613词面词元，其中606721个风险词元。635条无标记答案全保留；3条原风险标签仅覆盖非字母数字部分，保留原整答标签但不伪造词面正标签。尚未训练；自动标注和约96%风险答不代表人标或部署分布。见auxiliary_psiloqa_v1/manifest.json、auxiliary_psiloqa_review_v1/REPORT.md。
