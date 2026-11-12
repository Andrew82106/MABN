# 正式基线结果表

这里只列忠实的论文结构/公式迁移。公共数据为3,680条fit/615组与159条cal/154组，组间无重叠；全3,839答共有696,220个合格4-BPE窗（fit/cal为653,979/42,241），另按固定几何排除772个无字母数字字符候选窗（692/80）。可训练基线只用fit训练；推理型基线覆盖全3,839答。主比较仍只在cal159上按统一整答标签、阈值和指标计分；作者原生尺度另列复核。固定的模型外评测映射不是基线模型改造。本项目融合版和消融不进入本表。

这里的“迁移”只表示把论文算法接到本项目数据字段与划分，**不表示修改算法结构**。模型与统一评测严格分成两层：

| 基线 | 冻结的原方法层 | 本项目统一评测层（无参数、模型外） |
|---|---|---|
| Lookback Lens | 8原始BPE滑窗、1024维注意力比率、作者默认逻辑回归；训练完成后分数冻结 | 每个raw token取覆盖它的冻结8-BPE分数均值；项目4-BPE窗再取词元均值；整答取项目窗最大值；最后按项目标签计分 |
| LUMINA | 每个回答词元固定使用作者代码公式 `0.5×IPR−0.5×MMD²`；没有分类器或可学习融合 | v2只从无标签输入重建项目4-BPE窗并取冻结词元分数均值；先冻结映射，整答取项目窗最大值；随后才打开标签计分 |
| RAGLens | 匹配Llama2-7B的作者layer-15 SAE pre-activation；全答逐通道max、MI top-1000、官方EBM/GAM整答分类器；完整方法尚未运行 | 冻结整答概率原值广播到全部4-BPE窗；作者峰值词元只作解释诊断，不冒充逐词元风险分数 |
| GHOST | 四维特征先对全答词元无权均值，再进入论文参数的750树RF | 同一冻结整答概率原值广播到该答全部4-BPE窗；整答仍用原概率；最后按项目标签计分 |
| RefChecker＋官方NaiveEmbedLocalizer | 官方Mistral-SFT抽取claim-triplet；NLI或RepC输出E/N/C硬标签；带冻结sup-SimCSE权重和官方阈值的作者localizer输出HTML定位 | E→0、N/C→1；HTML→字符→4-BPE，定位失败或未落入合格窗则整答广播；整答取项目窗max。官方strict只另列原生复核；完整方法尚未运行 |
| MVA | 作者完整3072维注意力特征、双向Transformer、CRF训练与Viterbi二元词元序列；完整超参身份尚未冻结 | 每个项目4-BPE窗取冻结Viterbi词元位的max；整答再取项目窗max。只作二元坐标投影，不重解码、不调平滑 |
| ReDeEP | 作者公式逐token连续分数 `H_t=ΣPKS−βΣECS`；固定Khead=1、Klayer=10、α=1、β=.2，具体头/层只在共同fit内按论文相关性规则选择；没有另训分类器或损失 | 冻结原生token分数按字符/BPE重叠无参数映射到项目4-BPE窗，整答取窗max；作者整答均值仅另作原生复核 |
| RAGognizer（2026 arXiv预印本） | 官方Llama-2 checkpoint、released LoRA、集成`hallu_head_neg_16`及原生sigmoid token概率；模型卡默认`transformer_heads`路径、BF16、无postprocessor | 冻结原生概率后按token→字符→项目BPE相交均值；4槽窗按实际槽均值，唯一N<4答保留一个短窗；整答取窗max；作者阈值0.6523仅附录 |

右栏的分数映射不以标签值决定输出，也不训练参数。Lookback/GHOST现有加载器会解析含标签的元数据行，但映射函数不访问标签字段；LUMINA正式v2进一步做到映射阶段不打开任何gold文件，并逐值复放旧分数。所有适配分数先冻结，标签随后只用于阈值与指标。Lookback按锁定配方接收全部3,680条fit，但2条短答不足8-BPE而无原生训练span，实际LR行来自3,678答；GHOST随机森林则拟合全部3,680个fit整答行，按论文最终结构/参数训练，但未复跑作者搜索且seed42为本地约定。这与“统一评测适配器没有重训”是两件事。

所有已完成行使用同一calibration评测分母：159个回答、154个材料组、42,241个4原始BPE步长1窗口，正类分别为100个回答和5,984个窗口。窗口与整答分别按同一 `F1 → precision → 较高阈值` 规则选阈值，判断条件为 `score >= threshold`；F1、AUROC和AP均已保存。下表为简表，只展示F1；作者原生指标放在最后一列，不与统一指标混排。

| 方法 | 状态 | 统一4-BPE窗口F1 | 统一整答F1 | 作者原生复核 |
|---|---|---:|---:|---|
| Lookback Lens 结构迁移 | 完成并独立复算 | 0.600882 | 0.845455 | 8-BPE AUROC 0.855601、AP 0.624460、fit阈值F1 0.587706 |
| GHOST 结构迁移 | 完成并独立复算 | 0.320361 | 0.772201 | 原生仅整答；窗口值是零参数均匀广播 |
| LUMINA 作者公式迁移 | 完成并独立复算 | 0.331299 | 0.785047 | 全词元均值：AUROC 0.743390、AUPRC 0.838979、F1Opt 0.791489 |
| RAGLens 官方Llama2-7B＋SAE管线 | 下载前门禁完成；正式链路未执行 | N/A | N/A | 作者RAGTruth整答macro-F1 0.7636；局部解释没有定位F1 |
| RefChecker 官方源码冻结 | 完整正式基线未执行：8 GiB硬件及官方运行接口受限 | N/A | N/A | 原生为claim-triplet级E/N/C硬标签；不能以量化或替换抽取器冒充正式结果 |
| MVA结构迁移 | 尚不能忠实开训 | N/A | N/A | 作者Data2Text已选参数未公开；不能用本地轻量网络冒充 |
| ReDeEP论文公式迁移 | 完成、身份绑定并独立复算 | 0.329560 | 0.772358 | 全词元均值：AUROC 0.655593、Pearson 0.280707；源码公式另作诊断，不择优 |
| RAGognizer官方Llama-2原版模型 | 完成并独立复算 | 0.507511 | 0.806867 | 作者0.6523整答：F1 0.788845、AUROC 0.724407、AP 0.781285；仅附录 |

已完成五项的统一指标如下；每格依次为 `AUROC / AP / F1`：

| 方法 | 统一4-BPE窗口 | 统一整答 |
|---|---:|---:|
| Lookback Lens 结构迁移 | 0.872400 / 0.638061 / 0.600882 | 0.848814 / 0.906297 / 0.845455 |
| GHOST 结构迁移 | 0.632713 / 0.239092 / 0.320361 | 0.588983 / 0.734942 / 0.772201 |
| LUMINA 作者公式迁移 | 0.691406 / 0.270062 / 0.331299 | 0.702203 / 0.790876 / 0.785047 |
| ReDeEP 论文公式迁移 | 0.669274 / 0.257497 / 0.329560 | 0.590678 / 0.732595 / 0.772358 |
| RAGognizer官方Llama-2原版模型 | 0.824938 / 0.367650 / 0.507511 | 0.706949 / 0.745699 / 0.806867 |

ReDeEP官方源码实际公式的诊断结果为：窗口 `0.673038 / 0.239372 / 0.332532`，整答 `0.620678 / 0.761164 / 0.772201`；它不替换论文公式主结果。ReDeEP整答F1也不能单看：cal中100/159答为正，全部预测正的F1已经是0.772201；论文公式只高0.00016，整答AUROC为0.590678。窗口指标更能反映它在本场景的局部定位能力。

当前本文候选为4-BPE窗口F1 0.690281、整答F1 0.891089。只有统一映射完成后的基线才能与它直接比较；已有跨粒度数字仅作复核。GHOST原生没有定位输出，0.320361只是把整答分数均匀广播后接受同一任务的结果。当前候选及阈值已反复见过cal，仍只是开发对照。

在当前统一开发评测上，本文候选两级F1均高于已完成的Lookback Lens、LUMINA、GHOST、ReDeEP和RAGognizer迁移。所有F1都按当前项目cal-F1Opt口径理解，不能冒充独立测试成绩。

RefChecker 的N/A表示没有执行完整官方链路，不能据此判断其强弱。官方开放抽取器的未量化权重本身为13.489 GiB，超过本机8 GiB显存；RepC另有官方发布代码的设备和调用接口限制。NLI或localizer单独可运行也不构成完整RefChecker。源码、模型revision、映射规则和阻塞证据见[RefChecker正式基线可行性审计](./research/refchecker_baseline_feasibility_r32_v1/REPORT.md)。

RAGLens 的N/A同样不表示方法弱。作者指定的Llama2 layer-15 SAE单文件约4.00 GiB，未量化Llama2权重约12.55 GiB；官方非70B入口把二者同时移到GPU，本机8 GiB无法执行。当前环境还与作者锁定依赖不一致。不能以NF4激活、其他SAE、其他层或新增词元头补出“RAGLens”成绩。下载前身份、资源和零参数广播规则见[RAGLens正式基线可行性门禁](./research/raglens_baseline_feasibility_v1/REPORT.md)。

执行边界见[基线迁移规则](./BASELINE_PROTOCOL.md)，源码身份见[冻结清单](./FORMAL_BASELINE_FREEZE.md)，最新七类方法总审计见[统一口径审计v2](./results/formal_baseline_policy_audit_v2/REPORT.md)，原模型独立核查见[完整性审计](./results/formal_baseline_integrity_audit_v1/REPORT.md)，GHOST统一映射复算见[适配报告](./results/ghost_k4_evaluation_adapter_v1/REPORT.md)，LUMINA无gold映射复放见[v2适配报告](./results/lumina_k4_evaluation_adapter_v2/REPORT.md)，ReDeEP全量身份与复算证据见[正式迁移审计](./research/redeep_formal_baseline_v1/REPORT.md)。

RAGognizer是2026年arXiv预印本。正式链保留官方模型卡默认的集成检测头、LoRA、BF16和原生sigmoid概率；全3,839答输出冻结后才按统一计分尺评估cal159。完整身份、Unicode/短窗边界、资源实测和独立复算见[RAGognizer正式迁移审计](./research/ragognizer_formal_baseline_v1/REPORT.md)。
