# 正式 baseline 统一口径审计 v2

审计日期：2026-09-13。结论：3项已完成迁移在各自准确命名的冻结身份下通过统一评测；RAGLens、RefChecker、MVA、ReDeEP均保持N/A。未运行GPU、未重训或修改任何baseline模型、未读取official test。

统一口径为：fit 3,680答/615组、cal 159答/154组，材料组零重叠；fit/cal分别有653,979/42,241个4原始BPE、stride 1合格窗口，全量为3,839答/696,220窗。另有692/80个候选窗因不含字母数字字符而按共享几何固定排除。可训练基线只用fit训练，推理型基线覆盖全3,839答；主开发比较仍只在cal159上进行。回答分数=`max(全部合格窗口分数)`；窗口和回答分别按`F1 → precision → 较高阈值`选cal阈值，预测为`score >= threshold`；统一报告F1、AUROC、AP。作者原生聚合和指标只另列复核。

| baseline | 冻结原生输出 | 唯一允许的无参数、无标签、确定性映射 | 状态或N/A原因 |
|---|---|---|---|
| Lookback Lens结构迁移 | 8-BPE、stride 1的冻结风险概率 | 每个raw token取覆盖它的全部8-BPE分数均值；4-BPE窗再取词元均值；回答取窗max | 完成。接收3,680条fit；2条不足8-BPE而无原生span，实际LR行来自3,678答；独立复算最大差`3.33e-16` |
| GHOST最终结构/参数迁移 | 750树RF的整答风险概率 | 概率原值广播到该答全部4-BPE窗；回答取窗max，故整答值不变 | 完成。实际拟合全部3,680个fit整答行；未复跑作者50组五折搜索，seed42是本地约定 |
| LUMINA作者公式迁移 | 每词元固定`0.5×IPR−0.5×MMD²` | 4-BPE窗取冻结词元均值；回答取窗max | 完成。v2映射阶段物理不读gold；LUMINA/IPR/−MMD的窗口、回答max、作者均值均逐值复放旧结果 |
| RAGLens | `predict_proba`整答风险概率 | 原值广播到全部4-BPE窗；回答取窗max；解释峰值不作定位分数 | N/A。未来须为全3,839答推理、只用3,680条fit训练；当前BF16 Llama2约12.55GiB＋指定SAE约4.00GiB，超过8GiB，且无忠实本地入口 |
| RefChecker-Mistral-SFT＋NLI＋官方localizer | claim级E/N/C硬标签及冻结localizer HTML | E→0，N/C→1；固定HTML对齐到字符和4-BPE；失败或未落入合格窗则整答广播；回答取窗max | N/A。必须推理全3,839答；未量化claim抽取器13.489GiB超过8GiB，完整链未运行 |
| RefChecker-Mistral-SFT＋RepC＋官方localizer | 同上，checker为RepC | 同上 | N/A。除13.489GiB抽取器外，RepC主干约13.489GiB＋分类器约1GiB，并有官方设备/调用接口阻断 |
| MVA | 完整CRF的Viterbi二元词元序列 | 4-BPE窗取冻结词元位max；回答再取窗max；不重解码、不平滑 | N/A。论文Data2Text已选配置、top5/checkpoint和完整选择身份未公开，不能用构造器默认值或本地轻量网络补位 |
| ReDeEP | 作者整答连续分数；内部token量不是已验证定位输出 | 冻结整答分数原值广播到全部4-BPE窗；回答取窗max | N/A。公开评分入口有分区/截列问题，选择与归一化provenance不全；官方FP16 7B也超过8GiB，现有NF4缓存不可冒充 |

## 共享评测证据

已完成三项共同覆盖calibration `159`答、`154`材料组、`42,241`个窗口；正类为`100`答、`5,984`窗。GHOST正式训练使用3,680个fit整答行；Lookback使用同一fit分区，但冻结8-BPE几何令2条短答无训练span，实际639,955个LR行来自3,678答；cal均未进入拟合。LUMINA无训练并对全3,839答完成冻结推理。三项适配入口静态检查均无`fit/partial_fit/fit_transform/backward/optimizer.step`调用。当前23项可本地核验的冻结代码/协议哈希全部匹配；RAGLens远端身份沿用其下载前门禁保存的来源/API快照。

训练证据：Lookback的[准备记录](../lookback_official_span_v1/preparation_complete.json)绑定3,680/159答并列出2条短答N/A，[训练汇总](../lookback_official_span_v1/summary.json)记录`real_fits=1`与639,955个fit span；GHOST的[冻结协议](../ghost_official_answer_rf_v1/protocol.json)固定`fit_answers=3680`且禁止cal进入fit，[训练汇总](../ghost_official_answer_rf_v1/summary.json)记录fit `n=3680`与`real_fits=1`。全量696,220窗及分区数由[LUMINA无gold映射冻结](../lumina_k4_evaluation_adapter_v2/mapping_complete.json)和[RefChecker全量CPU审计](../../research/refchecker_baseline_feasibility_r32_v1/CPU_AUDIT.json)独立记录；后者的[自检](../../research/refchecker_baseline_feasibility_r32_v1/SELF_CHECK.json)已通过。

| baseline | 4-BPE AUROC / AP / F1 | 回答 AUROC / AP / F1 |
|---|---:|---:|
| Lookback Lens | `0.872400 / 0.638061 / 0.600882` | `0.848814 / 0.906297 / 0.845455` |
| GHOST | `0.632713 / 0.239092 / 0.320361` | `0.588983 / 0.734942 / 0.772201` |
| LUMINA | `0.691406 / 0.270062 / 0.331299` | `0.702203 / 0.790876 / 0.785047` |

关键绑定：Lookback模型/原生分数/适配分数=`ba796d97… / 225487aa… / 1d16cc20…`；GHOST模型/原生分数/适配分数=`ffe86a51… / 8a980797… / ba7b8072…`；LUMINA v2入口/适配分数=`dd9d8a38… / d96dcd01…`，其映射冻结记录为`64564d4c…`。RefChecker当前边界v3为`bf5c6526…`。

## 本轮修正

1. LUMINA旧v1在分数落盘前访问gold做一致性断言。新v2从无标签`feature_inputs.jsonl`独立重建同一窗口，先冻结输出，评分阶段才打开gold；所有旧分数逐值完全相等，模型与公式未改。
2. RefChecker v2直接用官方strict作整答结果。v3改为共同`answer=max(window)`；风险claim若不能覆盖任何合格窗就广播全答。由此有风险claim时max必为1，无风险claim时max为0，与strict风险值规则上等价；strict只作原生复核。
3. MVA在正式结果表的“待完成”改为N/A；MVA和ReDeEP补齐上述唯一允许映射，但完整身份未冻结/执行前仍不得计分。
4. RAGLens/RefChecker旧门禁只覆盖634条原生fit＋159条cal。当前协议已改为完整范围：RAGLens全3,839答提取、只用3,680条fit训练；RefChecker全3,839答推理。696,220个合格窗及772个固定排除窗均由现有数据文件逐行核实。

Lookback/GHOST当前映射函数不访问标签值，但其旧元数据加载器会解析含标签整行；这是功能无标签保证。LUMINA v2已额外做到文件级隔离。所有现有数字仍是反复使用的cal开发结果，不是封存test结论。

证据入口：[Lookback独立复算](../lookback_k4_evaluation_adapter_v1/INDEPENDENT_RECOMPUTE.json)、[GHOST独立复算](../ghost_k4_evaluation_adapter_v1/INDEPENDENT_VERIFY.json)、[LUMINA v2映射冻结](../lumina_k4_evaluation_adapter_v2/mapping_complete.json)、[RAGLens门禁](../../research/raglens_baseline_feasibility_v1/REPORT.md)、[RefChecker v3边界](../../research/refchecker_baseline_feasibility_r32_v1/FORMAL_BOUNDARY_V3.json)、[MVA协议](../mva_official_baseline_protocol_v1/protocol.json)、[ReDeEP协议](../redeep_official_baseline_protocol_v1/protocol.json)。
