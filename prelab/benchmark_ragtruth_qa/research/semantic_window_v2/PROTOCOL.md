# Semantic window v2 frozen protocol

本轮只用现有 native QA：fit 634 答、615 个 source-connected 组；calibration 159 答只在模型、特征变体、C 和两级阈值全部冻结后评一次。official test 不读取。正式 baseline 的代码、模型、分数和报告只做哈希核对，不写入。

v2 直接对项目原生 4-BPE、stride-1 合格窗口训练 any-error LR。停用 conflict head、`max(any, conflict)` 和 add-gate；不再训练 claim 风险后把分数 max 铺到整条 claim。claim 的冻结无标签特征仅按窗口内 lexical token 所属关系做加权平均。

固定 claim 特征：12 维 whitebox；4 个预定义 8-layer band 内的 source-attribution 均值和 head 标准差，以及 source/previous/other、top1/top3 concentration、sentence entropy 和 passage HHI，共 36 维；可选 6 维 attribution-selected NLI。固定窗口几何 14 维，描述 claim 边界、相对位置和覆盖。最后一个变体可加入 3 个已有 clean 窗口信号：fit 上的 source-group OOF Lookback/large 概率与 generation NLL。

预注册四个变体：

1. `whitebox_geometry`（26 维）
2. `whitebox_attribution_geometry`（62 维）
3. `whitebox_attribution_nli_geometry`（68 维）
4. `whitebox_attribution_geometry_local`（65 维）

每个变体只试 `C ∈ {0.001, 0.01, 0.1}`，统一为带 StandardScaler 的 L2 logistic regression。fit 做 5 折 GroupKFold；同一 source-connected group 不跨折。每折训练权重依次令 group 等权、group 内 answer 等权、answer 内窗口等权，再做一次 fit-fold 二分类平衡，不在类别平衡后重置组权重。

每个候选只用完整 fit OOF 分数分别选择窗口和整答阈值，规则为 `F1 → precision → 较高阈值`。模型选择键固定为：`min(窗口F1, 整答F1) → 窗口F1 → 整答F1 → 窗口AP → 整答AP → 更少维 → 更小C → 预注册顺序`。选中后才在全 fit 拟合一次并冻结。

calibration 报告 fit 阈值的严格结果，并另报 cal-F1Opt 诊断；后者不改变模型或结论身份。与 Lookback `.600882/.845455`、历史 incumbent `.690281/.891089` 的比较只作为同一反复开发 calibration 上的点估计，不当作独立测试结论。由于本轮归因缓存只有 634 个 native fit 回答，而正式 Lookback 已用 3680 fit，本轮只称 native-634 development pilot。

