固定MiniCheck内部状态融合，全部是fit634/cal159开发结果。额外核查模型已读完整陈述与资料，属于离线方法。

| 方法 | fit窗F1 | cal窗P | cal窗R | cal窗F1 | cal整答F1 |
|---|---:|---:|---:|---:|---:|
| minicheck_hidden64_C0.001 | 0.608 | 0.558 | 0.684 | 0.615 | 0.850 |
| minicheck_hidden64_lookback_nll_C0.001 | 0.685 | 0.603 | 0.616 | 0.609 | 0.849 |
| minicheck_hidden64_lookback_nll_risk_C0.001 | 0.687 | 0.599 | 0.625 | 0.612 | 0.840 |
| minicheck_fixed | 0.405 | 0.273 | 0.903 | 0.419 | 0.775 |
| minicheck_calibrated | 0.520 | 0.484 | 0.671 | 0.562 | 0.810 |
| frozen_lookback | 0.635 | 0.541 | 0.577 | 0.558 | 0.861 |
| all_positive | 0.227 | 0.142 | 1.000 | 0.248 | 0.772 |
| all_negative | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| lookback_minicheck_fusion | 0.674 | 0.580 | 0.631 | 0.604 | 0.852 |
| full_lb_pca64_tcn | 0.633 | 0.613 | 0.600 | 0.607 | 0.854 |
| full_lb_harp64_tcn | 0.634 | 0.589 | 0.633 | 0.610 | 0.854 |

| 方法 | 冲突类 | 阳性窗命中 | 原span至少命中 | 原span完整覆盖 |
|---|---|---:|---:|---:|
| minicheck_hidden64_C0.001 | Evident Conflict | 402/997 | 25/43 | 14/43 |
| minicheck_hidden64_C0.001 | Subtle Conflict | 74/109 | 5/5 | 2/5 |
| minicheck_hidden64_lookback_nll_C0.001 | Evident Conflict | 279/997 | 21/43 | 8/43 |
| minicheck_hidden64_lookback_nll_C0.001 | Subtle Conflict | 58/109 | 5/5 | 2/5 |
| minicheck_hidden64_lookback_nll_risk_C0.001 | Evident Conflict | 298/997 | 21/43 | 10/43 |
| minicheck_hidden64_lookback_nll_risk_C0.001 | Subtle Conflict | 64/109 | 5/5 | 2/5 |
| minicheck_fixed | Evident Conflict | 672/997 | 31/43 | 29/43 |
| minicheck_fixed | Subtle Conflict | 89/109 | 5/5 | 3/5 |
| minicheck_calibrated | Evident Conflict | 445/997 | 20/43 | 16/43 |
| minicheck_calibrated | Subtle Conflict | 82/109 | 4/5 | 2/5 |
| frozen_lookback | Evident Conflict | 180/997 | 21/43 | 4/43 |
| frozen_lookback | Subtle Conflict | 50/109 | 5/5 | 1/5 |
| lookback_minicheck_fusion | Evident Conflict | 274/997 | 20/43 | 7/43 |
| lookback_minicheck_fusion | Subtle Conflict | 74/109 | 5/5 | 2/5 |

映射全量793答、213159个原始词元，671625个非空白字符无缺覆盖；原金标与窗口分母未变。PCA64只在原20288个fit抽样位置拟合，解释方差97.70%。
三族各3个C均保存，阈值与C只按既定cal规则选择；此处的整体与分类型成绩均需在封存test上再确认。Subtle Conflict仅5个cal span。
不能把新增MiniCheck状态称作纯Llama白盒；不能因PCA64有高解释方差就断言保留了全部有用判别信息。
