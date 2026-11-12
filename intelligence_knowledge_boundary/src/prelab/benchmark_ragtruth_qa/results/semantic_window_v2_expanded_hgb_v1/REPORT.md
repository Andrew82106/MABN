# Expanded semantic window HGB v1

这是我们的非线性读出：复用已审计的 26 维 `whitebox+geometry`、同一 653,979 个窗口、同一层级权重和 answer-max 聚合；没有改动 baseline。

三档 HGB 在拟合前已预注册，仅用 3,680 答案/615 个 source-connected groups 的五折 OOF 选择。选择规则先最大化 window/answer F1 的较小值，再比较 window F1、answer F1、两级 AP 与模型复杂度。

| candidate | all-domain OOF window F1 | all-domain OOF answer F1 | native held-OOF window F1Opt | native held-OOF answer F1Opt |
|---|---:|---:|---:|---:|
| whitebox_geometry__hgb_leaf7 | 0.596859 | 0.693027 | 0.614321 | 0.746099 |
| whitebox_geometry__hgb_leaf15 | 0.596192 | 0.689595 | 0.612126 | 0.743551 |
| whitebox_geometry__hgb_leaf31 | 0.596182 | 0.690956 | 0.610154 | 0.746099 |

选中 `whitebox_geometry__hgb_leaf7`。全量拟合与两个 OOF 阈值冻结后，cal159 只评估一次。

| calibration comparison | threshold provenance | window F1 | answer F1 |
|---|---|---:|---:|
| selected HGB | fit-OOF frozen | 0.642096 | 0.775281 |
| expanded whitebox+geometry LR | frozen expanded-fit OOF | 0.640311 | 0.788571 |
| semantic_claim__old_tree__large_weight0.4 | calibration F1Opt | 0.690281 | 0.891089 |
| Lookback adapted project-4 window | calibration F1Opt | 0.600882 | 0.845455 |

HGB/LR 使用 fit-OOF 冻结阈值；incumbent 与 Lookback 是 calibration-F1Opt，阈值来源不匹配，只作描述性参照。native held-OOF 是诊断，不参与本轮选择或阈值冻结。官方 test 未读取。
