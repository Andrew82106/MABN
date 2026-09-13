# Semantic window v2 — native-634 development pilot

v2 修复了 v1 的主要评分路径：直接在项目 4-BPE 窗口训练 any-error 读出，停用 conflict-max 和 add-gate，也不再把 claim 风险分数 max 铺到整条 claim。claim 特征只按窗口内真实 lexical token 归属做加权平均。

fit OOF 选择 `whitebox_geometry__C0.1`，宽度 26。四个特征变体、三个 C、两级阈值和选择规则均在读取 calibration 前冻结。

## Fit OOF 候选

| 候选 | 维度 | 窗口F1 | 窗口AP | 整答F1 | 整答AP |
|---|---:|---:|---:|---:|---:|
| `whitebox_geometry__C0.001` | 26 | 0.609652 | 0.588294 | 0.743262 | 0.788623 |
| `whitebox_geometry__C0.01` | 26 | 0.609888 | 0.589064 | 0.741935 | 0.792105 |
| `whitebox_geometry__C0.1` | 26 | 0.610815 | 0.588625 | 0.742857 | 0.795561 |
| `whitebox_attribution_geometry__C0.001` | 62 | 0.606254 | 0.607111 | 0.740541 | 0.778449 |
| `whitebox_attribution_geometry__C0.01` | 62 | 0.605279 | 0.599549 | 0.741797 | 0.785026 |
| `whitebox_attribution_geometry__C0.1` | 62 | 0.605077 | 0.594410 | 0.745578 | 0.784656 |
| `whitebox_attribution_nli_geometry__C0.001` | 68 | 0.606280 | 0.617505 | 0.750643 | 0.782937 |
| `whitebox_attribution_nli_geometry__C0.01` | 68 | 0.607283 | 0.610358 | 0.758621 | 0.787902 |
| `whitebox_attribution_nli_geometry__C0.1` | 68 | 0.606459 | 0.604702 | 0.757412 | 0.785258 |
| `whitebox_attribution_geometry_local__C0.001` | 65 | 0.608684 | 0.612510 | 0.748370 | 0.784140 |
| `whitebox_attribution_geometry_local__C0.01` | 65 | 0.609369 | 0.605384 | 0.742188 | 0.791609 |
| `whitebox_attribution_geometry_local__C0.1` | 65 | 0.610005 | 0.600344 | 0.747688 | 0.788923 |

选择键依次为两级 F1 的较小值、窗口 F1、整答 F1、两级 AP、更少维、更小 C、预注册顺序。calibration 未参与选择。

## 选中模型完整指标

| 口径 | n | TP/FP/FN/TN | P | R | F1 | AUROC | AP | threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fit OOF 窗口 | 168123 | 13696/9672/7781/136974 | 0.586101 | 0.637705 | 0.610815 | 0.873569 | 0.588625 | 0.800747254 |
| fit OOF 整答 | 634 | 299/178/29/128 | 0.626834 | 0.911585 | 0.742857 | 0.789076 | 0.795561 | 0.432928054 |
| cal 严格窗口 | 42241 | 3716/1664/2268/34593 | 0.690706 | 0.620989 | 0.653995 | 0.907753 | 0.700032 | 0.800747254 |
| cal 严格整答 | 159 | 94/26/6/33 | 0.783333 | 0.940000 | 0.854545 | 0.881525 | 0.923673 | 0.432928054 |
| cal F1-opt 窗口诊断 | 42241 | 3851/1826/2133/34431 | 0.678351 | 0.643549 | 0.660492 | 0.907753 | 0.700032 | 0.770433959 |
| cal F1-opt 整答诊断 | 159 | 91/18/9/41 | 0.834862 | 0.910000 | 0.870813 | 0.881525 | 0.923673 | 0.515770307 |

## 与冻结参照比较

| 方法/口径 | 窗口F1 | 整答F1 |
|---|---:|---:|
| v2 严格 fit 阈值 | 0.653995 | 0.854545 |
| v2 cal F1-opt 诊断 | 0.660492 | 0.870813 |
| Lookback 统一 cal F1-opt | 0.600882 | 0.845455 |
| 历史 incumbent（cal 选型） | 0.690281 | 0.891089 |

v2 严格结果相对 Lookback 为窗口 +0.053113、整答 +0.009091；相对 incumbent 为窗口 -0.036286、整答 -0.036544。共同 cal-F1Opt 诊断相对 Lookback 为窗口 +0.059610、整答 +0.025359。

## 边界与审计

- fit 的五折均按 source-connected group 隔离；每个 fit 窗口恰有一个外折预测。
- whitebox 和可选 local 的 fit Lookback/large 输入来自既有上游 group-OOF；本层没有重训上游。它们不是为本层五折重新嵌套生成，故 OOF 身份限于冻结上游加本层来源组留出。
- calibration 只在完整 fit freeze 后执行一次；cal-F1Opt 只作诊断。
- 正式 baseline 文件的前后哈希一致，official test 未打开，GPU 未使用。
- 本轮归因训练范围只有 native 634 答；正式 Lookback 已用 3680 fit。分数可按相同 cal 口径点对点查看，但不能解释为同训练预算胜出。
- calibration 已被项目反复用于开发；本报告不是独立测试结论。

可复跑入口：`python src/run_semantic_window_v2.py run-all --output-dir results/<fresh-directory>`。
