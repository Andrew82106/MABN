# Semantic window v3 — evidence-union native-634 pilot

v3 保持 v2 的 whitebox12+geometry14 主干和直接 4-BPE 窗口监督，只增加冻结的 44 维 evidence-union claim 统计；第二个变体再加入 v2 的 36 维归因统计。逐 claim 特征按窗口内真实 lexical token 归属求均值，没有 conflict-max、add-gate 或 claim 风险 max 铺开。

fit OOF 选择 `backbone_union__C0.001`，宽度 70。两种变体、三个 C、两级阈值和选择规则均在读取 calibration 前冻结。

## Fit OOF 候选

| 候选 | 维度 | 窗口F1 | 窗口AP | 整答F1 | 整答AP |
|---|---:|---:|---:|---:|---:|
| `backbone_union__C0.001` | 70 | 0.609944 | 0.588455 | 0.759531 | 0.786105 |
| `backbone_union__C0.01` | 70 | 0.609138 | 0.584870 | 0.757310 | 0.785938 |
| `backbone_union__C0.1` | 70 | 0.609094 | 0.582922 | 0.757225 | 0.787659 |
| `backbone_union_attribution__C0.001` | 106 | 0.608832 | 0.599543 | 0.747748 | 0.783427 |
| `backbone_union_attribution__C0.01` | 106 | 0.606521 | 0.592079 | 0.748261 | 0.788068 |
| `backbone_union_attribution__C0.1` | 106 | 0.602766 | 0.589280 | 0.750337 | 0.787422 |

选择只使用 fit OOF：依次比较两级 F1 的较小值、窗口 F1、整答 F1、两级 AP、维度、C 和预注册顺序。

## 选中模型完整指标

| 口径 | n | TP/FP/FN/TN | P | R | F1 | AUROC | AP | threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fit OOF 窗口 | 168123 | 13445/9164/8032/137482 | 0.594675 | 0.626019 | 0.609944 | 0.875333 | 0.588455 | 0.808510009 |
| fit OOF 整答 | 634 | 259/95/69/211 | 0.731638 | 0.789634 | 0.759531 | 0.796320 | 0.786105 | 0.723667505 |
| cal 严格窗口 | 42241 | 3617/1615/2367/34642 | 0.691323 | 0.604445 | 0.644971 | 0.906335 | 0.719675 | 0.808510009 |
| cal 严格整答 | 159 | 73/8/27/51 | 0.901235 | 0.730000 | 0.806630 | 0.890678 | 0.932968 | 0.723667505 |
| cal F1-opt 窗口诊断 | 42241 | 4142/2381/1842/33876 | 0.634984 | 0.692179 | 0.662349 | 0.906335 | 0.719675 | 0.709057647 |
| cal F1-opt 整答诊断 | 159 | 92/17/8/42 | 0.844037 | 0.920000 | 0.880383 | 0.890678 | 0.932968 | 0.511082847 |

## 与冻结参照比较

| 方法/口径 | 窗口F1 | 整答F1 |
|---|---:|---:|
| v3 严格 fit 阈值 | 0.644971 | 0.806630 |
| v3 cal F1-opt 诊断 | 0.662349 | 0.880383 |
| v2 严格 fit 阈值 | 0.653995 | 0.854545 |
| v2 cal F1-opt 诊断 | 0.660492 | 0.870813 |
| Lookback 统一 cal F1-opt | 0.600882 | 0.845455 |
| 历史 incumbent（cal 选型） | 0.690281 | 0.891089 |

v3 严格结果相对 v2 严格为窗口 -0.009024、整答 -0.047916；相对 Lookback 为 +0.044089/-0.038825；相对 incumbent 为 -0.045310/-0.084459。

## 边界与审计

- fit 五折按 source-connected group 隔离，每个 fit 窗口恰有一个外折预测。
- evidence-union 的 claim 行通过 response index、claim id、microclaim index 和 hypothesis SHA-256 四重绑定；原生 fit/cal 候选缓存均 100% 命中。
- whitebox 的 fit Lookback/large 输入已从上游三折 held artifacts 逐窗重建，确认为 source-group OOF。
- calibration 只在完整 fit freeze 后执行一次；cal-F1Opt 只作诊断。
- 正式 baseline 前后哈希一致，official test 未打开，GPU 未使用。
- 本轮仍只有 native 634 答，不能解释为与 3,680-fit 正式方法同预算比较，也不自动替换 incumbent。
- calibration 已被项目反复用于开发；本报告不是独立测试结论。

可复跑入口：`python src/run_semantic_window_v3.py run-all --output-dir results/<fresh-directory>`。
