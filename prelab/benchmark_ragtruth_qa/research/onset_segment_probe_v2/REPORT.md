# OSR v2：direct-emission anchor 的严格 fit OOF

## 结论

**停止，不进入 calibration。** 预注册主模型 `anchored_filter` 在 RAGTruth fit634 的严格五折 source-group OOF 上，4-BPE 窗口 F1 为 **0.389269**，整答 `max` F1 为 **0.691329**；远低于既有开发参考 0.690281 / 0.891089。全过程只读 fit 标签及现有 `hidden_last`、Lookback、NLL 缓存，只用 CPU，未读 calibration/test，未修改 baseline。

v2 的直接风险发射改善了部分排序：相对 v1，窗口 AUROC 从 0.715829 升到 0.748662，整答 AUROC 从 0.547944 升到 0.629324。但它没有解决实际误报：clean-window FPR 从 0.162514 升到 **0.192764**，起点窗召回只有 **0.292434**。因此不值得为它打开 calibration，也不值得继续做银标回放。

## 冻结结构

协议在 OOF 执行前冻结为 207 个参数的三线性头：每个 lexical BPE 分别预测直接风险 emission、从 clean 进入风险的 onset，以及处于风险后的 continue/stop。每折先按组、回答、cell 等质量加权，再在各头内做两类平衡；推理时用该训练折的类平衡前真实层级先验反校正。

三个读出事先固定，没有按结果换主模型：

- `anchored_filter`（主模型）：先由 onset/continuation 得到转移预测，再乘当前词元 emission 的似然比；当直接 emission 趋近 0 时，风险可直接退出。
- `emission_identity`：纯 emission、无转移的 identity 控制。
- `transition_only`：去掉 emission anchor 的转移消融。

v1 把风险答尾的最后一个 lexical 表征重复成 EOS stop。v2 不再制造这个不存在的 EOS 特征，只训练真实 lexical `1→0` stop，得到 503 个 stop、16,591 个 continue。各折层级自然先验范围为：emission 0.10685--0.11156、onset 0.00610--0.00824、continuation 0.92138--0.92636。

## 完整 OOF 指标

所有阈值各自在 pooled fit OOF 上按 F1、precision、recall、最大阈值依次打破平局；没有使用 calibration 调阈值。

| 读出 / 层级 | Precision | Recall | F1 | AP | AUROC |
|---|---:|---:|---:|---:|---:|
| anchored / 4-BPE | 0.298387 | 0.559762 | 0.389269 | 0.298355 | 0.748662 |
| anchored / answer=max | 0.556797 | 0.911585 | 0.691329 | 0.620486 | 0.629324 |
| identity / 4-BPE | 0.287442 | 0.533315 | 0.373551 | 0.286756 | 0.732505 |
| identity / answer=max | 0.547406 | 0.932927 | 0.689966 | 0.626683 | 0.635980 |
| transition-only / 4-BPE | 0.310536 | 0.579690 | 0.404424 | 0.323860 | 0.749234 |
| transition-only / answer=max | 0.544170 | 0.939024 | 0.689038 | 0.663994 | 0.655408 |

主模型的窗口 F1 比 identity 高 0.015718，但整答 AUROC 低 0.006656；它没有同时胜过 identity 的窗口与整答排序。更直接地，transition-only 在窗口 F1、窗口 AP/AUROC、整答 AP/AUROC 上都高于主模型。当前 emission anchor 提供了形式上的退出通道，却没有提供有效的判别增益。

主模型同一窗口阈值下的定位诊断：

| 类别 | 窗口数 | 命中/报警数 | Recall / FPR |
|---|---:|---:|---:|
| released onset | 2,551 | 746 | 0.292434 recall |
| internal continuation | 18,926 | 11,276 | 0.595794 recall |
| clean | 146,646 | 28,268 | 0.192764 FPR |

646 个发布 span 中，任意命中 420 个（0.650155），完整覆盖 126 个（0.195046）。在 122,359 个 clean lexical token 上强制令前态风险为 1 后，anchor 分数低于主窗口阈值的比例仅 0.132830，均值仍为 0.878074。也就是说，公式允许退出，但当前 emission 证据通常弱到不足以推翻 continuation 的高先验。

三个 prior-corrected 头的 pooled OOF 指标如下；continue 的高 F1 主要来自 97.06% 的原始条件正率，stop AP 仍只有 0.216322。

| 头 | Precision | Recall | F1 | AP | AUROC |
|---|---:|---:|---:|---:|---:|
| direct emission | 0.271008 | 0.554461 | 0.364068 | 0.277863 | 0.733249 |
| onset | 0.043351 | 0.130282 | 0.065055 | 0.023365 | 0.770287 |
| continuation | 0.971921 | 0.999337 | 0.985438 | 0.992787 | 0.837669 |
| stop | 0.324873 | 0.254473 | 0.285396 | 0.216322 | 0.837669 |

## 五折稳定性

以下均使用 pooled OOF 阈值，避免每折单独调阈值。

| Fold | window F1 | window AP | window AUROC | answer F1 | answer AUROC | clean FPR |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.410546 | 0.324606 | 0.753064 | 0.706587 | 0.654297 | 0.200193 |
| 1 | 0.296955 | 0.221655 | 0.665993 | 0.679012 | 0.652727 | 0.205113 |
| 2 | 0.407506 | 0.315251 | 0.802811 | 0.659218 | 0.683468 | 0.187447 |
| 3 | 0.387610 | 0.285723 | 0.739758 | 0.694301 | 0.528163 | 0.192501 |
| 4 | 0.439406 | 0.377277 | 0.790345 | 0.719512 | 0.645544 | 0.179388 |
| mean ± population std | 0.388404 ± 0.048621 | 0.304903 ± 0.051052 | 0.750394 ± 0.048140 | 0.691726 ± 0.021062 | 0.632840 ± 0.053922 | 0.192928 ± 0.009108 |

窗口 AUROC 五折均高于 0.55，说明缓存中存在稳定的局部排序信号；但 fold 1 的窗口 F1 仅 0.296955，fold 3 的整答 AUROC 仅 0.528163，且所有折 clean FPR 都在 0.179--0.205。它不具备接近现有系统的稳健定位能力。

## 门禁、边界与复现

预注册门禁 9 项只通过 4 项：通过 answer AUROC、内部延续召回、五折 window AUROC 和 anchored window F1 不低于 identity；失败项是窗口 F1、整答 F1、起点召回、clean FPR，以及 anchored answer AUROC 不低于 identity。状态为 `STOP_BEFORE_CALIBRATION`。

独立复核重新读取 fit gold，重算 group fold、三头先验校正、三个递推、固定 4-BPE 几何、answer=max、全部候选阈值与主指标，并验证 6 个输出哈希和各折 loss 下降，结果为 PASS。`fit_oof_v2/` 保存逐 token/window/answer/span OOF 分数、五折参数、完整指标、manifest 和复核结果。

```powershell
py -3 .\run_fit_oof.py --selfcheck
py -3 .\run_fit_oof.py --run
py -3 .\verify_fit_oof.py
```

`SHA256SUMS.txt` 绑定协议、代码、报告和全部结果文件。运行 manifest 记录总耗时 32.88 秒、CPU-only、`calibration_or_test_labels_read=false`、`gpu_started=false`、`baseline_mutated=false`。
