# Forced-Evidence Quote Probe V2：开发评测报告

## 结论

V2 已完成 **256 答、3,776 个 claim 的 fit-only 五折嵌套 OOF 评测**，但不推进。P3（模型生成引用＋白盒信号）的窗口 AP 为 `0.230349`，高于 P0 和 P2，且相对 P2 在 5/5 折均为正；然而只有 `1,893 / 3,776 = 50.1324%` 的生成引用能在原资料中逐字找到，远低于预注册的 `95%` 硬门。因此 `advance=false`，按协议停止，不能在这批 pilot 上继续调参。

这不是 calibration 或 official test 成绩。评测只用了 fit 数据；`calibration_read=false`、`official_test_read=false`。完整机器结果见 [EVALUATION.json](../../results/forced_evidence_quote_probe_v2/EVALUATION.json)。

## 评测对象与边界

- 256 个不同资料组，均为原生 `llama-2-7b-chat` 回答。
- 3,776 个 claim，其中风险 claim 486 个（12.87%）。
- 68,373 个固定 4-BPE、stride 1 窗口，其中风险窗口 7,664 个（11.21%）。
- 256 个整答，其中风险回答 131 个（51.17%）。
- 五折按资料组隔离；阈值只由相应 outer-train 的 inner-OOF 选择，再应用到未参与选阈值的 outer fold。
- AP、F1 都是越高越好；窗口和整答均报告 pooled outer-OOF 结果。

四个条件为：

| 条件 | 输入 |
|---|---|
| P0 | 原答案上的 1,024 维 Lookback 衍生特征；使用本实验自己的折内 scaler、回答归一权重、类别平衡和固定 `C=1e-4` LR。 |
| P1 | 标签盲 BM25 机械选句的 21 维表面/关系特征。 |
| P2 | P1 的机械引用，再加入同一 Llama2 的 logprob、关系读出、hidden 和 attention。 |
| P3 | 同一 Llama2 自由生成引用，再加入与 P2 同结构的白盒信号。 |

**P0 只是“Lookback 衍生的本地匹配控制”，不是正式 Lookback Lens 基线。** 它借用了 Lookback 类原始特征，但训练配方是本 pilot 的配方，不能写成作者方法复现或正式基线成绩。

正式基线遵守“我们的考卷、作者的方法”：统一使用本项目的数据、材料组划分、标签、4-BPE 窗口、整答标签、阈值与指标；同时冻结作者方法的模型结构、特征、训练配方和原生读出，只允许在最终输出之后做预先规定、无参数、与标签无关的确定性映射。详见 [BASELINE_PROTOCOL.md](../../BASELINE_PROTOCOL.md) 和 [FORMAL_BASELINE_FREEZE.md](../../FORMAL_BASELINE_FREEZE.md)。本报告不把 P0 与正式 Lookback Lens 结果混算。

## 主结果

| 条件 | 窗口 AP | 窗口 F1 | 整答 AP | 整答 F1 |
|---|---:|---:|---:|---:|
| P0 本地匹配控制 | 0.219783 | 0.308440 | 0.605605 | **0.684783** |
| P1 机械引用表面特征 | 0.225463 | **0.334980** | 0.482519 | 0.673629 |
| P2 机械引用＋白盒 | 0.173349 | 0.223847 | 0.569954 | 0.648199 |
| P3 生成引用＋白盒 | **0.230349** | 0.312379 | **0.610725** | 0.620438 |

P3 的窗口 AP 比 P0 高 `0.010566`，比 P2 高 `0.057000`；相对 P2 的窗口 AP 在 5/5 折均提高，所以预注册的性能方向门全部通过。但 P3 的窗口 F1 只有 `0.312379`，整答 F1 只有 `0.620438`，没有形成稳定的整体改进。它目前只显示出较弱的排序信息，不能称为新模型胜出。

## 为什么没有推进

唯一未通过的推进条件是引用忠实度：

| 推进门 | 结果 |
|---|---:|
| source-exact rate ≥ 95% | **失败：50.1324%** |
| P3 window AP ≥ P0 + 0.01 | 通过 |
| P3 window AP ≥ P2 + 0.01 | 通过 |
| P3−P2 window AP 至少 4/5 折为正 | 通过：5/5 |
| P3 answer AP ≥ P2 answer AP − 0.01 | 通过 |

生成解析/停止无效率仅 `0.1324%`，即主要问题不是格式解析，而是模型生成的“引用”约一半并非资料原句。逐字 exact 的 claim 风险率为 `10.72%`，非 exact 为 `15.03%`；这个差异存在，但不足以支撑可靠定位。

协议预先规定：任一推进门失败，就执行 `stop_without_retuning_on_pilot`。因此 V2 已停止，不能根据本结果在同一批样本上修改 prompt、特征、C、阈值或融合方式后再报 V2 成绩。

## 失败来自信号，不来自窗口几何

后验诊断把真实 claim 标签直接按冻结映射投影到窗口，得到：

| 理想诊断 | 窗口 F1 | 窗口 precision | 窗口 recall | 整答 F1 |
|---|---:|---:|---:|---:|
| claim-geometry oracle | 0.895955 | 0.811521 | 1.000000 | 1.000000 |

这个 oracle 使用了真实标签，只用于回答“现有 claim→4-BPE 映射是否天然做不到精确定位”，不能作为可部署成绩。结果说明映射本身可以支持较高 F1；当前低分主要来自输入信号没有学出哪些 claim 有事实错误。

claim 层的后验 OOF 诊断也支持这一点：P1/P2/P3 的 claim AP 分别为 `0.193157 / 0.168824 / 0.238388`，最优描述性 F1 分别为 `0.318372 / 0.263289 / 0.295053`。P3 虽有最高 AP，但区分强度仍弱。完整诊断见 [FIT_ONLY_DIAGNOSTICS.json](./FIT_ONLY_DIAGNOSTICS.json)。

## 可作出的严格结论

1. “让模型先生成引用，再读取其内部状态”在本 pilot 上带来少量窗口排序增益。
2. 自由生成引用没有成为可靠的资料锚点：source-exact 只有 50.13%，所以 V2 不能推进。
3. 4-BPE 定位目标本身可学；下一版本需要改善证据选择和事实判别信号，而不是放宽窗口标签。
4. V2 结果只属于开发诊断。calibration 与 official test 继续封存；新版本必须先另立协议，再重新评测。

数值执行和冻结规则见 [NUMERICAL_PROTOCOL.md](./NUMERICAL_PROTOCOL.md)；finalizer 与 evaluator 的独立静态审查均为 PASS，见 [FINALIZER_INDEPENDENT_REVIEW.md](./FINALIZER_INDEPENDENT_REVIEW.md) 和 [EVALUATOR_INDEPENDENT_REVIEW.md](./EVALUATOR_INDEPENDENT_REVIEW.md)。
