# Semantic source attribution v1 failure diagnosis

本诊断只读 fit 与 calibration；未打开 official test，未改正式 baseline。

## 结论

失败不是单一原因。归因原始块能学到一定信号，但加到稳定的 whitebox 后增益很小；随后又被三个问题放大：只有 195 个正例的 conflict head 被强制取 max、claim 分数被 max 投影到整段窗口、top-3 证据对冲突证据的覆盖不足。

## 已有四个 head

| head | fit OOF AUROC | fit OOF AP | fit F1-opt | cal AUROC | cal AP | cal F1-opt |
|---|---:|---:|---:|---:|---:|---:|
| clean_any | 0.8542 | 0.5333 | 0.5829 | 0.8716 | 0.6628 | 0.6620 |
| semantic_any | 0.8765 | 0.5879 | 0.6175 | 0.8909 | 0.6725 | 0.6804 |
| clean_conflict | 0.5799 | 0.0294 | 0.0648 | 0.5307 | 0.0570 | 0.1279 |
| semantic_conflict | 0.6655 | 0.0381 | 0.0793 | 0.6870 | 0.0537 | 0.1189 |

semantic any-error 的 cal AUROC/AP 为 0.8909/0.6725。但 fit 消融中，1394 维全组合的 F1 只比 12 维 whitebox 高 0.0028，AP 反而低 0.0166；所以不能把 semantic 与 clean 的全部差距归功于新归因。

全组合五折原始系数余弦均值只有 0.512，whitebox 为 0.983。semantic conflict 的 full-train AP 比 OOF 高 0.531，是明显记忆训练集；其 OOF AP 只有 0.038。

## 投影和融合损失

| 只用 any-error head | fit OOF窗口F1 | cal严格窗口F1 | cal F1-opt窗口F1 | cal严格整答F1 |
|---|---:|---:|---:|---:|
| clean | 0.5770 | 0.6385 | 0.6458 | 0.8374 |
| semantic | 0.6188 | 0.6395 | 0.6564 | 0.8416 |

沿实际融合路径，semantic any-only 的 cal 严格窗口 F1 是 0.6395；加入 conflict-max 后降到 0.6107；再走 frozen add-gate 后降到 0.5775。
semantic-max 相对 clean-max 新找回 534 个 TP，却丢掉 638 个原 TP；新增 746 个 FP，同时消掉 2231 个旧 FP。
当前 frozen add-gate 在 cal 新增 459 个 TP，同时新增 590 个 FP，增量精确率只有 0.438。
即使给 claim 完美 0/1 标签，再用同一 max 投影，fit/cal 窗口 F1 上限也只有 0.9158/0.8894；cal 会制造 1489 个结构性 FP。

## 低成本特征块消融

每个 target 只在 fit OOF 比较 raw1024、region288、NLI66、whitebox12、全组合，C 固定为 0.001/0.01/0.1；下面的 cal 是 fit 选定后唯一一次诊断。

| target | fit选中 | C | fit F1-opt | fit AP | cal严格F1 | cal F1-opt | cal AP |
|---|---|---:|---:|---:|---:|---:|---:|
| any_error | combination1394 (1394) | 0.001 | 0.6154 | 0.5848 | 0.6562 | 0.6753 | 0.6651 |
| conflict_only | region288 (288) | 0.001 | 0.0944 | 0.0401 | 0.0760 | 0.1138 | 0.0513 |

完整消融表和系数稳定性见 REPORT.json。

## NLI 与 top-3

- fit: top-3 `1-entailment` 对 any-error 的 AUROC/AP 为 0.724/0.267；attention top-3 覆盖 union 最佳 entailment 的比例为 0.746，最佳 contradiction 的覆盖比例仅 0.373；与 BM25 六候选至少重合一句的比例为 0.971。
- calibration: top-3 `1-entailment` 对 any-error 的 AUROC/AP 为 0.731/0.277；attention top-3 覆盖 union 最佳 entailment 的比例为 0.713，最佳 contradiction 的覆盖比例仅 0.396；与 BM25 六候选至少重合一句的比例为 0.965。

fit/cal 的回答长度、claim 长度和来源句数接近；主要类别变化是 conflict claim 从 2.154% 升到 3.088%。因此没有证据把失败归因于明显的长度分布漂移。cal 中 90 个阳性 claim 同时含风险与正常 token，其中 55 个 claim 的风险 token 不到一半。

## 下一版应改什么

1. 先停用 `max(any, conflict)`；以 any-error 为主输出，conflict 只作为受控附加证据或等待更多冲突样本。
2. 不再把一个 claim 分数无差别铺到它覆盖的所有窗口；按 claim 内 token 的归因/边界分配风险，或直接训练窗口 readout。
3. 证据选择改成 attention 与 BM25/NLI 的并集或可学习 rerank，并保留多句联合 NLI；top-3 attention 不能当作唯一证据入口。
4. any 主头先以稳定的 whitebox 为主干，只加入经过 OOF 证明有增量的低维归因残差；避免 1394 维直接压在 195 个 conflict 正例上。
5. 下一轮仍在我们的固定数据、4-BPE 窗口与整答口径评测；正式 baseline 结构保持不变。
