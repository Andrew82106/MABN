# Auxiliary conflict v2 失败诊断（fit-only）

## 结论

**没有明显可行的无参数融合，建议停止这条 v2 分支。** Candidate 在 v4 漏报区的冲突 AP 有小幅排序信号，但弱于同预算 control，不能归因于 auxiliary conflict 训练；`max`、raw residual、等权 rank-sum、AND 都没有同时保住总体风险排序并补回冲突。

## 范围与核对

- 只读 QA fit 的 source-connected fold 0：747 个回答、132,213 个 4-BPE 窗口。
- 未读 calibration/test，未训练、未载入模型权重、未用 GPU、未搜索权重或阈值、未修改 baseline。
- 从 `tokens_fit.jsonl` 的原始 span→BPE 映射重建全部窗口。overall label、answer index、EC、SC 均与冻结数组逐位一致；四类标签并集也逐位等于 overall label。窗口数：EC 928、SC 65、EBI 7,694、SBI 3,062（少量跨类重叠）。
- Candidate/control 的 flat-token 分数重新投影到窗口，再按回答取 max，均与存储结果 bitwise exact。完整哈希和断言见 `RESULTS.json`。

## 主结果

沿用冻结 v4 cutoff `0.8502532839775085`，不重新选阈值：

| 分数 | AP | F1 | TP | FP | alerts |
|---|---:|---:|---:|---:|---:|
| v4 | 0.639050 | 0.622716 | 7,259 | 4,321 | 11,580 |
| max(v4, candidate) | 0.626571 | 0.619440 | 7,297 | 4,529 | 11,826 |
| max(v4, control) | 0.356756 | 0.505012 | 8,665 | 13,917 | 22,582 |

Candidate max 只新增 246 个告警，其中 **38 TP / 208 FP，增量精度 15.45%**；因此召回增加不足以抵消误报。固定为 v4 的 11,580 告警预算后，candidate 变成 7,163 TP / 4,417 FP，相比 v4 **少 96 TP、多 96 FP**；冲突召回仅从 0.13092 到 0.13696（+0.00604）。

## v4 漏报区与相关性

v4 cutoff 以下有 120,633 个窗口，其中 863 个 EC/SC，冲突率 0.715%。

| 漏报区冲突 AP | v4 | candidate | control |
|---|---:|---:|---:|
| 全部 | 0.01456 | 0.02202 | **0.02689** |
| 高覆盖（≥0.5） | 0.01552 | 0.02433 | **0.02775** |
| 低覆盖（<0.5） | 0.01496 | 0.01680 | **0.02928** |

Candidate 的弱 lift 是真实的，但每个切片都不及 control。全体冲突 AP 也同样：v4 0.01280、candidate 0.01860、control 0.02432。说明它没有学出 auxiliary 特有的冲突判别优势。

Candidate 与 v4 的 Spearman 相关为 0.852，漏报区仍为 0.830；candidate 与 control 为 0.838。负窗口中 candidate-v4 相关高达 0.836，正窗口只有 0.447：大部分一致性来自容易的负例，新信号在错误窗口上的排序不稳定。

分数尺度也解释了 max 的表现。v4/candidate/control 的中位数分别为 0.025/0.241/0.009，99 分位为 0.983/0.825/0.996；超过 v4 cutoff 的窗口分别为 11,580/778/14,469。Candidate 被压在中间区，极少越过 cutoff；control 则有过大的高分尾部，直接淹没 v4 排序。

## Candidate 新增错误画像

新增 38 个 TP 按窗口真值为：EC 11、SC 0、EBI 15、SBI 12。它没有新增任何 SC。所有新增 FP 在窗口级当然都是 clean；208 个 FP 中，117 个来自完全 safe 的回答，91 个来自其他位置含错误的回答。

| 切片 | 新增 TP | 新增 FP | 增量精度 |
|---|---:|---:|---:|
| 高覆盖 | 19 | 176 | 9.74% |
| 低覆盖 | 19 | 32 | 37.25% |
| 回答 ≤117 token | 4 | 9 | 30.77% |
| 118–165 | 4 | 35 | 10.26% |
| 166–233 | 19 | 63 | 23.17% |
| ≥234 | 11 | 101 | 9.82% |
| 回答前 1/3 | 15 | 62 | 19.48% |
| 中 1/3 | 12 | 83 | 12.63% |
| 后 1/3 | 11 | 63 | 14.86% |

高覆盖新增 TP 中包含全部 11 个 EC，但同时带来 176 FP；低覆盖的 19 个 TP 全是 EBI/SBI。FP 分布跨越回答全程，最长回答更差，但没有一个干净、稳定的长度或位置区域能自然形成 gate。

## 无参数替代诊断

以下只比较 AP，不选阈值，也不形成新候选：raw residual=`candidate-v4`；rank 为 held-fold 内两者等权 rank-sum（仅限探索，部署不可直接使用）；AND 为 `min(v4,candidate)`。

| Candidate 诊断分数 | overall risk AP | overall conflict AP | v4-miss conflict AP |
|---|---:|---:|---:|
| v4 | **0.63905** | 0.01280 | 0.01456 |
| candidate raw | 0.35328 | **0.01860** | **0.02202** |
| max / OR | 0.62657 | 0.01309 | 0.01641 |
| raw residual | 0.05318 | 0.01325 | 0.01408 |
| equal rank-sum | 0.54613 | 0.01515 | 0.01958 |
| min / AND | 0.50333 | 0.01502 | 0.01720 |

- **Residual：不值得继续。** 它几乎抹掉总体风险排序，漏报区冲突 AP 也低于 candidate raw。
- **Rank：不值得继续。** 等权 rank-sum 虽保留一点冲突 lift，但 overall AP 从 0.63905 降到 0.54613；而且 held-fold rank 是传导式诊断，不能作为正式映射。
- **AND：不值得继续。** 固定 cutoff 下仅 532 个告警，precision 0.758，但 recall 0.034、冲突 recall 0.006；它按定义无法找回任何 v4 miss。

因此，现有 v2 输出没有可审计证据支持继续做 residual、rank 或 AND gate；若再调权重/阈值，只会成为新的监督搜索，超出本诊断且不能修复 candidate 不胜 control 的根因。

## 复现文件

- `analyze.py`：NumPy-only、CPU、fit-only 重算与映射核对。
- `RESULTS.json`：完整分布、相关、切片、哈希和无参数诊断数值。

