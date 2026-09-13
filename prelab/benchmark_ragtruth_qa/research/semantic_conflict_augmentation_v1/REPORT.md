# Semantic Conflict Augmentation v1：fit-only 可行性报告

## 核心发现

RAGTruth QA 原 fit 的冲突监督确实太少：634 个回答中只有 91 个含冲突；646 个错误 span 中只有 109 个 Evident Conflict、7 个 Subtle Conflict。映射到共同 4-BPE 评测后，168,123 个窗口里只有 3,261 个冲突窗口，占全部窗口 1.94%、全部正窗口 15.18%。SC 只有 141 个窗口。冲突小头很容易把普通负窗学得很好，却学不到冲突内部的实体、数字、否定、时间和来源槽位。

fit 人工冲突 span 的表面线索也说明需要关系级训练：116 个 span 中，74 个含数字表面模式、42 个显式提到 passage、33 个含专名代理、18 个含否定、11 个含时间；这些集合互有重叠，也不能直接当标签。

## 旧路线为什么失败

1. **`Original:` 不是可直接替换的答案。** 旧审计的 4,381 个 conflict span 中，4,159 个能解析出一个 `Original:`，但固定 100 条复核只有 14 条能原样替换；按原层比例回代仅约 2.7%。大量内容是 schema 值、整句引文或给标注者的解释，直接使用会破坏语法或泄露 `source states/not mentioned` 等标签捷径。
2. **FAVA 局部 pair 有量但缺少证据认证。** 10,040 个实体/关系局部修复可保证字符补丁可逆，却不能保证“修后端”受资料支持，也不能保证整答无其他错误。
3. **旧 auxiliary conflict pilot 没学出可迁移边界。** 在 fit-only 对照中，candidate 的漏报区冲突 AP 低于同预算 control；与原模型做 `max` 只新增 38 TP、208 FP，且没有新增 SC。根因是辅助域和 QA 域不对齐，以及 OR 融合只能增加告警、不能删除旧 FP。

本方案针对这三个问题：正例直接取当前 QA 的原资料句；扰动只改一个明确槽位；两端保留完全相同的问题与资料；不使用解释性 `Original:` 文本；按 source/donor 连通分量隔离；训练时只施加局部相对偏好。

## 实际产物

自动候选共 5,067 对；严格规则后保留 **4,109 对**：

| 类型 | 严格 pair | 覆盖 owner group |
|---|---:|---:|
| attribution | 1,514 | 599 |
| negation | 1,944 | 564 |
| temporal | 348 | 168 |
| number | 291 | 131 |
| entity | 12 | 7 |

全部严格 pair 通过单连续编辑、原支持句存在、扰动全文不与任一原资料句完全相同、无说明性标签措辞等机械检查。共享支持句跨 split component 的数量为 0，最大连通分量只有 3 个原 group。

entity 只有 12 对，这是有意保守后的真实短板，不能靠高权重伪装成充足数据。实体冲突仍应主要依赖 RAGTruth 人工 span 和以后单独的人审增强；本版最适合先测试来源归属、否定、时间和数量。

## 92 条 post-freeze 规则质检

| 类型 | clear silver | ambiguous | reject |
|---|---:|---:|---:|
| entity | 12 | 0 | 0 |
| number | 19 | 1 | 0 |
| negation | 19 | 1 | 0 |
| temporal | 18 | 1 | 1 |
| attribution | 16 | 2 | 2 |
| 合计 | **84** | **5** | **3** |

主要残余噪声是不完整原句、句外指代和标题被误当陈述。规则已在抽这批 post-freeze 样本前冻结，本报告保留这些失败，不再据此追调后重报同一批成绩。

## 建议

先用严格 pair 做冻结 NLI 成对压力测试，不直接训练完整模型。若 NLI 对 number/temporal/negation/attribution 能稳定让扰动端 contradiction 更高，再把它作为 semantic attribution 模型的冲突子头辅助监督。最终融合用 group-OOF 校准头并允许抑制 FP；不要重走 `max/OR` 路线。

本轮完成的是数据可行性与训练协议，不是模型效果。未训练、未用 GPU、未读 calibration/test、未改正式 baseline。
