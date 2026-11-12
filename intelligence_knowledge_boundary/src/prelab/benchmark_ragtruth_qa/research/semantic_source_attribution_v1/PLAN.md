# Semantic source attribution v1

## 研究问题

在资料增强问答中，模型生成一个事实单元时，是否真正依赖了与该事实对应的资料句；这种具体来源归因能否补足当前模型对事实冲突的漏检。

## 共同考卷

- 数据仍为原始 RAGTruth QA：fit 634 答、calibration 159 答；两部分按 source-connected group 隔离。
- 定位单位仍为 4 个原始 Llama BPE、步长 1 的可评窗口；整答分数仍取本答窗口最大值。
- 人工字符跨度、窗口标签、整答标签、阈值规则和 F1/AUROC/AP 均不改；official test 保持封存。
- 正式外部 baseline 的模型结构、特征、训练配方和原生读出不改，只接受事先规定的无参数输出映射。

## 无标签白盒抽取

使用现有 Llama-2-7B-Chat NF4 teacher-forced 重放。它不是原始生成轨迹，也不是生成前预警。

对回答中已经生成的每个 lexical token `q`，在每层每头计算：

`contribution(q,k) = attention(q,k) * ||V(k)||_2`

- `q` 是当前词元读入后的 post-token query。
- 历史回答区域只含严格早于 `q` 的 key，排除当前词元。
- 每条资料句在本头内先对所含 token 取最大贡献；同一 claim 再对其 lexical query 取平均。
- 保存完整 `claim × source sentence × 32 layers × 32 heads` 的 FP16 CSR 缓存，同时保存 source、三篇 passage、other-context、strict-previous-answer 的 FP32 层级汇总。
- 资料句、claim 和 token 的边界只由既有冻结字符坐标确定；抽取不读标签，不选择层、头或句子。

## 下游模型

第一轮只比较低自由度、可审计的候选：

1. 归因单独读出：1024 维具体资料句归因的固定汇总特征。
2. 归因＋已有 NLI：按归因确定性选择 top-1/top-3 资料句，再使用冻结 ModernBERT NLI 的支持/中立/冲突概率。
3. 归因＋NLI＋既有白盒：加入已缓存的 Lookback、NLL 和隐藏状态汇总。
4. any-error 与 conflict 两个独立训练头；两头的组合规则在 fit 内冻结。

fit 内按 source-connected group 做五折 OOF。模型家族、正则和候选数在读取 calibration 成绩前冻结；阈值只从 fit OOF 选择。claim 风险以 max 投影到所覆盖的 4-BPE 窗口，整答再取窗口 max。

历史 calibration 优化过的 incumbent 只作只读结果参照，不作为新头的训练输入。最接近的干净脚手架是 Aligned Evidence Head；第一轮新方法应与它比较并记录新增 TP、FP，尤其是 Evident/Subtle Conflict 的召回。

## 前进门槛

- 先通过全量几何、公式、哈希、OOF 隔离和独立复算。
- calibration 窗口 F1 目标至少 0.75；整答 F1 至少 0.75，且最终候选应尽量不低于当前保留点估计 0.891089。
- 若未达门槛，只依据预先保留的诊断继续：证据句选错、正确句的 NLI 判别失败、归因层头失效或 claim 到窗口扩散；不通过修改正式 baseline 制造差距。

