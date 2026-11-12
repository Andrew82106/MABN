# Token source attribution v4

目标：删除 claim 内 query-token 平均。每个 lexical answer BPE 独立保留 32 层区域 share 与逐来源句归因；评分层再与冻结 claim-sentence NLI 的 E/N/C 合并。

- 白盒公式：`attention(q,k) * ||V_head(k)||_2`，fragment 内逐 head 取 max；head 最后求均值。
- 因果：query 是当前 token 的 post-token state；previous answer 严格 `< q`；未来 key 显式 mask。
- 存储：不保存完整 attention，也不保存 token×sentence×head；native 原始数组约 189 MiB。
- 评分：逐 token 每层 8 维；四个原始 BPE 直接汇总成窗口样本，source-group OOF；正式 v4 只追加 label-blind NLL，历史 cal-selected Lookback/large 禁入候选。
- 封存：抽取不读标签；模型与阈值只由 fit OOF 冻结；cal 只评一次；official test 不打开；baseline 不修改。
- 扩展：同一 runner 支持 `--scope expanded-fit` 的独立缓存；v4 首轮评分仅 native634+cal159。
