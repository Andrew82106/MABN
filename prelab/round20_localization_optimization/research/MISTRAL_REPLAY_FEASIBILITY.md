# RAGTruth 原生 Mistral 回放可行性

核查日期：2026-09-11。只查看原论文、官方 README、代码目录与作者 issue 回复；另查后续复现工作的代码。未下载模型、未运行 GPU/生成、未读取未用 QA 的回答或错误 span。

**当前可以规划“固定已声明模型的教师强制重建”，还不能证明“精确还原 RAGTruth 原始 Mistral 生成状态”。原 checkpoint 版本不能凭简称猜测。** 教师强制重建指直接送入已保存的回答词元，逐位置计算内部状态，不重新抽样回答。

| 条件 | 官方可确认内容 | 仍未知/含义 |
|---|---|---|
| 原生模型 | 原论文写 Mistral-7B-Instruct；数据 schema 的模型名也是简称。[论文](https://aclanthology.org/2024.acl-long.585.pdf)、[README](https://github.com/ParticleMedia/RAGTruth#dataset) | 未确认 v0.1/v0.2/v0.3、HF repo/revision、权重和 tokenizer 哈希。不能指定任意版本后声称原生精确。 |
| QA 输入 | 3 个 MS MARCO 检索段落；官方保存 `prompt` 文本；附录 B 给通用 QA 指令：仅按上下文回答，缺证据时拒答。[论文 §3.2、附录B](https://aclanthology.org/2024.acl-long.585.pdf) | 段落/上下文 token 截断上限、截断方向与执行代码未确认。论文观察到的长度最大值不等于配置的截断阈值。 |
| 外层模板/系统提示 | README 明确 Llama/Mistral 外层为 `<s>[INST] {prompt} [/INST]`。[README](https://github.com/ParticleMedia/RAGTruth#dataset) | 未见独立系统消息的完整原生成实现；不能自行补入默认 system prompt，也不能把当今 tokenizer 默认 chat template 当成当年的字节与 token 序列。 |
| 推理与抽样 | 项目协作者 thuwyh 确认使用 TGI 与抽样；schema 保存 temperature。[作者回复](https://github.com/ParticleMedia/RAGTruth/issues/6#issuecomment-2264472559)、[schema](https://github.com/ParticleMedia/RAGTruth#dataset) | TGI/依赖版本、精度、top-p/top-k、seed、输出上限等未确认；后续参数询问未见补全。[issue 12](https://github.com/ParticleMedia/RAGTruth/issues/12) |
| 原始 token 轨迹 | 文档提供 prompt/response 字符串及 span 字符偏移。[schema](https://github.com/ParticleMedia/RAGTruth#dataset) | schema 未提供原 input_ids/generated_ids。重新 tokenize 同一字符串不能自动证明与当年 token 轨迹一致。 |
| 原生成 pipeline | 核过 commit `c103204b9ce28d6bbad859304bf30de72b8ed8fe` 的完整非截断目录树。仅数据、README、图和 detector baseline；未见原回答生成脚本。[固定仓库快照](https://github.com/ParticleMedia/RAGTruth/tree/c103204b9ce28d6bbad859304bf30de72b8ed8fe) | baseline 的训练/部署是错误检测器，不能当原回答生成流程；其中 TGI 2.0.1、模型长度等参数不证明原生成参数。[baseline 文档](https://github.com/ParticleMedia/RAGTruth/blob/main/baseline/readme.md) |

抽样参数缺失主要影响“再次抽出完全相同的回答”；若权重、输入及原回答 token 已确定，教师强制前向不需要重新知道随机 seed。本次更实质的缺口是 checkpoint、tokenizer、输入和原 token 轨迹未全部确认。即使日后确认这些，执行精度/内核差异仍需记录，不能直接承诺逐位相同。

## 后续公开重建代码能提供什么

2025 年论文 *First Hallucination Tokens Are Different From Conditional Ones* 的官方 [RAGTruth_Xtended 仓库](https://github.com/jakobsnl/RAGTruth_Xtended) 提供内部状态重建代码。其 [`MODEL_MAP`](https://github.com/jakobsnl/RAGTruth_Xtended/blob/main/rtx/utils.py#L10) 把原简称映射为 `mistralai/Mistral-7B-Instruct-v0.2`，使用 float16，但没有固定 revision。这是后续作者的实现选择，不是原 RAGTruth 作者对 checkpoint 的确认。

静态读取还发现，该仓库 [`process_samples`](https://github.com/jakobsnl/RAGTruth_Xtended/blob/main/rtx/reproduction/reproduce_logits.py#L65) 分别 tokenize `prompt` 与 `response` 后拼接，未在该路径显式加入原 README 的 `[INST]` 外层；上游 [`create_combined_entry`](https://github.com/jakobsnl/RAGTruth_Xtended/blob/main/rtx/create_dataset.py#L46) 直接转存源 prompt。因此，不能直接把此仓库的“reproduced”称呼作为精确回放已获验证的证据。本次未执行或检查它的数据输出。

## 本轮可固定的决策

- 优先保留 RAGTruth QA 为人工 span 候选，具体未暴露规模依据本地 source 元数据审计。
- 原模型版本状态写 `unconfirmed`。若未来采用 v0.2 等指定版本，应明确命名为“声明假设的重建”，并固定权重、tokenizer、完整模板、精度及字符/token 对齐。
- 精确原生回放需要原作者/原工件补充 checkpoint 与输入轨迹证据；本轮未联系作者。
- 原回答文字一旦改变，原人工 span 不能沿用。用 Qwen 强制回放该文本也不能改称 Qwen 原生生成的错误定位实验。
