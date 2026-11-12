# Llama2 检查点与 RAGTruth 质量口径

核查日期：2026-09-11。来源核查仅使用匿名 Hugging Face API、HTTP HEAD、小配置/索引/许可文件，以及 RAGTruth 官方文档与 baseline 源码；随后按明确授权完成下文记录的公开检查点下载。未加载模型、未接受或申请 gated 许可、未读取具体 QA 回答或 test 标签文件。

**NousResearch 提供可公开访问的备选；其 safetensors 与 Meta 当前仓库的文件哈希相同，但分词器配置不同。拟运行的是声明模型与 NF4 设置下的教师强制重建，不是原始生成 trace 的精确复现。**

## 检查点与访问状态

| 项目 | Meta 官方发布 | NousResearch 发布 |
|---|---|---|
| 仓库 | `meta-llama/Llama-2-7b-chat-hf` | `NousResearch/Llama-2-7b-chat-hf` |
| 本次固定 commit revision | `f5db02db724555f92da89c216ac04704f23d4590` | `351844e75ed0bcbbe3f10671b3c808d2b83894ee` |
| 匿名访问实测 | API `gated=manual`；权重 HEAD 与配置 resolve 均 HTTP 401。公开元数据可读。 | API `gated=false`；两片 safetensors HEAD 均 HTTP 200，小配置可读。 |
| safetensors | 2 片，共 **13,476,872,576 bytes**（约 13.477 GB / 12.551 GiB） | 2 片；大小及 LFS SHA256 与左侧逐片相同。 |
| PyTorch `.bin` | 2 片，共 **13,476,950,097 bytes** | 3 片，共 **26,953,781,249 bytes**。与 safetensors 属不同文件集合，不能混下载。 |

来源：[Meta API 元数据](https://huggingface.co/api/models/meta-llama/Llama-2-7b-chat-hf?blobs=true)、[Nous API 元数据](https://huggingface.co/api/models/NousResearch/Llama-2-7b-chat-hf?blobs=true)、[Meta 固定文件树](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/tree/f5db02db724555f92da89c216ac04704f23d4590)、[Nous 固定文件树](https://huggingface.co/NousResearch/Llama-2-7b-chat-hf/tree/351844e75ed0bcbbe3f10671b3c808d2b83894ee)。HEAD 请求未取权重内容；“同文件”证据来自服务端元数据，未来实际下载后仍应本地验哈希。

| 两仓共有 safetensors 文件 | bytes | HF LFS SHA256 |
|---|---:|---|
| `model-00001-of-00002.safetensors` | 9,976,576,152 | `66dec18c9f1705b9387d62f8485f4e7d871ca388718786737ed3c72dbfaac9fb` |
| `model-00002-of-00002.safetensors` | 3,500,296,424 | `0fd6895090da1b2ccffdb93964847709a3b31e6b69fe7dc5a480dce37c811b1d` |

两仓 `model.safetensors.index.json` 的 Git blobId 同为 `cbe75f3ff6fdfd5c5e084d6a4968e6d10ce0490d`。Nous 公开索引列出 323 个 tensor，tensor 数据 `total_size=13,476,835,328`；它不含文件头，故略小于上述下载总量。这里只需选择 safetensors 一套；NF4 是加载时量化，不能把 NF4 显存用量当成原权重下载量。[固定索引](https://huggingface.co/NousResearch/Llama-2-7b-chat-hf/blob/351844e75ed0bcbbe3f10671b3c808d2b83894ee/model.safetensors.index.json)

## 分词器、模板及来源披露

| 项目 | 核查结果 |
|---|---|
| 基础词表模型 | 两仓 `tokenizer.model` 都为 499,723 bytes，LFS SHA256=`9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347`。 |
| 完整 tokenizer 并不相同 | `tokenizer.json`：Meta 1,842,767 bytes，Nous 1,842,764 bytes，Git blobId 不同。`tokenizer_config.json` 与 `special_tokens_map.json` 也不同。不能只凭 SentencePiece 模型同 SHA 宣称编码行为相同。 |
| Meta 当前模板 | 公共 API 中可见 `chat_template`：区分可选 system 消息、要求 user/assistant 交替；user 内容会 `strip()`，assistant 内容前后加空格并接 EOS。special tokens 的 `normalized=false`。 |
| Nous 当前配置 | 无 `chat_template`；`legacy=false`，`add_bos_token=true`，`add_eos_token=false`，special tokens 的 `normalized=true`。因此不要依赖库版本提供的默认模板。 |
| Nous padding 配置不一致 | 模型 `config.json` 为 pad ID 0；special_tokens_map 把 pad 写为 `<unk>`；tokenizer_config 为 null；generation_config 为 32000，恰等于 vocab_size，超出合法 token ID 范围。教师强制路径应显式固定 padding 与 attention_mask，避免继承生成配置。 |
| 原 RAGTruth 模板 | 原 README 明确使用 `<s>[INST] {prompt} [/INST]`。重建应以原保存 prompt 和该外层为依据，显式固定 BOS、回答边界及 tokenization 选项；不能擅加默认 system 消息。原生成 token IDs 未确认，仍须披露重建假设。 |

来源：[Meta 公共 API](https://huggingface.co/api/models/meta-llama/Llama-2-7b-chat-hf)、[Nous tokenizer 配置](https://huggingface.co/NousResearch/Llama-2-7b-chat-hf/blob/351844e75ed0bcbbe3f10671b3c808d2b83894ee/tokenizer_config.json)、[Nous special tokens](https://huggingface.co/NousResearch/Llama-2-7b-chat-hf/blob/351844e75ed0bcbbe3f10671b3c808d2b83894ee/special_tokens_map.json)、[Nous generation 配置](https://huggingface.co/NousResearch/Llama-2-7b-chat-hf/blob/351844e75ed0bcbbe3f10671b3c808d2b83894ee/generation_config.json)、[RAGTruth README](https://github.com/ParticleMedia/RAGTruth#dataset)。Meta 受限配置正文未下载；其模板及 special-token 内容来自公开 API。

两仓模型卡都声明这是 Meta 的对话模型、转换为 Hugging Face Transformers 格式。Nous 不是 Meta 官方发布组织；本次未发现该仓库独立的完整转换日志。可准确披露为“NousResearch 发布的 HF 格式检查点，其当前 safetensors 文件的 HF 元数据哈希与 Meta 官方当前版本一致”，不要额外声称 tokenizer/config 或 RAGTruth 当年运行环境也一致。[Meta 模型卡](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/blob/f5db02db724555f92da89c216ac04704f23d4590/README.md)、[Nous 模型卡](https://huggingface.co/NousResearch/Llama-2-7b-chat-hf/blob/351844e75ed0bcbbe3f10671b3c808d2b83894ee/README.md)

两仓 `LICENSE.txt` 的 Git blobId 同为 `51089e27e6764fb9f72c06a0f3710699fb6c9448`，`USE_POLICY.md` 也相同；均受 **Llama 2 Community License 与其使用政策**约束。Nous 的 HF card 未填写 `license` 元数据，但正文仍指向 Meta 许可并保留接受许可说明。公开 HTTP 可读不代表换成 MIT/Apache 许可；本次未代用户接受条款。[Meta 许可](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/blob/f5db02db724555f92da89c216ac04704f23d4590/LICENSE.txt)、[Nous 许可](https://huggingface.co/NousResearch/Llama-2-7b-chat-hf/blob/351844e75ed0bcbbe3f10671b3c808d2b83894ee/LICENSE.txt)

## 可预注册的质量与 span 规则

以下官方 baseline 均固定于 RAGTruth commit `c103204b9ce28d6bbad859304bf30de72b8ed8fe`。规则在接触新 test 标签前冻结，不据测试结果调整。

| 字段 | 官方含义/做法 | 建议冻结的主分析规则 |
|---|---|---|
| `quality` | `incorrect_refusal`：上下文存在答案但模型误拒答；`truncated`：回答意外截断。README 没有声明完整枚举。官方 prepare_dataset 对 train/test 都要求精确等于 `good`。 | **只保留 `quality == 'good'`，其他值整条排除**，记录排除原因。缺失/未知值不补为 good，另记 schema 异常；后一点是本项目实现建议。 |
| `implicit_true` | README 指内容正确但上下文未提及；论文更谨慎称可能真实。官方严格 RAG 口径仍把无上下文支持的内容计为错误，baseline 保留全部 labels。 | **保留这些原 span 作为正标签**。不据此排除回答；若另报宽松口径，须预先单独注册，不能替换主分析。 |
| `due_to_null` | 主要指 Data2txt 中把 JSON null 当作 false/no 导致的无依据内容；是 span 属性，不是回答质量旗标。 | **不据此剔除 span 或回答**。QA 子集无需另造基于此字段的排除规则。 |

官方依据：[字段定义](https://github.com/ParticleMedia/RAGTruth/tree/c103204b9ce28d6bbad859304bf30de72b8ed8fe#dataset)、[quality == good 过滤](https://github.com/ParticleMedia/RAGTruth/blob/c103204b9ce28d6bbad859304bf30de72b8ed8fe/baseline/prepare_dataset.py#L31-L38)、[保留全部原 span](https://github.com/ParticleMedia/RAGTruth/blob/c103204b9ce28d6bbad859304bf30de72b8ed8fe/baseline/dataset.py#L105-L109)、[按 labels 是否为空判回答](https://github.com/ParticleMedia/RAGTruth/blob/c103204b9ce28d6bbad859304bf30de72b8ed8fe/baseline/predict_and_evaluate.py#L79-L81)、[论文 §3.4](https://aclanthology.org/2024.acl-long.585.pdf)。

建议另固定独立的结构校验：原 response 文本不改写；span 偏移和原文不一致、非法偏移等应报数据完整性异常，不静默挪动标注；不能按有无错误、错误个数或模型分数决定纳入。该段是实现建议，不是声称官方已采用同样校验。source 去重、事件分组及 fit/cal/final test 的边界由独立输入审计决定。

## 实际下载完成记录

2026-09-11 14:57:02 UTC 完成，下载进程 session 18090 正常退出。固定上述 Nous revision，目录为 `prelab/models/Llama-2-7b-chat-hf`。10 个允许文件实际合计 **13,479,255,401 bytes**；仅两片 safetensors、索引、模型 config、4 个 tokenizer 文件、LICENSE 与 USE_POLICY，无 `.bin` 或残留 `.partial`。不需要生成的 `generation_config.json` 未纳入下载清单，源文件差异仍保留在上文核查。

两片权重的实际 SHA256 分别为上表的 `66dec18c…faac9fb` 和 `0fd68950…811b1d`，均与完整 LFS SHA256 匹配；其他文件按 LFS SHA256 或 Git blob SHA1 校验，且全部记录实际 SHA256。下载途中一次连接中断由原进程从 8,419,016,704 bytes 自动续传，未重启下载进程。最后确认目录文件集合、10 个文件大小及全部校验标志均一致。

完整文件名、字节数、实际哈希、来源哈希、起止时间与完成状态见 [model_download_manifest.json](model_download_manifest.json)。下载仅 HTTP 与 CPU 文件校验；`model_loaded=false`、`account_authorization_modified=false`。
