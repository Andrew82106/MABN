# RAGognizer 全量计划几何只读诊断

日期：2026-09-13  
范围：只读取 fit/calibration 的无标签计划输入、冻结几何、相关实现与 tokenizer 元数据；未访问任何 test 文件，未运行 GPU，未改动数据、模型、基线输出或评测标签。

## 结论

`fit/14641` 的失败不是 tokenizer 漂移，也不是坏数据，而是 RAGognizer 迁移代码曾把所有名为 `k4` 的窗口误当成“必须恰好 4 个 BPE”。项目冻结协议明确规定：`N>=4` 时窗口长度为 4；`0<N<4` 时保留唯一一个长度为 `N` 的短窗口。14641 是全 3,839 答中唯一一个不足 4 个项目 BPE 的回答。

正确且最小的处理是保留该答和原窗口，以 `end=min(start+4,N)` 得到真实槽数，把 `slot_count=end-start` 写进无标签计划，并按实际槽数求均值。不得删行、补零、复制概率、强制除以 4 或改写 gold。

另外有 3 条 fit 回答触发作者展示打包器的 UTF-8 byte-fallback 缺陷。检测头的原生概率仍然存在；可以只对这 3 条采用 fast-tokenizer 字符 offset 做确定性坐标恢复，同时保留每个原生 byte-token 概率和重复字符区间。这个操作属于无参数、无标签的输出坐标适配，不改变作者模型或概率。

## 冻结 4-BPE 几何的真实来源

权威规则和生成链如下：

1. `prelab/benchmark_ragtruth_qa/ANNOTATION_PROTOCOL.md` 的“4个raw BPE的滑动窗口”规定：窗口长度上限为 4、步长 1；`0<N<4` 时只保留一个长度 `N` 的短窗口。
2. `prelab/benchmark_ragtruth_qa/src/feature_qa.py::encode_view` 生成回答的冻结 BPE 坐标：把完整字符串一次性分词，使用 fast-tokenizer offset 与回答字符范围相交，保留跨 prompt/answer 边界的首词元，并把相对 offset 裁剪到回答字符区间。
3. `prelab/benchmark_ragtruth_qa/src/build_gold.py::windows_for_count` 实现为 `[(start, min(start+4,count)) ...]`；`build_gold.py` 再对每个实际窗口合并所含 token offset。
4. 扩充 fit 的 `prelab/benchmark_ragtruth_qa/fit_expansion/prepare_expansion.py` 复用同一 `feature_qa.prepare_row`、`build_gold.windows_for_count` 和 `merge_intervals`，因此没有另设一套几何。
5. RAGognizer 读取的 `evaluation_geometry.jsonl` 由 `prepare_redeep_formal_baseline_v1.py` 从上述冻结窗口逐行去除 label/risk 字段得到；其 SHA256 为 `b87c570dd0536d61bd0d720fc7b46906a046f1f523dd83365fc71d4862a06b07`。

项目几何使用：

- tokenizer：`NousResearch/Llama-2-7b-chat-hf@351844e75ed0bcbbe3f10671b3c808d2b83894ee`
- 本地路径：`prelab/models/Llama-2-7b-chat-hf`
- `tokenizer.json` SHA256：`f7b50bcf6d6672eade5e43514d48e9c1e4e63a56aef7b14acdaca94ce93436f7`
- `tokenizer.model` SHA256：`9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347`
- 运行版本：Transformers `4.51.3`，Tokenizers `0.21.4`
- 精确模板：`<s>[INST] {released_prompt} [/INST] {original_response}`
- 参数：`add_special_tokens=False`、不 padding、不 truncation、不加 EOS。

RAGognizer 作者坐标使用 Meta tokenizer 元数据快照 `f5db02db724555f92da89c216ac04704f23d4590`；其 `tokenizer.json` SHA256 为 `bcd04f0eadf90287bd26e1a183ac487d8a141b09b06aecb7725bbdd343640f2e`。它与项目 tokenizer 的 JSON 不同，因此预先冻结的统一规则是所有行都走字符相交映射，即便某一行恰好逐 token 相同，也不临时切换规则。

## 14641 的逐项证据

- partition：`fit`
- group：`rtqa_group_b6b8a47d9bf4d9a9`
- 原回答：`exclude`，7 个字符
- answer SHA256：`35443277fe297662cfc34d304b04f2f5fb2b853a8292901001ef6232de91a7a3`
- 项目 BPE：唯一 token id `19060`，完整输入位置 `478`，raw 回答 offset `[-1,7]`，clipped offset `[0,7]`
- RAGognizer 作者 BPE：唯一 token id `19060`，概率位置 `479`，回答 offset `[0,7]`
- 冻结窗口：`14641__k4_00000`，`token_start=0`，字符区间 `[0,7]`
- 正确计划：`slot_count=1`，字符相交映射为本地槽 `0 -> 作者 token 0`

两个 tokenizer 在此行的 token id 与回答字符区间恰好相同；完整输入位置相差 1 来自各自的 prompt/template 过程，不是回答几何不一致。旧逻辑检查 `start+4>N` 会在 `0+4>1` 时错误拒绝该合法短窗。

当前成功生成的 `INFERENCE_PLAN.jsonl` 已包含这条 1 槽窗口；本次只读时文件 SHA256 为 `575c7df9dc697f7a594143277ec13c1c6b31bdf531e971bd01bbc5555096ea9a`。执行代理后续若同步审计字段或重新冻结计划，哈希应随内容重新登记，不能继续引用旧哈希。

## 3 条真正的 UTF-8 fallback

真正触发 `fast_offsets_author_pack_byte_fallback` 的行只有 3 条，而且都在 fit：

| answer_id | 特殊字符 | 作者 byte token ids | 特殊字符出现次数 | 重复覆盖字符数 |
|---|---|---:|---:|---:|
| `15069` | `⅔` | `229,136,151` | 1 | 1 |
| `15066` | `⅔` | `229,136,151` | 1 | 1 |
| `12426` | `℉` | `229,135,140` | 4 | 4 |

作者官方 `_pack_probs` 会逐个 token 解码，再由 `_ensure_tokens_in_response` 在原回答中查找解码文本。上述 Unicode 字符被拆成 3 个 UTF-8 byte token；单独解码某个 byte 会得到 `U+FFFD`（`�`），所以展示打包器找不到该字符并抛错。这个失败发生在概率已经由检测头产生之后，因此不代表检测头没有分数。

合规 fallback 应满足：

1. 仅当官方 pack 确实失败、且失败行存在单 token 解码为 `U+FFFD` 的响应 token 时启用；普通行仍必须逐 token 验证官方 pack 输出。
2. 运行时重新取得同一 tokenizer 对同一完整输入的 fast offsets，核对 model input ids、冻结 probability indices、token ids 和字符区间。
3. 从完整检测头输出按冻结 probability indices 原值取概率；一个 byte token 对应一个概率，不能合并或丢弃。
4. 保留 fast offsets 的重复字符区间，再按已冻结的“半开字符区间相交、每个相交作者 token 只计一次”规则映射到项目 BPE。
5. 报告中明确称为“作者展示打包器 Unicode 坐标修复”；不称作者原生 pack 精确复现。

这 3 条的回答字符覆盖均完整：uncovered 均为 0；具有多个原生 token 覆盖的字符数分别为 `1/1/4`。它们都在 fit，因而不会直接改变当前 calibration 159 的分数，但全 3,839 答覆盖门禁要求它们仍必须成功产生冻结原生概率输出。

## 当前审计统计需要纠正的地方

旧 `CPU_AUDIT.json` 曾把 `byte_fallback_row_count` 写成 2,173。这个数包含所有带 `<0x..>` token 的回答，其中大量只是换行 `<0x0A>`，官方 pack 可以正常处理；它不是坐标 fallback 数。

应拆成两个概念：

- `raw_byte_token_rows=2173`：仅描述响应中出现原始 byte token；
- `utf8_coordinate_fallback_rows=3`：真正因单 token 解码为 `U+FFFD` 而无法通过官方 pack 的行。

独立审计、运行器和文档还应同步以下约束：

- 独立窗口检查使用 `end=min(start+4,N)`，并核对计划中的 `slot_count=end-start`；
- 适配器按 `slot_count` 求算术均值，短答不能固定除以 4；
- 增加 1 槽短窗单元测试，并断言全量只有 14641 触发该路径；
- `METHOD_FREEZE.json`、`PROTOCOL.md` 把“恰好四槽”改为“正常答四槽；N<4 时按冻结协议使用唯一 N 槽短窗”；
- 计划、独立审计和 GPU 运行器统一 `coordinate_texts`/`packed_texts` 字段名及含义：普通行为官方 packed text，fallback 行为 fast-offset 对应的原回答字符片段；
- GPU 运行器不能对 3 条 fallback 仍无条件调用会抛错的官方 `_pack_probs`。只在已冻结 fallback 身份匹配时走上述坐标恢复，否则失败关闭。

这些修复只纠正评测坐标和审计描述，不修改 RAGognizer 的 LoRA、检测头、BF16 推理、sigmoid 概率、阈值规则或任何 gold。
