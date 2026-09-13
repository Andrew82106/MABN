# Round9 runner

已实现 `src/run9.py`，复用本地 Qwen2.5-7B NF4 生成、Round7 ReDeEP/Lookback/LUMINA 和 Round9 binding 提取。没有加载其他大模型，也没有自动检索、自动选题或自动选择随机资料。

## 输入与冻结

准备好的 `data/inputs.jsonl`、`protocol.json`、`data/lumina_random_manifest.json` 必须先存在。每行保留运行身份，但展示内容只包含 system、prompt、questions、passages 的 title/text；questions 和 passages 必须与 prompt 的实际 Questions/Search results 区逐字相同。

`prepare-freeze` 只写新的 `data/freeze.json`，拒绝覆盖旧 freeze。它锁定：

- inputs、随机资料 manifest 和 protocol；
- 已有 PLAN、ANNOTATION_GUIDE、EVALUATION_DRAFT、FEATURES_DRAFT；
- src 下所有 Python 源文件及文件名单。

另用 external_source_sha256 固定四个实际依赖：Round7 的 model7.py、attention7.py、lumina7.py，以及 Round8 的 evaluate8.py；键分别为 ../round7_evidence_grounding/src/... 和 ../round8_token_localization/src/evaluate8.py，按真实 Round9 源码目录解析，不随 --root 测试目录改变。

protocol 的 generation/model 中实际模型参数必须与复用的 model7 一致；checkpoint 名称、路径、冻结状态、NF4 与 bfloat16 也检查。正式冻结前，应完成上述源码和方案文件。冻结后改动这些文件会阻止运行和审计；不能直接改 hash 混用既有缓存。

随机 manifest 格式：

```json
{
  "schema": "round9-random-context-v1",
  "pool_file": "../round7_evidence_grounding/data/dev_inputs.jsonl",
  "pool_sha256": "原始开发池文件的 SHA256",
  "assignments": {
    "本轮 group_id": {
      "donor_id": "固定且不带标签含义的来源标识",
      "passages": [
        {
          "title": "开发池中的原始标题",
          "text": "开发池中的完整原始片段",
          "origin_row_id": "片段所在的开发池 row_id"
        }
      ]
    }
  }
}
```

开发池限定为 R7 的 `dev_inputs.jsonl`，其 40 行 split 均为 development。每个 donor 片段必须在声称的来源行中完全相同；donor 片段数量等于目标输入片段数量，同题两条件复用同一份 donor，不允许与目标任一条件有完全相同的标题或正文。语义层面的主体独立性仍依靠数据审核；这些字面检查不证明语义无关。runner 不新增或选择 donor。

## 执行

从工作区根目录运行，根代理统一负责 GPU：

```powershell
& 'prelab/.venv/Scripts/python.exe' -X utf8 prelab/round9_evidence_binding/src/run9.py --stage prepare-freeze
& 'prelab/.venv/Scripts/python.exe' -X utf8 prelab/round9_evidence_binding/src/run9.py --stage generate
& 'prelab/.venv/Scripts/python.exe' -X utf8 prelab/round9_evidence_binding/src/run9.py --stage core
& 'prelab/.venv/Scripts/python.exe' -X utf8 prelab/round9_evidence_binding/src/run9.py --stage attention
& 'prelab/.venv/Scripts/python.exe' -X utf8 prelab/round9_evidence_binding/src/run9.py --stage lumina
& 'prelab/.venv/Scripts/python.exe' -X utf8 prelab/round9_evidence_binding/src/run9.py --stage audit --require-complete
```

`--stage all` 依次执行 generate/core/attention/lumina，一次进程复用同一个模型。可用 `--row-ids ID1,ID2` 或 `--limit N` 执行预先指定的少量行；`--area` 改变输出子目录，默认 data。审计只加载 tokenizer，不加载模型、不修复缓存、不拟合或评测。

## 缓存与特征接口

生成逐条写入 `data/generation_records/{row_id}.json`，每条完成后原子更新 `generated.jsonl` 与 generation manifest。保存原始生成 IDs、移除终止特殊符号后的精确响应 IDs、字符范围、原输出和解析状态，不重采样或改写答案。

core、attention、lumina 分别写入 `data/features`、`data/attention`、`data/lumina` 的同名 NPZ/JSON；JSON 为提交标志。完整缓存通过 hash 校验后直接跳过，不加载模型。只有 NPZ 没有 JSON 时视为未提交，可重新计算。已有 JSON 对应文件缺失、哈希变更或代码/协议/原始生成版本不符时明确停止，不自动覆盖。每条开始和提交前重新核对冻结资料。

每份特征 JSON 含 source_generation_sha256、arrays_sha256、stage_signature_hash、完整源码签名和协议/数据 freeze hash。core 的 binding_feature_names 位于 JSON 顶层；全部 NPZ 附精确 token_ids，兼容 evaluate9.Bank。原始辅助方法的无效回答项均值可能为 NaN；用于 token 评测的矩阵必须全部有限。空回答或内部特殊符号等不满足原提取器要求的情况会停止并保留已生成记录，不悄悄丢弃该行。

传给 core 的输入只含可见字段；传给旧辅助方法的输入再加 row_id/question_id 运行标识。禁止实际 subjects、aliases、references、category、condition、coverage、split 等进入提取器。旧 model7 生成函数要求 split/condition 作为返回元数据，包装层只传固定字符串 unlabelled；真实运行身份在生成后附回记录。编号项的最终范围只供旧辅助方法离线聚合，Round9 候选选择和逐 token 特征不使用未来输出。

## CPU 验证

`tests/test_run9.py` 的 12 项测试通过，使用真实 tokenizer 和模拟生成/特征函数；未加载 7B 权重。覆盖逐阶段运行、部分完成恢复、全缓存不加载模型、修改冻结文件/代码/协议/外部依赖时先于模型加载拒绝、source-generation 哈希绑定、NPZ 被改或未提交、固定 donor 溯源、可见字段白名单，以及真实 evaluate9.Bank 读取 816 维组合特征。测试不创建或读取金标文件。

`tests/test_binding9.py` 的 11 项 CPU 小模型/手工张量验证也通过。GPU 数值、耗时和显存仍由根代理统一冒烟确认。本 runner 不改 Round7/8 源文件或正式数据。
