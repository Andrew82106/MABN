# 局部配对词元接口

`token_inputs.jsonl` 保留全部20080个独立版本。每条 `model_inputs` 只有原 `retrieved_passages`、空 `question`、本侧 `response`；`target` 与来源身份留在输入字段之外。没有 `risk_mask`、`answer_risk` 或整篇答案真值。

每条成功映射记录包含：

- `input_ids` / `attention_mask`：ModernBERT完整输入，无padding和截断；`encoder_offsets` 是完整输入原字符坐标。
- `answer_encoder_start/end`：每个encoder词元相对本侧回答的原字符坐标，非回答为 `-1`；`answer_encoder_positions` 给回答位置。
- `raw_full_input_ids` / `answer_token_positions` / `response_token_ids`：本侧完整Llama辅助分词及回答轴。它不是历史生成轨迹。
- `response_token_offsets` / `response_token_offsets_raw`：回答相对原字符区间；前者截到回答范围，后者保留开头边界跨越。
- `mapping=[rows,columns,weights]`：从encoder响应词元到raw BPE的稀疏映射。rows是raw位置，columns是完整encoder位置；按原回答非空白字符均分，重复owner共享权重。每个含非空白字符的raw行质量约1。
- `lexical_mask`：每个raw词元是否包含 `isalnum` 字符，不代表真值。`encoder_response_lexical_mask` 是同义encoder位置标记。
- `target`：本侧独立原 `[start,end)` 字符范围、原text和作者建议角色。`target_raw_token_indices` / `target_encoder_token_indices` 按目标内非空白字符取交集；对应 `*_lexical_indices` 只考虑目标内 `isalnum` 字符。
- `nfc_proof`：组合重音修复的实际规范化证明；原文本/target坐标没有改动。

目标外全部未知，不能把未出现在目标索引中的位置当正确或错误。原侧与建议修复侧也不代表整篇答案错误/正确。标点目标的非空白索引可非空，但lexical索引为空；即使BPE同时含范围外字母也不扩大监督。

`input_index.jsonl` 与 `token_byte_offsets.npy` 同序。用 `byte_offset` 和 `byte_length` 从二进制打开的JSONL读取一条，并核 `record_bytes_sha256`。`paired_index.jsonl` 保留全部10040对，嵌入两侧索引和原source/group/answer身份，显式列 `eligible_for_future_local_pair_training` 和不可用原因。

只有显式使用eligible子集时，`eligible_pair_base_weight` 才适用：材料组均权→组内原答均权→同原答可用pair均权；两版本仍是一个pair单位。候选原建议权重同时保留，不能混用两个分母。这里没有生成正式训练入口、损失函数或全答标签。

读取前须验证 `complete.json`→`manifest.json`→各文件哈希。`CPU_OUTPUT_CHECK.json` 是单独的全20080输入文本/切片、索引、映射质量与局部位置核验。任何映射或超长异常均留在完整记录与 `exceptions.jsonl`，不会通过丢弃原记录凑满“成功”数量。
