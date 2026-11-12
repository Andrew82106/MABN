# FAVA NFC字符映射检查

CPU检查实际exit0。仅检查真实失败答`fava_train_2892`、9个合成/普通文本和8条既有FAVA映射；未重跑全量分词，未改原文、标签、旧QA方法或v1产物。

本机tokenizers 0.21.4中，`NormalizedString('s\u030c').nfc()`输出`š`，但单个规范化字符的`.original`切片仅返回`s`。因此不能直接把该切片视为完整原字符覆盖。NFC规范是规范分解后再合成，见[官方API](https://huggingface.co/docs/tokenizers/api/normalizers#tokenizers.normalizers.NFC)；这里的缺口和切片行为由实际本地运行确认。

新helper只接受实际normalizer为单独NFC。对缺失mark，必须证明：有已覆盖starter；连续组合字符经HF与Python NFC一致合成一个字符；NFD等价；删除缺失mark会改变结果；HF源锚点正是该starter。随后仅为该mark增加starter已有的encoder token所有权。原字符平均及重复token分摊规则保持。其他缺口明确失败；完整覆盖时直接走旧函数。

真实样例的830个encoder token保持：原字符287（U+030C）映给763，原字符290（U+0301）映给764。所有非空白raw token映射质量为1；普通文本和旧8条映射与旧函数、存档均exact。缺普通字母、不能合成的重音、无starter重音和非NFC normalizer均拒绝。

调用：

```python
from fava_nfc_character_map import character_map
charmap, proof = character_map(
    text, raw_offsets, begin, end,
    normalizer=bert.backend_tokenizer.normalizer,
    return_diagnostics=True,
)
```

v2调用方应保存`proof`；原`begin/end`继续表示HF原offset，`charmap`另含已证明的mark归属。只用于辅助FAVA。没有宣称任意Unicode缺口都可修复。

首轮检查脚本误用了JSON字符串hash核对原UTF-8文本hash，已改为既有`build_gold.digest_text`；记录在`CHECK_FAILURE_01.json`。该修正未改helper、数据或容差。
