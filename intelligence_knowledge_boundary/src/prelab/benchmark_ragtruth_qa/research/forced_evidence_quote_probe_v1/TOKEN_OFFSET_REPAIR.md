# P3 生成词元到最终字符串偏移的修复方案

## 结论

GPU smoke 的首条失败不是模型输出异常，而是原断言错误：`decode(ids)` 后再
`encode(text)` **不保证得到原 IDs**。SentencePiece 的编码不是解码的逆函数；多个
词元序列可以解码成同一个字符串。

最小反例（本地冻结的 Llama-2 tokenizer）：

```text
IDs/tokens: [29871, 12199] = ["▁", "hello"]
decode:     "hello"
re-encode:  [22172] = ["▁hello"]
```

因此不能用重新编码取得原生成词元的字符偏移，也不能用逐前缀 decode 的长度差永久
分配偏移。正确做法是：保持原生成 IDs，按冻结 tokenizer 的 decoder 状态机做一次
**带词元来源的最终态解码**。

## 本地 decoder 与可用接口

环境：`transformers==4.51.3`、`tokenizers==0.21.4`。冻结模型为
`NousResearch/Llama-2-7b-chat-hf@351844e75ed0bcbbe3f10671b3c808d2b83894ee`。

- `tokenizer.json` SHA-256：
  `f7b50bcf6d6672eade5e43514d48e9c1e4e63a56aef7b14acdaca94ce93436f7`
- 该版本没有 `tokenizers.DecodeStream`，backend 也没有 `decode_stream`。
- 本地未安装 Python `sentencepiece`；不需要为本修复安装它。原生 SentencePiece 解码
  即使可用也只给最终文本，不能恢复任意非规范 IDs 的原边界。

`tokenizer.backend_tokenizer.to_str()` 给出的 decoder 链是：

```text
Replace("▁" -> " ")
ByteFallback
Fuse
Strip(content=" ", start=1, stop=0)
```

修复函数必须先核对这条链。decoder 配置变化时应直接失败，不能默默套用此算法。

## 确定性映射算法

输入是生成时保存的原始 IDs，而不是由文本重新编码出来的 IDs。

1. 若生成因 EOS 停止，只从最终解码中移除最后一个 EOS；它的偏移记为
   `(-1, -1)`。内部 special token 不删除，因为当前解码使用
   `skip_special_tokens=False`。
2. 用 `convert_ids_to_tokens` 得到原 token piece，并对每个 piece 执行
   `▁ -> ASCII space`。
3. 普通 piece 直接追加；它拥有追加字符串的整个半开区间 `[start, end)`。
4. 对每个**最大连续 byte-token run**（`<0xHH>`）：
   - 整段 bytes 能严格按 UTF-8 解码时，按 Unicode 字符切分；组成同一个字符的 2–4
     个 byte token 共享该字符区间。例如 `E4 B8 AD -> 中`，三个 token 都映射到
     `[0, 1)`。
   - 整段不能严格解码时，冻结的 `ByteFallback` 会为 run 中**每个 byte**输出一个
     `U+FFFD`；第 k 个 byte token 映射到第 k 个替换字符。不能用 Python 的
     `errors="replace"`，因为它对不完整多字节序列可能只给一个替换字符。
5. `Fuse` 后，若总字符串以一个 ASCII space 开头，执行一次全局 Strip，并同步左移
   所有区间。完全被 Strip 删除的 token 记为 `(0, 0)`。
6. 最后用 `tokenizer.decode(active_ids, skip_special_tokens=False,
   clean_up_tokenization_spaces=False)` 作字符串 oracle。只核对最终字符串；不重新编码。

普通 token 与 byte run 相邻时没有跨界合并：遇到普通 token 就结束 byte run。例：

```text
[<0xE4>, ▁hello, <0xB8>] -> "� hello�"
offsets                       [(0,1), (1,7), (7,8)]
```

## 可直接替换的函数

```python
import json
import re
import numpy as np

_BYTE_PIECE = re.compile(r"<0x([0-9A-F]{2})>\Z")
_EXPECTED_DECODER = {
    "type": "Sequence",
    "decoders": [
        {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
        {"type": "ByteFallback"},
        {"type": "Fuse"},
        {"type": "Strip", "content": " ", "start": 1, "stop": 0},
    ],
}


def _assert_frozen_llama_decoder(tokenizer):
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    assert backend.get("decoder") == _EXPECTED_DECODER, (
        "Unsupported tokenizer decoder; provenance mapper must be reviewed",
        backend.get("decoder"),
    )


def generated_token_offsets_exact(tokenizer, generated_ids, terminal_eos_id):
    """Decode original Llama IDs and return final-text provenance intervals.

    Intervals are half-open Python character offsets. They may overlap for
    multi-byte UTF-8 fallback tokens and may be zero-width after leading-space
    Strip. Only an explicitly removed terminal EOS receives (-1, -1).
    """
    _assert_frozen_llama_decoder(tokenizer)
    ids = list(map(int, generated_ids))
    active_count = len(ids)
    if terminal_eos_id is not None:
        assert ids and ids[-1] == int(terminal_eos_id)
        active_count -= 1
    active_ids = ids[:active_count]

    pieces = tokenizer.convert_ids_to_tokens(
        active_ids, skip_special_tokens=False)
    assert len(pieces) == len(active_ids)
    assert all(isinstance(piece, str) for piece in pieces)
    pieces = [piece.replace("▁", " ") for piece in pieces]

    offsets = [None] * len(active_ids)
    chunks = []
    cursor = 0
    index = 0
    while index < len(pieces):
        match = _BYTE_PIECE.fullmatch(pieces[index])
        if match is None:
            piece = pieces[index]
            chunks.append(piece)
            offsets[index] = (cursor, cursor + len(piece))
            cursor += len(piece)
            index += 1
            continue

        stop = index
        byte_values = []
        while stop < len(pieces):
            byte_match = _BYTE_PIECE.fullmatch(pieces[stop])
            if byte_match is None:
                break
            byte_values.append(int(byte_match.group(1), 16))
            stop += 1

        raw_bytes = bytes(byte_values)
        try:
            decoded_run = raw_bytes.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            # tokenizers::ByteFallback is all-or-nothing for a maximal run:
            # an invalid run emits one U+FFFD per input byte.
            decoded_run = "\N{REPLACEMENT CHARACTER}" * len(byte_values)
            for local in range(len(byte_values)):
                offsets[index + local] = (
                    cursor + local, cursor + local + 1)
        else:
            byte_cursor = 0
            for char_index, char in enumerate(decoded_run):
                width = len(char.encode("utf-8"))
                char_interval = (cursor + char_index,
                                 cursor + char_index + 1)
                for local_byte in range(byte_cursor, byte_cursor + width):
                    offsets[index + local_byte] = char_interval
                byte_cursor += width
            assert byte_cursor == len(byte_values)

        chunks.append(decoded_run)
        cursor += len(decoded_run)
        index = stop

    raw_text = "".join(chunks)
    stripped = 1 if raw_text.startswith(" ") else 0
    if stripped:
        adjusted = []
        for left, right in offsets:
            if right <= 1:
                adjusted.append((0, 0))
            else:
                adjusted.append((max(0, left - 1), right - 1))
        offsets = adjusted
    decoded = raw_text[stripped:]

    oracle = tokenizer.decode(
        active_ids, skip_special_tokens=False,
        clean_up_tokenization_spaces=False)
    assert decoded == oracle, (
        "Provenance decoder disagrees with frozen tokenizer", decoded, oracle)

    if terminal_eos_id is not None:
        offsets.append((-1, -1))
    assert len(offsets) == len(ids)
    assert all(pair is not None for pair in offsets)
    assert all(pair == (-1, -1) or
               0 <= pair[0] <= pair[1] <= len(decoded)
               for pair in offsets)
    return decoded, np.asarray(offsets, dtype=np.int32)
```

## CPU 验证结果

在冻结 tokenizer 上用固定随机种子 `20260912` 做了 189,542 个序列测试；每个序列
都要求建议算法生成的文本与 `tokenizer.decode` **逐字符完全一致**，并检查偏移长度、
边界和最终文本覆盖：

| 测试 | 数量 | 结果 |
|---|---:|---|
| 全部词表 ID 的单 token | 32,000 | PASS |
| 全部 256×256 byte-token 对 | 65,536 | PASS |
| 随机 byte-token 三元组、四元组 | 40,000 | PASS |
| 随机任意词表序列（长度 2/3/8/32，偏置边界 token） | 52,000 | PASS |
| 指定对抗样例 | 6 | PASS |

关键样例：

```text
[▁, hello]            -> "hello"      [(0,0), (0,5)]
[E4, B8, AD]          -> "中"          [(0,1), (0,1), (0,1)]
[61, FF, 62]          -> "���"         [(0,1), (1,2), (2,3)]
[E4, ▁hello, B8]      -> "� hello�"    [(0,1), (1,7), (7,8)]
[▁hello, terminalEOS] -> "hello"       [(0,5), (-1,-1)]
```

`E4`、`E4 B8`、`E4 B8 AD` 的逐前缀解码分别是 `"�"`、`"��"`、`"中"`。
这证明逐前缀长度差也不可靠；第三个 byte 到来时会重写前两个前缀字符。建议函数按
最终最大 byte run 一次处理，不存在 pending 状态遗漏。

## 必须同步修改的断言与失败条件

- 删除“最终文本重新编码后 IDs 必须相同”的断言。
- 允许普通 token 在 Strip 后得到 `(0,0)`。
- 允许多个 byte token 共享同一个正宽字符区间；不能要求 offset 严格递增或互不重叠。
- `generated_content_indices` 现有的 `end <= start` 跳过逻辑可保留；共享区间会让组成一个
  Unicode 字符的所有 byte token 一起进入内容集合，正好保留每个生成 token 的
  logprob/hidden 信息。
- 以下情况 fail closed：decoder 链不匹配、token ID 无法转成 piece、终止 EOS 契约
  不匹配、最终文本与 tokenizer oracle 不同、偏移越界。
- 修复后必须重新做 CPU selfcheck、独立静态审查和 GPU smoke；旧 smoke 失败文件应保留
  为协议审计记录，不能覆盖。

