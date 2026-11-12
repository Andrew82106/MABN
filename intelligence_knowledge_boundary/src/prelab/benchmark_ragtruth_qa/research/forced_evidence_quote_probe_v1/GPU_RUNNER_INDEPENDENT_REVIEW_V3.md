# GPU runner V3 独立复审

**结论：PASS。** 冻结 runner 可进入 8 条 GPU smoke。`full_extract_allowed=true` 只表示静态代码已获批；当前仍缺少独立 `GPU_SMOKE_INDEPENDENT_REVIEW.json`，因此完整提取门保持关闭。

## 冻结对象

| 对象 | SHA-256 |
|---|---|
| GPU runner | `a142d00ef1601aff6af203cfaf60644a11c37f108c8c4043f641b51f8e01e0cb` |
| CPU selfcheck | `4f6636e78c3f14264c3c17fd991c1188aaab4fbe3c9666c012e8ebbc88aa0773` |
| PROTOCOL | `78e5558b61592238b0979c7c60aea23c79b8c7d3181e88113d3972b4d056f7aa` |
| PLAN | `6d439ee257dde65d2ef2c9974cd3dd716f4f0d06890cfb7df9c8af6da562680c` |
| TOKEN_OFFSET_REPAIR | `24296dfad9450384fc683efdc155cdc1b73bd82e1425f168495d8ff3842b577d` |

## 核验结果

- 唯一样本输入是哈希锁定的 `label_free_inputs.jsonl`。3,776 条记录、256 个回答均严格符合 allowlist；动态文件审计未打开 fit、calibration、test 或 gold，也没有评分路径。
- 3,776 条提示与 P2 整串分词全部复算通过。提示长度为 255–799，P2 为 271–888；机械引文 digest 为 `7ee3fa403002e49eb353668c84d6bda48103216961945f8dd382f136fced0570`。标点及跨边界 token 按最大字符重叠归属，平局直接失败。
- 固定 smoke 为索引 `[2826, 994, 1533, 1695, 1299, 1143, 3352, 3752]`，长度 `[255, 354, 385, 435, 508, 568, 634, 799]`，包含最短与最长样本；smoke 不写正式 claim 记录。
- P3 仅用 KV cache 取得生成 ID 和 token 统计；hidden/attention 全部来自完整序列 `use_cache=False`、`past_key_values=None` 的重放。P2 使用整串 teacher forcing。每条 claim 都核对 cached 与 replay 的 argmax 和三类统计。
- P1/P2/P3 维度为 21/549/549，P2/P3 严格拼接 `21+11+5+256+256`。注意力重建使用官方 RoPE 与 `repeat_kv`；独立 CPU Llama eager oracle 的最大误差为 0，hidden 完全不变。
- 词元偏移精确镜像冻结的 Replace→ByteFallback→Fuse→Strip。独立复算 189,542 个序列，包括全 32,000 单 ID、全 65,536 byte pair、40,000 个 byte 三/四元组、52,000 个混合序列及 6 个对抗样例；文本 oracle 与偏移均通过。允许非规范 ID、多字节共享区间、非法 byte 逐个替换、Strip 零宽区间，只有终止 EOS 使用 `(-1,-1)`。
- malformed quote 仍标为无效；嵌套标签和首个 close 后的文本不会进入 relation context。原子 NPZ→metadata→commit、resume 哈希/行签名、残缺记录 fail closed 与 WDDM 独占门均通过。
- 模型只从本地加载，`trust_remote_code=false`。runner 不改 baseline，也不写分数。

## 门禁

Canonical review 已批准 `gpu-smoke`。`extract` 除本报告外，还必须读取与当前 runner 及 `GPU_SMOKE.json` 哈希绑定的独立 smoke PASS 报告；缺少该报告时，runner 在加载模型或使用 GPU 前失败。

本复审未启 GPU、未加载预训练模型、未读取 gold/cal/test、未运行评分，也未修改 runner 或 baseline。
