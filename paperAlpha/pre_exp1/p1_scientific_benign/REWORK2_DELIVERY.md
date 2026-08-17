# P1 二次返工交付（历史说明，供主 Agent 查阅）

本文件是历史二次返工交付说明，不是当前/latest/authoritative 交付；本文件与
`REWORK3_DELIVERY.md` 均为历史说明；当前唯一说明为 `REWORK4_DELIVERY.md`，
当前派生审计为 `acceptance4-20260801T120000000000Z`。
它记录既有 P1 pilot 的工程、验证、证据解释和文档返工，不是新实验。历史边界固定为：
`p1_go=false`、不得 P2、既有真实 pilot 未重跑。
实施 Agent 不宣布 P1 Go 或最终 No-Go；交付状态为**待主 Agent 验收**。

禁止项均已遵守：未调用 Ollama、远程 API、讯飞 API 或任何 provider；未使用网络服务；未
读取、散列、复制或输出 `paperAlpha/.env`；未运行真实 pilot、warm-up、第二次 P1 或 P2；
未修改 `doc/`、P0、P1 Phase A、qualification、shared fixture/schema 源文件。

## 1. 固定环境与依赖

```text
Conda: multi_agent_graph
Interpreter: D:\anaconda\envs\multi_agent_graph\python.exe
Python: 3.11.15
pytest: 8.4.2
installed distribution: paperalpha-pre-exp1==0.2.0
新增依赖：无
```

开始和交付复核均显示上述解释器及 Python 版本；未执行依赖安装。

## 2. A–E 修复证据

### A. 共享 schema provenance 精确集合

`core.py` 现在只把以下四个文件写入 `shared_schema_files`，并由这四项的相对路径—SHA
映射计算 `shared_schema_tree_hash`。下表路径均以
`D:\Projects\Multi_Agent_Graph_Analysis\paperAlpha\` 为绝对根；模板和 Python 合同源码仍在各自诚实命名的其他 map 中：

| 精确相对路径 | SHA-256 |
|---|---|
| `data/shared/p1_benign/schemas/event.schema.json` | `282fd3c0f190970fde61af6f42959772f9bcc26ade9d47199208c064bb7ad83d` |
| `data/shared/p1_benign/schemas/manifest.schema.json` | `4698f563639952a3492dc14fd739c748a66f3f10f455b2a3b5bec85748a4b786` |
| `data/shared/p1_benign/schemas/outcome.schema.json` | `e8853d16d1295eec82fe55d8db20bafa52665ceb9ea14a1ff8351ed2063fc8b3` |
| `data/shared/p1_benign/schemas/replay.schema.json` | `21356cf86c875978fe4cb46933c525d000f72d5b11005379b7d04322d467418e` |

四项唯一集合的 tree hash：
`ee9c47d5012061fb738255b7e6efc5f7c6a7a7118648cd69403231b6580ca927`。

科学层测试从预先写死的四条相对路径校验集合、逐文件 hash、tree hash，并分别验证漏项、
替换项和额外项均 fail-closed。

### B. 合法语义篡改的生产 fresh replay

`test_semantic_tamper_uses_fresh_production_replay_and_is_rejected` 使用 `tmp_path` 生成有效
小型工件，翻转 `task_success`，重算 episode_finished payload hash、event ID、events/outcomes
artifact hash；随后**实际调用生产 `replay_run(..., write_output=True)`**，由生产函数写出新的
`replay.json`。测试断言该文件等于调用返回值且 `passed=false`，之后 validator 返回结构化
`passed=false`、`artifact_hashes_match=true`，并报告
`outcome contradicts authoritative sink/event recomputation`。测试没有手工写入或修补 replay JSON，
也没有触碰真实 pilot。

### C. 覆盖 in-flight provider 调用的硬 deadline

`runner.py` 新增 deadline provider wrapper：每次调用前计算剩余预算，把剩余值传入 provider
timeout；调用返回后再次检查时钟，跨过边界的结果被拒绝，不能被当成成功，也不会进入下一
episode。定向假 provider 测试让调用开始时仍在预算内、自然返回前将可控时钟推进到
`3601 s`；证据为 `generate_calls=1`、结构化 `failure_stage=deadline`、
`accepted_as_success=false`、`partial_outcomes=1`，runner 抛出 `TimeoutError` 并停止后续 episode。

### D. preflight drift 结构化失败记录

preflight 现在在 provider 调用前捕获 metadata drift，写入临时工程 pilot 的
`preflight_<RUN_ID>.json` 和失败 manifest；记录 run/context、失败类型和原因、
`provider_calls=0`、`network_calls=0`、`accepted_as_success=false`。定向测试通过公开 pilot CLI
（模拟 drift、未调用真实 Ollama）验证：退出码 `1`、输出无 traceback、结构化记录存在且零
provider/network 调用；记录只写入 `tmp_path`，不写入真实 pilot。

### E. 当前/历史文档状态唯一化

- `DELIVERY.md` 已明确为“历史首次交付/返工前基线”，保留旧的 `13 passed` 与 `68/80` 事实但
  不再标为 current/latest/authoritative。
- `REWORK_DELIVERY.md` 已明确为第一次返工历史交付。
- 本文件是历史二次返工说明，不再是当前交付；后续三次返工与 acceptance3 也仅为历史证据。
  当前 `paperAlpha/README.md`、`data/README.md`、本层 README、`IMPLEMENTATION.md` 和
  `data/registry.json` 均指向 `REWORK4_DELIVERY.md` 与 acceptance4，且保持
  `p1_go=false`、既有 pilot 未重跑和不得 P2。
- `acceptance2-20260731T230556448674Z` 是历史审计 ID；历史 deterministic dry ID 为
  `P1-BENIGN-PILOT-DRY-20260731T150355012369Z`；后续历史 dry 已更新为
  `P1-BENIGN-PILOT-DRY-20260731T155623039469Z`；三次返工 acceptance3 为历史审计，
  当前审计为 `acceptance4-20260801T120000000000Z`。

## 3. 离线测试与退出码

以下命令均从 `D:\Projects\Multi_Agent_Graph_Analysis\paperAlpha` 执行，固定环境下全部退出码 0：

```powershell
conda run --no-capture-output -n multi_agent_graph python -B -m pytest pre_exp1\p1_scientific_benign\tests -p no:cacheprovider  # 26 passed
conda run --no-capture-output -n multi_agent_graph python -B -m pytest pre_exp1\p1_benign\tests -p no:cacheprovider             # 224 passed
conda run --no-capture-output -n multi_agent_graph python -B -m pytest pre_exp1\p1_model_qualification\tests -p no:cacheprovider  # 32 passed
conda run --no-capture-output -n multi_agent_graph python -B -m pytest pre_exp1\tests -p no:cacheprovider                   # 57 passed
```

科学层共 `26 passed`，已包含 A–D 定向测试。历史只读函数复验结果：

| 数据角色 | validation | replay | replay live/network |
|---|---:|---:|---:|
| P1 Phase A `P1-BENIGN-DRY-20260731T094841301886Z` | PASS | PASS，2/2 | 0/0 |
| qwen qualification `P1-BENIGN-QUAL-QWEN3-20260731T121840716022Z` | PASS | PASS，3/3 | 0/0 |
| ministral qualification `P1-BENIGN-QUAL-MINISTRAL3-20260731T121957481982Z` | PASS | PASS，3/3 | 0/0 |
| P0 `P0-SMOKE-20260731T053210335011Z` | PASS | PASS，4/4 | 0/0 |

## 4. 公开 CLI 只读复验

### 既有真实 pilot

run：`P1-BENIGN-PILOT-QWEN3-20260731T132033188154Z`

- `validate_scientific_run.py --run-id ...`：退出码 `1`，保留历史 validation 失败事实；结构化
  错误包括标签不一致、179/180 structured outputs、provenance drift 和历史 manifest 缺少
  正确 shared schema provenance；无 traceback、无写回。
- `replay_scientific_run.py --run-id ...`：退出码 `0`，`20/20 matched`，live/network/provider
  calls `0/0/0`；无写回。

### 历史 engineering dry（本文件记录时）

run：`P1-BENIGN-PILOT-DRY-20260731T150355012369Z`

- validation：退出码 `0`，所有检查通过，四 schema provenance 正确；
- replay：退出码 `0`，`20/20 matched`，transcript model actions `180`，live/network calls `0/0`；
- 输出位于 `engineering_dry/`，不是第二个真实 P1 batch。

## 5. 二次派生审计与证据 hash

历史时点唯一新增目录：

```text
D:\Projects\Multi_Agent_Graph_Analysis\paperAlpha\data\pre_exp1\p1_scientific_benign\pilot\acceptance_repair\acceptance2-20260731T230556448674Z\
```

manifest、validation、replay、run report、gate report 和 source hash 清单的 SHA-256：

```text
manifest.json       3958d8635104aae623f96592a9e4409cae20f3d46ea1327c5d04ac95cf2f27c3
validation.json      02b582c287a98f5e6ec42d6a094cdd005f1156978257e92248de414339469417
replay.json          b18334bc32af558725cb7101b958744ad7e4f07fa494b8590203d5d9fa590ee8
run_report.md        85a981ef2867e0a32ea343094e9eda66c2656b98d6de558a9a4ae14b82605387
p1_gate_report.md    f5b1bbac540a58837979fc8e85bcadce065c7574673cf40ceb73defad8667361
source_hashes.json   caf206f4c0c95e3747d34f7993a589246bc7bd20ea4bfa0102ca8f24a7a3708c
```

manifest 明确记录：`derived_from_run_id` 为既有 pilot、`no_provider_calls=true`、
`network_calls=0`、`p1_go=false`、候选类别为 `候选返工（实现交付未验收）`，以及四个 schema
的精确 provenance。二次返工历史时点的修复工具 provenance 与当时
`capture_provenance()` 比较为相等（仅作历史记录）。

前两次审计目录未被覆盖；开始和交付复核的 manifest SHA-256 保持不变：

```text
acceptance-20260731T1415/manifest.json  05c0002051785d9c9e02e397b1709bd7667d89d04251bf9a5279d7e7021cf976
acceptance-20260731T1422/manifest.json  3d2832e415858d6ac98f545203bbef055c7dc01c758f32b675d08d4729d11099
```

## 6. 原始 pilot 与 P0 保护

既有真实 pilot 核心证据开始/交付 SHA-256 完全相同：

```text
events    8de3e520fe86b3b1be74104d2aa1762ef8ce7b8775be5033f9cc3c51f050d38e
manifest  66c48af2d881f298bfa438cf7c3e8f07dde29b6a8a329382cc4d9525875f7ab1
outcomes  b8887f9a2cb175e2ad885b77f315f30e710ada3321e251a31647c633e7d6e2ee
```

交付端文件清单/size/mtime 快照如下；开始与交付 SHA 对比均为 unchanged：

| 文件 | size(bytes) | mtime UTC |
|---|---:|---|
| `raw/events_P1-BENIGN-PILOT-QWEN3-20260731T132033188154Z.jsonl` | 1072786 | `2026-07-31T13:27:48.0168635Z` |
| `manifests/manifest_P1-BENIGN-PILOT-QWEN3-20260731T132033188154Z.json` | 11934 | `2026-07-31T13:27:48.5160994Z` |
| `processed/outcomes_P1-BENIGN-PILOT-QWEN3-20260731T132033188154Z.jsonl` | 28964 | `2026-07-31T13:27:48.0222967Z` |

二次审计 `source_hashes.json` 同时保存原 pilot replay、validation、run、gate、preflight
和 validation report 的 hash。P0 raw 八项 SHA-256 复核为全部匹配；没有修改、删除或移动
任何原始 pilot/P0 证据。

## 7. 当前 pilot 的历史事实与研究边界

派生审计按权威 sink/event 逐字段复算：`vendor_id=19`、`registration_status=19`、
`risk_level=19`、`recommendation=17`，合计 **74/80**。原 pilot 事实仍为：

- `ProviderParseError=1`；
- structured outputs `179/180`；
- report published once `19/20`；
- task success `17/20`；
- transcript replay `20/20`；
- 原始真实模型调用事实 `180`，而本轮返工及 replay/audit 没有 provider/network 调用。

这些事实没有被改写。`p1_go=false` 仍是当前工程状态标记；最终研究判定留给主 Agent，
本轮不宣布 Go/最终 No-Go，不开始 P2。

## 8. 一致性、无副作用与待验收事项

- 文档检索确认旧 `13 passed`、`68/80` 只出现在明确标注为历史首次交付或历史原始报告的位置；
  历史时点说明、README、data registry 曾指向 `REWORK3_DELIVERY.md` 和三次审计；本文件仅为历史证据，
  当前已迁移至 `REWORK4_DELIVERY.md` 和 acceptance4。
- 相对链接检查覆盖 `REWORK2_DELIVERY.md`、历史 `DELIVERY.md`/`REWORK_DELIVERY.md` 和
  历史 README/registry 路径；registry 已登记当前 repair ID、当前 dry ID、`p1_go=false`、
  `p2_allowed=false` 和 `pilot_rerun_allowed=false`。
- `paperAlpha` 全树未发现 `__pycache__`、`.pytest_cache`、`.pyc`、venv 或 egg-info；未生成
  P2/后续阶段 artifact。没有访问 `.env`。

请主 Agent 独立复验：四 schema 是否为精确集合、fresh replay 是否确实来自生产
`replay_run()`、in-flight deadline 是否拒绝越界结果、preflight 失败记录是否可诊断、当前文档
是否唯一一致，以及所有受保护 hash 是否不变。

完成本文件后实施 Agent 停止，等待主 Agent 验收。
