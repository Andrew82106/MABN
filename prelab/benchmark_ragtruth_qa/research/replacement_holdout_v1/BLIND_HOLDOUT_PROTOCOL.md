# OSINT-PEH 盲测封存与一次性评分协议

状态：设计稿。执行前需由数据保管人与建模负责人共同冻结版本；本文件本身不授权下载、解封或评分。

## A. 角色隔离

1. **建模方**：只能访问现有 fit/cal、冻结模型和最终的无标签 test 输入包；不能访问 gold、上游自动标签、`answerable` 字段或人工裁决记录。
2. **数据保管方**：采集资料、构造配对条件、运行固定生成器、组织人工标注并保存 gold；不得参加方法选择。
3. **评分方**：在隔离环境中接收冻结预测和 gold，仅运行冻结 scorer；第一次正式评分后把该测试记为 `spent=true`。

若现实中只能由同一台机器执行，gold 必须保存在工作区之外、由不同账户/密钥控制；建模进程只获得无标签包。仅把文件命名为 `sealed` 不构成隔离。

## B. 四个不可逆门

### Gate 0：来源规则冻结

在采集正文前固定：来源白名单与许可规则、发布日期窗口、200 个事件组目标、六领域配额、五主体问题模板、`5/5` 与 `3/5` 配对方式、干扰文档匹配规则、去重阈值、采样 seed。

### Gate 1：方法冻结

在任何 test 问题、资料或回答交给建模方前生成 `development_freeze.json`，至少绑定：

```json
{
  "freeze_id": "...",
  "timestamp_utc": "...",
  "method_names": ["..."],
  "model_checkpoint_sha256": "...",
  "tokenizer_revision_sha256": "...",
  "chat_template_sha256": "...",
  "feature_code_tree_sha256": "...",
  "model_artifact_sha256": {"method": "..."},
  "window_threshold": {"method": 0.0},
  "answer_threshold": {"method": 0.0},
  "window_rule": "4 raw BPE, stride 1",
  "answer_rule": "max over every eligible window",
  "primary_metrics": ["window_f1", "window_auroc", "window_ap", "answer_f1", "answer_auroc", "answer_ap"],
  "baseline_set": ["..."],
  "candidate_set_closed": true
}
```

该文件写入后只允许因哈希不一致或程序无法运行而宣布测试失败，不能修改候选或阈值后继续使用同一 test。

### Gate 2：输入与 gold 双封存

数据保管方生成两个物理分离的包：

- `test_inputs.tar.zst`：仅含 `group_id`、`condition_id`、问题、按真实顺序提供的资料、模型回答、词元 offset 和必要的生成 provenance。
- `test_gold.age`：含字符 span、类型、三位标注、裁决、回答标签、complete/partial 元数据和源事实答案。

公开清单 `blind_manifest.public.json` 只放哈希与计数：

```json
{
  "holdout_id": "osint-peh-v1",
  "source_rule_sha256": "...",
  "source_snapshot_merkle_root": "...",
  "input_package_sha256": "...",
  "encrypted_gold_sha256": "...",
  "groups": 200,
  "answers": 400,
  "conditions": {"complete": 200, "partial": 200},
  "selection_seed_commitment": "sha256(seed || salt)",
  "source_overlap_audit_sha256": "...",
  "license_ledger_sha256": "...",
  "gold_visible_to_modeling_team": false
}
```

每一行另有 `row_commitment = SHA256(holdout_id || group_id || condition_id || question_hash || documents_hash || response_hash)`；公开 manifest 只给承诺值，不给 gold。

### Gate 3：预测冻结和一次性评分

建模方对无标签包运行所有已经冻结的方法，输出每个合格窗口一次且仅一次：

```json
{"row_commitment":"...","method":"...","window_index":0,"risk_score":0.0}
```

预测包须先通过无 gold 的结构检查：方法集合完整、row/window 无缺失无重复、分数有限且方向统一为“越大风险越高”、每个回答的 answer score 可由窗口 max 精确重建。随后计算 `predictions_sha256` 并关闭写权限。

评分方只接受同时绑定 `development_freeze_sha256`、`blind_manifest_sha256` 和 `predictions_sha256` 的一次请求。评分完成写出：

```json
{
  "score_receipt_id": "...",
  "spent": true,
  "development_freeze_sha256": "...",
  "blind_manifest_sha256": "...",
  "gold_sha256": "...",
  "predictions_sha256": "...",
  "scorer_sha256": "...",
  "metrics": {"...": "..."},
  "bootstrap": {"groups": 200, "draws": 5000, "seed": 20261001},
  "scored_at_utc": "..."
}
```

第一次正式 receipt 生成后，scorer 拒绝相同 `holdout_id` 的第二份预测。若要修改方法，必须建立新的独立 holdout。

## C. 无标签来源隔离审计

隔离审计只读候选输入和现有 fit/cal 输入，不读 gold。整组拒绝规则在 Gate 0 固定：

1. 任一规范化问题或资料块 SHA-256 完全相同。
2. 规范化字符 5-gram MinHash 估计 Jaccard `>= 0.70`。
3. 目标事实的规范化 `(主体, 关系, 时间)` 与 fit/cal 人工或规则索引一致。
4. URL canonical form、官方公告编号、CVE/监管编号或事件唯一 ID 相同。
5. 独立审核员认为属于同一事件、同一主体同一事实或改写版本。

审计程序只向建模方返回保留/拒绝计数、规则命中数和报告哈希；不返回候选正文。所有被拒组保留在保管方日志中，不能换成看过回答表现后挑出的样本。

## D. 标注与 gold 质量门

- 每条回答两人独立标注，第三人裁决；三人均签署版本与时间记录。
- 先冻结原始回答 SHA，再标注；标注过程中不得编辑回答。
- 自动检查 span 边界、substring、Unicode NFC、重叠合并和空 span。
- 报告字符级两两 IoU、风险回答一致率、类型一致率；这些只用于说明标签质量，不用于删难例。
- 随机 10% 由第四人盲审；发现系统性规则问题时整套 gold 作版本升级，旧 hash 永久保留且在首次评分前完成。首次评分后不得改 gold 追分；确有标注勘误时同时报告原版和勘误版，主结论仍以原冻结版为准。

## E. 4-BPE 转换的防泄漏实现边界

转换分两步：

1. `build_geometry_no_gold` 只从回答和冻结 tokenizer 生成 raw token、字符 offset、4-BPE window 和 eligible mask，并先落盘、哈希。
2. `attach_gold_and_score` 只在隔离评分环境读 span，把风险字符投到已冻结 geometry；不得重新分词、删窗或改回答。

窗口含任一风险词元即为正；整答标签与整答分数都分别取全部合格窗口标签/分数的最大值。主报告固定输出窗口级与整答级 F1、AUROC、AP。

测试 gold 不能被 feature extractor、模型 adapter、threshold loader 或预测结构检查器 import。代码审计应扫描测试执行图中的 `labels`、`gold`、`answerable`、`golden_answer` 访问，并以操作系统权限做最终阻断。

## F. 失败处理

- 缺少任何冻结方法的预测：整次正式评分不开始；不得只比较成功方法。
- tokenizer/模型/code hash 不匹配：宣布运行无效，修复后仍使用同一输入仅限 gold 从未解密、也未产生指标的情况。
- 首次指标已经产生：测试即消耗；之后所有调试只准在 fit/cal 或新建数据上进行。
- 来源许可无法确认：该组在生成前整组剔除；不能评分后再剔除。
- 低正例率或高拒答率：如实报告，并用预先定义的任务完成度指标解释；不能重新采样。

## G. 建议执行顺序

1. 将旧 150 条标成 exposed diagnostic，仅保留复核用途。
2. 完成方法、baseline、阈值和 scorer 的最终哈希冻结。
3. 由独立保管方在冻结日之后采集 200 个 OSINT 事件组并完成许可/去重审计。
4. 构造相同问题的 `complete/partial` 两种资料，固定顺序后用目标模型一次生成并保存真实白盒轨迹。
5. 三人字符区间标注与裁决，生成双封存包和 Merkle/文件哈希。
6. 所有方法一次性生成预测、结构审计、关闭写权限。
7. 隔离 scorer 只运行一次，输出聚合指标和按事件组共同 bootstrap 的置信区间。
8. RAGognize 的公开迁移结果另表报告；不得用它反向修改已评分的正式模型。
