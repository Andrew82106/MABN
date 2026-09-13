# 人工字符 span → 原回答词元与窗口金标

版本：`ragtruth-qa-gold-v1`。本规则在首次标签导出前写定。只读取已冻结的634个fit、159个calibration回答及 `data/feature_preparation/plans.jsonl`；不读取官方测试、隔离来源的回答或标签，不使用预测、模型特征数值或探针状态。

## 不变的输入

- 只使用既有 `quality=good` 的793个完整原回答。来源/关联组/分区、原回答文字、原人工span和坐标均保持不变。
- 人工span保留四种类型以及 `implicit_true`、`due_to_null`、`meta`；目标是对资料的忠实性，不以世界中可能真实为理由去掉无依据片段。
- 字符坐标为原回答上的Python Unicode字符索引，左闭右开。保留所有原start/end/text；不规范化文字、不移边界、不扩到整词或整句。
- 只使用feature preparation已经固定的 `original.answer_token_ids`、`answer_token_positions`、raw offsets和clipped offsets。不重新分词，不读取no-context信号来决定标签。

## 词元标签

对每个原始回答BPE词元，取它与原回答相交的字符范围。raw offset可能从-1开始，因为模板分隔空格与答案首词同词元；保留该词元，并使用已固定的clipped offset映射答案字符。重复或相交offset（例如byte fallback）不去重、不压缩BPE索引。

- `lexical_mask[j]=1`：该词元覆盖的原回答字符中，至少有一个字符满足Python `str.isalnum()`。
- `risk_mask[j]=1`：至少一个**同时属于该词元、任一官方span且满足isalnum**的字符存在。仅标点或空白的span相交不使词元成为风险词元。
- 保存每个风险字符在原回答中的实际区间，以及每条官方span对应的风险词元索引。没有词元预测参与任何映射。

## 4个raw BPE的滑动窗口

按保留的原回答BPE顺序，窗口长度4、步长1；不先去掉标点。N≥4时起点为0至N−4，末尾不另加不足4的窗口；0<N<4时只保留一个长度N的短窗口。

窗口中没有lexical token则不进入主窗口评测，另存排除记录。否则，只要其中一个lexical token为risk token，窗口金标即1；全部不是风险词元则为0。窗口字符显示只依据实际词元offset，不能扩成整句话。

## 整答标签与边界情况

`answer_risk = int(len(original_labels)>0)`。全部793个quality-good原回答均可评。没有人工label的回答，包括可能的正常拒答，整答为0；其包含文字的窗口也按上述规则作为负例保留。**不额外用规则或模型识别拒答后排除窗口。** 这一点与R16中拒答不参与定位评测的口径不同，跨数据F1差异不能直接归因于算法优化。

若span为零长度、没有isalnum字符、未被词元覆盖，或整答risk=1却没有risk token，必须逐项记录并报告；不能悄悄补标、移动边界、删除回答或把整答改成0。区间越界或原text与坐标不符视为源完整性失败，记录后停止导出，由负责人处理，不能自行修复。标点/空白覆盖差异单独记录，不等于有无事实风险的语义裁决。

## 文件与核查

- `tokens_fit/calibration.jsonl`：每回答一行，含身份、原回答/原span、词元IDs/位置/offset、lexical/risk数组及风险字符区间。
- `windows_k4_fit/calibration.jsonl`：每个可评窗口一行，含身份、原BPE起止索引、词元/字符范围、风险词元与字符区间、金标。无lexical窗口另存 `windows_excluded_*`。
- `answers_fit/calibration.jsonl`：每回答一行，含原span、整答金标、词元/窗口计数与异常标志。
- `gold_edge_cases.json`：上述边界情况；零例也报告。
- `gold_selfcheck.json`：从完整输入offset独立重推回答token选择，检查未漏首token、每个官方span的可评字符全部覆盖，并以独立字符遍历核对词元和窗口标签。另检查标点窗口、短窗口、零长度span与重叠offset的明确小例子；不混入正式数据。
- `gold_manifest.json`：协议、脚本、开发源、feature preparation及导出文件哈希、实际分母和排除数。只写金标产物，不写任何模型/探针状态。

本基准的“金标”指保持官方人工span后得到的确定性标签映射；字符边界由原数据提供，本脚本不宣称重新完成语义人工审核。正式模型的token特征必须逐行对齐这些固定IDs、位置与offset后才能评分。
