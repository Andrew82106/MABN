# 缓存契约

目录 `results/lumina_qa_features_v1/features/{index:05d}.npz`，顺序完全继承3839行准备输入（3680 fit、159 calibration）；同名JSON为可恢复侧文件。

`lumina_features`: float32 `[N,7]`，列严格依次为：

1. `ipr`
2. `mmd`
3. `lumina`（原float32 `0.5*ipr-0.5*mmd`）
4. `original_answer_probability`
5. `original_max_probability`
6. `random_answer_probability`
7. `random_max_probability`

最后4列仅保存诊断原量，不改变基线；下游计分只按其冻结协议取前3列。

int64 `[N]` 数组：`token_ids`、`original_answer_positions`、`random_answer_positions`、`original_predictor_positions`、`random_predictor_positions`、`token_start`、`token_end`、`token_start_raw`、`token_end_raw`。前两类位置是各完整输入的绝对token位置；predictor=answer_position−1。字符offset相对同一原回答；raw允许负首端点、重复byte-fallback范围，不压缩标点。

字符串scalar：`signature_sha256`、`record_sha256`（q.digest完整输入行）、`response_id`、`record_index`。NPZ内置身份支持跨写入中断恢复。JSON包含该身份、NPZ SHA、两路输入hash、原计划hash、原/donor材料hash及来源组、分区、各长度和七列min/max；无gold。

`feature_manifest.json`：`status=complete`、`version`、`signature_sha256`、`records=3839`、`raw_answer_tokens=708506`、`feature_names`、`answer_order`、按原顺序的`entries`（file/npz_sha256及上述侧文件元数据）、`all_records_validated=true`、`labels_used=false`、`trained=false`、`test_opened=false`。

`features_complete.json`：`status=complete`、相同records/raw_answer_tokens/signature/feature_names、`all_records_validated=true`、`preparation_complete_sha256`、`no_test=true`、`trained=false`；`files_sha256`相对名绑定`feature_manifest.json`、`signature.json`、`protocol.json`、`CPU_CHECK.json`、`GPU_SMOKE.json`。只有完整3839经重新核验才写；存在progress不等于完成。下游不得读部分记录开始拟合/计分。
