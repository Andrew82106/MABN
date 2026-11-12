# V2 Evaluator 独立静态审查

**结论：PASS。** 被审 evaluator 为 `src/evaluate_forced_evidence_quote_probe_v2.py`，SHA-256 `a80e1ac7a107d96e32dd55e47e4c8535c5e7a189e491335f519acb604883ac8f`；CPU synthetic selftest 收据 SHA-256 为 `c1379a9412a54d43d639af532fdcb95ce3a007324d50d06bbee1ba7309d97c78`。

## 核对结论

1. 独立 AST 对比确认 15 个 V1 科学函数全部不变，覆盖 fold、真实 fit geometry、折内回答归一权重、折内 scaler/类别权重、固定分类器、严格 5 外折 × 4 内折、claim→4-BPE window 映射、阈值、answer max、pooled AP/F1、诊断和推进门。`SALT` 仍为 `forced_evidence_quote_probe_v1`；固定 C、维度、行数与组数也未变。
2. V2 的非科学差异限于独立 V2 结果/冻结路径、V2 runner 与 `NUMERICAL_PROTOCOL.md` 的 SHA 锁、runner-result 冻结字段，以及用于证明 V1 parity 的 AST 自检。runner SHA 为 `eb4a51353ba4edaff2cff877f576ff206f1fca1d6bd820928fb0db4acebfcca2`；numerical protocol SHA 为 `ad0a2198ba69e75c78852eab103e8a829b6f9fb21389f322d5879f016e989df1`。
3. `evaluate_real` 的前四步依次为 `assert_runtime_contract`、`assert_v1_scoring_parity`、`load_frozen_features`、`load_real_gold_geometry`。冻结文件缺失、schema/status/taint/source/hash/shape/dtype/finite/P1⊂P2 任一不合格，都会在任何 fit gold loader 前失败。
4. selftest 收据绑定当前 evaluator、V1 protocol/plan、V2 numerical protocol、V2 runner 和唯一 label-free input；其中 `real_gold_open_count=0`、`fit_gold_opened=false`、`calibration_read=false`、`official_test_read=false`、`GPU_used=false`、`real_evaluation_run=false`。

本审查只使用源码、AST、静态编译、文件哈希和既有 synthetic receipt；未导入或运行 evaluator，未打开真实 frozen feature arrays、fit gold、calibration 或 official test，未使用 GPU/CUDA，也未修改 evaluator、finalizer、V1 或 baseline。

**正式评分尚不允许。** 本 PASS 只解除 evaluator 静态审查门；仍须完成独立 finalizer review，并由哈希绑定的 GPU smoke、full extraction 与 feature freeze 全链路通过后，才能启动 evaluator 的正式 fit-only scoring。
