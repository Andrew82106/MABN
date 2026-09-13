原有 MiniCheck 开发数据的第 22 层补缓存已完成 CPU 准备，尚未运行 GPU。

- 原 fit 634 答 + calibration 159 答，共 793；official test 不参与。
- 直接读取原缓存的选中文档，共 8,852 个 claim 输入；不重新比较其他文档。
- 2,763,434 个有效完整输入 tokens，float32 状态约 11.32 GB，另有少量坐标。
- 每条旧 claim 的 token ID/字符坐标必须精确一致；因 selected-only 批次与旧 all-doc 批次可能不同，原最后层状态和 logits 采用 2e-4 绝对误差上限，支持分 2e-5。旧分数和状态保持原值，新结果仅用于补缓存及数值核查。

存储路径为 `encoder22_features/{response_id}.npz` 和同名 JSON。字段同 `../minicheck` 的新增答案缓存，另外保存 `original_selection_batch_padding_length`；metadata 和最终 manifest 都保留 partition。原 634 缓存或 calibration 文件均不会被覆盖。

`preparation_freeze.json` SHA256：`bdd67c423f2f29ea2f6dc8c3ad5999ab6779086c41ee77ea6df8cafc6cbe10b4`。

待主代理明确授权 GPU 后执行：

```powershell
prelab/.venv/Scripts/python.exe prelab/benchmark_ragtruth_qa/fit_expansion/run_minicheck_backfill.py infer
```

完成后，将本目录 `encoder22_feature_manifest.json` 与 `../minicheck/encoder22_feature_manifest.json` 合并读取，覆盖全部 3,680 fit + 159 calibration 答案。该入口不训练模型、不选择阈值。
