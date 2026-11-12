# GHOST 四维特征抽取入口：CPU 已就绪

入口：`src/run_ghost_feature_extraction_v1.py`。目前只完成 CPU 准备，没有运行或排队 GPU，没有训练分类器。

固定输入为原 3839 答：3680 fit / 159 calibration，共 708506 raw BPE，含标点；最长输入 1232。原文本、词元顺序、坐标和后续金标分母均不改。四维 float32 特征纯负载约 10.81 MiB，另存原坐标和身份；没有按标签筛词元。

命令均需从工作区运行：

```powershell
prelab/.venv/Scripts/python.exe prelab/benchmark_ragtruth_qa/src/run_ghost_feature_extraction_v1.py check
# 以下两个命令尚未运行，须另获 GPU 调度：
prelab/.venv/Scripts/python.exe prelab/benchmark_ragtruth_qa/src/run_ghost_feature_extraction_v1.py gpu-smoke
prelab/.venv/Scripts/python.exe prelab/benchmark_ragtruth_qa/src/run_ghost_feature_extraction_v1.py extract
```

`check` 验证冻结输入、源码、软件和本地模型全部资产哈希，不加载权重。`gpu-smoke` 使用现有 NF4 loader，固定第 0、985、196 条（首条 fit、最短 fit、最长完整输入），核有限值、重复 exact、预读因果性，以及完整前向保存 hidden_states 的独立公式参考。参考和生产都固定每 16 个 predictor states 调用 lm_head；不将不同 BF16 GEMM 批次几何的相等性作为前提。FP32 公式参考容差预先固定为绝对 8e−6、相对 0，不据结果调整。

`extract` 必须已有同签名 smoke 通过记录。它逐答原子保存 `features/00000.npz` 与 JSON；NPZ 自带签名、输入记录哈希和身份。再次执行时只跳过经身份、原坐标、dtype、有限性、文件哈希验证的条目；NPZ 已提交而 JSON 尚未提交的中断可恢复。损坏或签名不同的旧文件保留并报错。

NPZ 主键为 `ghost_features[N,4]`，另有 `token_ids`、`answer_token_positions`、`predictor_positions`、`token_start/end`、`token_start_raw/end_raw`。有序 `feature_manifest.json` 与 `features_complete.json` 仅在重新打开并验证全部 3839 份 NPZ 后写入。它们目前不存在。

CPU 实证：修正后 `check` session 56552 实际 exit0；32 层随机小 Llama 的 runner dense 参考最大误差 4.77e−7、重复 exact；缺少 smoke 时在 loader 之前停止；保存、缺 JSON 恢复、错误签名拒绝均通过。证据是 `CPU_CHECK.json`、`CPU_RUNNER_SELFCHECK.json` 和上游 `INDEPENDENT_REVIEW.json`。

首次 check 的失败原样保留在 `FAILURE_check_*.json` 与 `PREPARATION_FAILURE_01/`：为了哈希导入 bitsandbytes 触发了该包的 CUDA 初始化，末尾 CPU 断言拦截；没有加载模型或前向。新入口改为通过安装包元数据定位源码并直接哈希，修正后的独立进程确认未初始化 CUDA。原冻结 `ghost_geometry.py`、准备脚本及输入未改；本次只修改新入口，未改变公式或数值门槛。

这批特征是对冻结回答的重建前向，不能称其他生成器的历史原生轨迹。提取耗时和真实 GPU 数值仍未测，不能据 CPU 通过推断检测精度。
