# Round9 运行说明（未执行正式评分）

工作目录为仓库根目录。实际 CLI 的 `--help` 已验证：只有 `fit` 和 `test`，没有 `report` 子命令；当前也没有 `report9.py`。最终报告由根任务读取正式产物后撰写。

```powershell
.\prelab\.venv\Scripts\python.exe prelab\round9_evidence_binding\src\evaluate9.py fit
.\prelab\.venv\Scripts\python.exe prelab\round9_evidence_binding\src\evaluate9.py test
```

两条顺序单独运行。fit 前必须有全部生成、三种特征目录、三套已裁决 canonical annotations 和 annotation_freeze.json。fit 只解析 train/validation 标签，但会核对全部文件（含 test 标签）的字节哈希。成功写 results/freeze9.json、frozen_models.pkl、selection.json、training_weights.json、validation_metrics.json。test 验证这些冻结物后才解析 test 标签，输出 metrics_test.json、token_scores_test.jsonl、error_tokens_test.jsonl、test_complete9.json。不要删除冻结标记来重跑或重新调参。

## 时间及内存：工作量估计，未做计时试跑

本机 i5-14600KF，14 核 / 20 逻辑处理器、约 64 GiB RAM；代码将 PyTorch 与数值库限制为 4 线程，全程 CPU。固定工作量为 10 个 LR 拟合、6 个 MLP 拟合（最多各 60 epoch）、12 个 ReDeEP 验证候选。若 240 条训练回答各有约 20–60 个可用词元，拟合按数分钟至约 20 分钟预留，进程内存约 2–5 GiB；这是资源安排估算，不是已测性能。若大量输出达到 256 词元上限，可能到几十分钟，内存约 8–12 GiB。实际运行时间取决于有效词元数、LR 收敛、MLP 提前停止及并行 GPU 任务的 CPU 占用。

缓存每个词元约 24 KB 浮点特征，另有训练矩阵、缩放副本、求解器和 PyTorch 张量。fit_wall_seconds 计的是 fit_models 调用（含其训练/验证特征加载），不含前后的全文件哈希、元数据及标签解析、保存与最终验证报告；不能标为整个命令端到端延迟。test 的 load_seconds / cpu_score_seconds 也不含整个命令全部哈希和 bootstrap。若需要整段墙钟，根任务可用命令外部计时，单列口径。

## 400 条回答的盲标与复核

```powershell
.\prelab\.venv\Scripts\python.exe prelab\round9_evidence_binding\src\annotation9.py packet --start 0 --count 5
```

start/count 是输入行号，提供 --split 时则是该划分筛选后的行号；不是题组号。0–99、100–199、200–299、300–399 可分工，但必须以冻结 inputs 实际行序明确任务范围。尚未生成的行仅显示 NOT_YET_GENERATED，可之后补看。

根在最终冻结前已给 packet 增加 FULL_RESPONSE；它现在显示当前问题、资料、完整 response 和原始 item，不含检测分数。每位标注者必须检查 item 外前言/额外输出。优先按当前实际可见资料独立判断；参考只帮助理解目标，不能把另一条件或参考当可见证据。

通过 Python 导入 annotation9.add，把每项显式决定写入各自独立 decisions 文件；该函数不是 CLI 子命令。每项至少 item_id、stance、evidence_relation、risk、rationale，risk=1 还须显式 spans 原文。add 保存 exact text/start/end/生成文件 SHA；重复引用须 occurrence（从 0 开始），重复 item、旧 hash、文本改变、重叠 span 会拒绝。分批追加可行，不需要一次写 100 项。

独立复核应先独立给决定，保留单独 review 文件；对分歧先裁决，再得到每项唯一 canonical 决定。不能把初标和复核两份含同 item 的文件直接交给 serialize，它会拒绝重复。

```powershell
.\prelab\.venv\Scripts\python.exe prelab\round9_evidence_binding\src\annotation9.py serialize --files data/annotation_decisions/part0.json data/annotation_decisions/part1.json data/annotation_decisions/part2.json data/annotation_decisions/part3.json --output data/annotations_all.jsonl
```

上面文件名为运行示例，须换成实际裁决文件。serialize 要求生成已全 400 行、决定覆盖全部项且互不重复，因此只有最后汇总时才运行。随后根按输出 split 确定性拆成 annotations_train/validation/test.jsonl，保留初标、复核、裁决记录及字节哈希，写 annotation_freeze.json 后才能 fit。serializer 不会自动完成复核裁决或拆分，也不会自动生成最终 report。

本次仅查看源码、协议、CLI 帮助、硬件及生成数量元数据（当时 0 条）；未拟合、未读取检测分数、未修改冻结源码。
