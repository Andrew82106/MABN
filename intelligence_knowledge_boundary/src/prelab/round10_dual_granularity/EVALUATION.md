# Round10 评测接口与固定口径

本轮检验的主方法固定为 `lb_full`，即 Lookback + 8 维表面特征 + 16 维新增白盒特征。主要比较为同粒度的 `lb_full − lb`；白盒的额外价值看 `lb_full − lb_surface`，不能测试后更换主方法。

## 方法与训练

五组特征：`lb`、`lb_surface`、`lb_whitebox`、`lb_full`、`lb_old_binding`。每组分别训练回答项头和细标词元头，共 10 个 LR。每个头只比较 `C={0.1,1}`，L2/liblinear，最多 2000 次迭代，种子 20260912。

- 回答项头：按 Round7 的原适配方式，先对项范围内全部原始 BPE 特征取均值，包含标点，再训练 LR。不是平均词元概率，也不是 Round9 的项标签广播训练。
- 词元头：在原始位置的特征上训练精确风险片段标签，仅使用沿用 Round9 规则的可计分词元。数字拆成多个 BPE 时全部保留，标点/空白不计，功能词保留。已知旧数据两处 `[1]` 编号被原解析器计入的例外不追溯更改。
- 训练先使每题组等权，再在组内按条件和项平分，词元均分所在项权重，避免长回答主导。标准化仅用训练集的无标签基础权重。类别权重仅由训练数据计算，再将每题组最终损失权重归一为相等。不同特征组在相同粒度中使用完全相同样本与权重。
- 两类头分别按各自粒度的验证风险 micro F1 选择 C 与阈值；阈值并列先精确率，再更高阈值；C 并列取较小 C。没有测试选参、MLP 或随机种子搜索。
- 五个项头另生成整项广播定位对照，沿用项头原阈值，不再按词元标签挑阈值。广播使用完成后的回答，属于事后对照，不能称实时定位。

## 两个主分母

回答级一条输入对应一个短回答。已裁定断言使用原风险标签；经过明确复核、没有风险断言的纯拒答作为风险 0。未决和缺失排除并公开覆盖；这检测无依据陈述，不评价任务是否完成。额外报告与 Round7 接近的 asserted-only 子集，以及纯拒答误报。

词元级仍只使用原标为 resolved 的事实断言项，正常项的非风险词元必须保留。非风险词元标签只代表“未落入标注风险位置”，不表示每个词都已单独证实。拒答与未决文本仅报告报警率，不能自动赋全零词元金标。

两级分别报告 P/R/F1、混淆计数、AUROC/AP、缺预测数量、完整/部分资料及属性分层。词元级另报风险回答子集、句内排序、精确错误 span 覆盖、冗余报警和正常回答误报。没有容忍位移，没有调整分数到前后词元。

置信区间固定按 60 个新题组抽样 2000 次，种子 20260912。一个题组的两种条件和全部词元必须一起抽；两个粒度使用相同抽样。报告每方法 P/R/F1 区间和预定配对差。区间以固定训练结果与助手标注为条件，未包含重新训练和标注不确定性，也未做多重比较校正。

## 文件接口与冻结

- `data/inputs.jsonl`：Round9 train/validation 320 行精确复用 + 新 test 120 行；每行一个问题、一个回答项。旧 Round9 test 不进入该文件。
- `data/generated.jsonl` 与 `data/generation_records/{row_id}.json` 必须相同，保留原始回答、token IDs、字符 offsets、项 text/start/end。
- `data/features/{row_id}.npz`：`lookback_features[R,784]`、`binding_features[R,32]`，以及 token_ids/token_start/token_end。JSON 的轴名为 binding_feature_names。
- `data/soft_features/{row_id}.npz`：`new_features[R,16]`、`surface_features[R,8]` 和相同的 token 对齐字段。JSON 保存 new_feature_names/surface_feature_names、source_generation_sha256、arrays_sha256 和 runner 的 stage_signature。
- `data/annotations_{split}.jsonl`：原风险、stance、localization_status、实际风险 span、精确原文和生成文件 SHA；复用的开发标签不改字节。
- `data/safe_refusals_{split}.json`：逐条明确审查的 safe_refusal_item_ids。`data/question_label_policy.json` 保存 status=reviewed_frozen 与 reviewed_safe_refusal_files_sha256 三划分哈希。fit 不解析 test 拒答名单。
- `data/annotation_freeze.json`：status=frozen、canonical_spans_sha256 三划分哈希、question_label_policy_sha256。
- `data/freeze.json` 在新 test 生成前保存输入、reuse_manifest、protocol、PLAN、指南、所有本轮 src 源码的 files_sha256，以及 external_source_sha256。外部依赖固定包括 Round7 model7/attention7/lumina7、Round8 evaluate8、Round9 binding9/run9/annotation9。

fit 验证完整冻结关系，但 test 标签和 test 拒答名单只按原始字节计算哈希，不解析内容。它只读 train/validation 标签，保存模型、选择记录、训练权重、验证结果和 `results/freeze10.json`。正式 test 必须再次核对所有输入、标签、源码、权重和阈值；开始标记写下后不能重新选参，完成后拒绝重跑。

## 运行方式

从仓库根目录依次执行，正式数据与所有标签未冻结前不要执行 fit：

```powershell
prelab\.venv\Scripts\python.exe prelab\round10_dual_granularity\src\evaluate10.py fit
prelab\.venv\Scripts\python.exe prelab\round10_dual_granularity\src\evaluate10.py test
prelab\.venv\Scripts\python.exe prelab\round10_dual_granularity\src\evaluate10.py report
```

`report` 只整理既有正式结果表，不重新计算分数。正式结果为 metrics_test.json、answer_scores_test.jsonl、token_scores_test.jsonl；保存每项和每个原始词元，不挑选漂亮案例。复查策略没有修改。

合成验证可独立运行：

```powershell
prelab\.venv\Scripts\python.exe -m unittest discover -s prelab\round10_dual_granularity\tests -p test_evaluate10.py -v
```

资源：只使用 CPU，限制 4 线程；本轮 20 次候选 LR 拟合，开发特征规模较小。实际时间以 freeze10 的 fit_wall_seconds 为准，测试加载与纯评分时间分开保存，不能将纯 CPU 评分当成生成和 GPU 提取的端到端延迟。

开发集已经历前轮分析；新题组仍是单模型、英文短回答和人工构造资料缺失场景。助手盲标与复核不等于研究者人工金标；结果不能直接推广为真实联网情报系统性能，也不能承诺达到特定分数。
