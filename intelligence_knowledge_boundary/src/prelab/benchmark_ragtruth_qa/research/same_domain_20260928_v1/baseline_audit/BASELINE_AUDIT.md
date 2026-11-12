# 同域基线训练审计（2026-09-28）

## 当前结论

同域主比较可以统一为 RAGognize 官方 train 内、按材料组固定划分后的同一批 fit/validation/test。已发布 RAGognizer 检查点在该官方 train 上训练过，不能直接用于这次 validation/test。主比较必须从通用 Llama 基座、新 LoRA 和新检测头重新训练。旧公开检查点的外部测试表现保留为历史参照。

本目录已交付训练与标签不可见预测入口。RAGognizer最长两答GPU smoke已通过并退出，正式训练尚未启动：

| 入口 | 保持的方法结构 | 当前状态 |
|---|---|---|
| `ragognizer_same_domain.py` | Llama-2-7B + 七类投影的 r32/alpha16 LoRA + 原 4096→1024→1024→1 ReLU 中层头；响应 LM+BCE 联合损失 | CPU原头/梯度等价、全量映射及最长两答真实GPU训练step通过 |
| `lookback_same_domain.py` | 原 32层×32头、连续8原生BPE均值、默认 `LogisticRegression(max_iter=1000)` | CPU脚本已完成；消费上级 `whitebox_signals.py` 的缓存，无需第二次提取 |
| `predict_frozen_baselines.py` | 已选RAG/Lookback模型与已冻结阈值；只读inputs/geometry | 标签不可见CPU坐标检查通过validation271答、test279答；不打开sealed |

没有将 RAGognizer 降为固定表征上的 head-only 探针。NF4、双重量化与 BF16 是明确披露的精度适配，因此应称“原结构同域重训（NF4）”，不能称“已发布 BF16 检查点的逐值复现”。

## 数据与执行边界

只读取新目录 `data/fit` 和 `data/validation` 的 answers/sequences；脚本无 test 入口。最终 fit 为 1292答/617组，validation 为271答/133组，组交集为0。原始 train 的重复资料已由上级数据审计合并，采用新的共同 group_id。

RAGognizer 使用官方 Llama tokenizer/chat template，解析一层原 `[INST]` 包装后保留 system/user 内容；只监督回答原生 token。有效原 span 映射至原生 token；LM 保留回答结尾/EOS。统一评测再以字符交集映射至共同 BPE，取4窗均值和整答最大值。映射覆盖已在全部1563个 fit/validation 回答 CPU验证：137192/31283个合格共同窗口。

| 已准备原生输入 | fit | validation |
|---|---:|---:|
| 总输入token | 919487 | 206840 |
| 最长输入 | 1567 | 1423 |
| 输入长度P95 | 1164 | 1191 |
| 检测正token | 37597 | 8506 |
| 检测负token | 103683 | 23616 |

未截断、裁剪、按标签删样本。BCE正类权重只由fit计算，为 `(positive+negative)/positive`，沿用作者 balanced 分支。5个固定epoch，AdamW学习率4e-5、累积8、warmup0.1、cosine。预先固定：主checkpoint按共享no-hint验证`selection_key`择优，并同时保留验证联合loss最优checkpoint；并列取较早epoch。共同评测器统一报告 no-hint 与 all。

默认输入为已重建的 `ragognizer_prepared_v2`；no-hint明确取`not unanswerability_hint_setting`，fit/validation分别681/135答。旧prepared仅保留记录、不再默认读取。`PREPARED_V2_AUDIT.json`核验两版计数一致及全量坐标覆盖。

数据审计代理另行只读复核：1563条ID/group/partition/no_hint/answer_risk与统一数据一致；全部映射窗口标签与共同risk/lexical定义一致；3条fit和3条validation重新分词逐项复现。该复核未使用GPU或测试标签。

后续在任何正式基线训练前，预先增加统一池化对照：原mean4复现保留，同时由完全相同的token_scores计算lexical max4；两种都不改模型或训练。RAG每个epoch、Lookback每个C分别保存`VALIDATION_MEAN.json`和`VALIDATION_MAX.json`，按共同验证`selection_key`选择池化，并列优先mean。选中的pooling与阈值一起冻结，测试预测只执行已冻结选择，同时保留两种窗口分数便于复核。271条validation的两种聚合与共享评测器逐值一致，见`cpu_checks/POOLING_AUDIT.json`。

## 8GB资源判断

正式BF16全模型权重约13.82GB，本机不能完整驻留。旧推理使用CPU offload完成3839答约2.35小时，但它是无梯度推理，不能作为训练可运行的证据。

新训练保持79,953,920个LoRA参数和5,245,952个检测头参数。FP32权重、梯度与Adam两组状态共约1.27GiB；另加NF4基座、量化元数据、BF16冻结词嵌入/LM矩阵、激活与CUDA开销。启用非重入梯度检查点，并只对响应目标位置分块计算LM交叉熵，避免保存完整输入×词表logits。冻结词嵌入/LM矩阵恢复BF16，与原headed加载器行为一致；norm和可训练参数保持FP32。

这些措施不改变网络、目标或响应损失权重。原头前向、输入梯度、各层参数梯度与官方 `MLPHead` 在CPU逐值相同；分块LM交叉熵/梯度与整块计算数值一致。实际最长fit样本1567/1496token的前向分别1.17/0.91秒、反向2.40/2.22秒；Adam更新及全部七类×32层LoRA_B与MLP联合梯度通过。峰值allocated为5.60GiB、reserved为6.18GiB。详见`ragognizer_gpu_smoke_v1/GPU_SMOKE.json`。进程已退出、GPU已释放，没有启动正式训练。

按这两答token线性估计，单epoch纯训练33.5分钟，五epoch约2.8小时，加入验证/保存暂估3–3.5小时。该估计来自最长样本，仍需正式训练核验。

可选效率实现`--microbatch-size 2`已加入，默认仍为1。它以每答LM/BCE分别归一再平均，将1×累积8改为2×累积4，保持有效8答及optimizer/scheduler步数。32层tiny Llama+七类LoRA+3层MLP的CPU核验中，两答长度13/9、响应长度9/4、右侧padding；逐答与同批loss相同，453个参数梯度tensor最大差1.49e-8。见`cpu_checks/BATCH_CPU_SELFCHECK.json`。尚未对该可选模式运行GPU；仅在最长两答同批真实Adam smoke能容纳且更快时采用。

额外可选`--pair-within-update`仅在原来每8答更新组内部按长度配对，不跨组重排，也不修改下一epoch的shuffle状态。五epoch全部810组成员检查通过；相对未配对microbatch2，padding后的总token减少10.2%–11.7%，每轮仍是同一919487个真实token。见`cpu_checks/LENGTH_PAIRING_CHECK.json`。正式运行将保存每epoch原更新成员和执行顺序，是否采用仍等待GPU实测。

可用原专用环境：`tmp/venvs/ragognizer_cpu_audit/Scripts/python.exe`，已有torch2.5.1+cu121、transformers4.57.1、peft0.18.1、bitsandbytes0.49.2。通用prelab虚拟环境没有peft，不能直接启动RAG训练；Lookback可用通用环境。

GPU调度后命令（从工作区根目录运行；输出目录需为新目录）：

```powershell
& 'tmp/venvs/ragognizer_cpu_audit/Scripts/python.exe' -X utf8 'intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/baseline_audit/ragognizer_same_domain.py' smoke --out 'intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/baseline_audit/ragognizer_gpu_smoke_v1' --allow-gpu
```

smoke执行最长的两个fit回答和真实optimizer step，检查全部7类×32层LoRA_B及检测头梯度，并记录峰值显存。通过后同入口换为 `train` 和新的输出目录。训练不读取旧LoRA、旧检测头或测试标签。

Lookback等待共同signals完整后：

```powershell
& 'intelligence_knowledge_boundary/src/prelab/.venv/Scripts/python.exe' -X utf8 'intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/baseline_audit/lookback_same_domain.py'
```

全部fit/validation回答均至少8BPE，因此无需短答特殊处置。LR不加标准化、类权重或样本权重；只读本轮fit拟合，验证阈值交给共同评测器。默认C=1原基线与额外C∈{.01,.1,1,10}同结构验证选择对照分别报告，各档完整保留；tuned对照写入`tuned`目录。只增加solver预算处理未收敛：1000→3000→10000，不改变C或目标；仍未收敛则终止，不把结果发布为最终基线。

`await_lookback.py`已启动单次CPU流水线：每50秒等候`signals/COMPLETE_fit_validation.json`报告1563答且缓存全部存在，随后执行上述固定网格。状态保存在`lookback_pipeline/STATUS.json`，训练日志为`lookback_pipeline/training.log`，输出为`lookback_run_v2`；这不是GPU任务。

Lookback四档已完成，纯CPU训练/验证约105秒，C1/0.01/0.1/10分别531/140/395/439步收敛，均无需增加solver预算。默认C1选max，验证no-hint窗F1为0.62670；tuned选择C0.01+max，窗F1为0.65355。各C及mean/max均完整保留，未读取测试标签。

`await_ragognizer.py`是已获授权的一次性GPU等待流水线，每50秒只检查上级`GPU_READY_FOR_RAGOGNIZER.json`；必须确认`all_gpu_children_exited=true`才启动。随后仅跑一次microbatch2组内配对最长两答真实Adam smoke；同时比较去除加载后的耗时和独立前反向耗时，两者均快于原microbatch1且梯度/更新通过才选2，否则明确回退1。正式输出`ragognizer_run_v1`固定五epoch，再用冻结SELECTED导出`ragognizer_validation_predictions_v1`与`ragognizer_test_predictions_v1`。阶段、子进程PID、逐阶段日志、运行效率选择保存于`ragognizer_pipeline`；全子进程退出后才写根目录`RAG_PIPELINE_COMPLETE.json`。任何失败持久化traceback，不读取test标签或重新选测试阈值。

正式预测入口：`predict_frozen_baselines.py predict --method ragognizer|lookback --partition test --run-dir <已完成模型目录> --out <新输出目录>`。RAG使用专用环境并需GPU调度后的`--allow-gpu`；Lookback使用其训练时通用环境，tuned对照的run-dir指向`tuned`目录。该入口只读`inputs.jsonl`、`geometry.jsonl`、冻结模型与阈值，不调用评分器；输出`predictions.jsonl`和`PREDICTION_RECEIPT.json`。test输入/坐标预检覆盖279答、28977窗，没有读取测试标签。

只读test几何另确认：最短回答22BPE，279答全部满足Lookback原8BPE跨度要求，不存在短答缺覆盖。

## 其他基线需重新确定的内容

- **GHOST**：重新提取同域4维几何信号；只在共同fit拟合原750树RF。旧RAGTruth RF不能直接当同域重训。
- **ReDeEP**：公式与固定参数保持，但头/层选择、MinMax等用共同fit重新建立。不能继续使用旧RAGTruth的fit选择结果却称同train。
- **LUMINA**：没有监督分类头；按共同fit建立替换资料池并冻结配对/抽样。旧资料池会造成训练侧来源不一致。
- **RAGognizer发布检查点**：来源为RAGognize训练，RAGTruth为跨数据集评测；逐行launch manifest未公开。不能把“同一test”误写为“同一训练集/划分”。

## 官方代码缺口与实现身份

已先阅读旧审计，不重新训练/改动旧正式基线。锁定官方训练源码为commit `1d3e8fdd6de638dc2d06265829a4d9ced387e0ba`。

原`ft.py`不能直接用于本协议：初始化阶段即格式化test；`--ragtruth`分支的test formatter缺少return；非balanced分支打印未定义计数；LM头默认target为`lm_head`而训练样本字段为`labels`，存在联合LM损失未接入的代码缺口。新实现显式计算LM+BCE，保证论文联合目标；没有把这些运行时问题传播进主比较。作者原脚本五次调用单epoch trainer，新实现采用连续五epoch schedule；checkpoint精确启动参数未发布，此差异明确记录。

LoRA采用七类transformer投影。发布adapter_config列出`lm_head`，但发布权重的448个LoRA tensor仅覆盖32层七类投影，不含lm_head；本实现保持实际发布结构及作者量化分支对应目标，不凭配置字段新增LM LoRA。

主来源：[官方训练代码](https://github.com/F4biian/RAGognizer/blob/1d3e8fdd6de638dc2d06265829a4d9ced387e0ba/fine-tuning/ft.py)、[官方模型卡](https://huggingface.co/F4biian/RAGognizer-Llama-2-7b-chat-hf)、[官方数据](https://huggingface.co/datasets/F4biian/RAGognize)、[Lookback分类代码](https://github.com/voidism/Lookback-Lens/blob/e0a1fa3a898fbf6512af7be5567dea8ffe7a6620/step03_lookback_lens.py)、[PEFT量化训练指南](https://huggingface.co/docs/peft/en/developer_guides/quantization)。
