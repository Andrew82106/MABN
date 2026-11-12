# ReDeEP 与 LUMINA 同域适配

本轮保留方法公式，统一使用父目录的原问题、原检索资料、原 Llama 回答和分组划分。提取器不读测试标签，不重新生成回答，也不重写聊天模板。正式论文和旧实验产物未修改。

## 已冻结配置

`CONFIG.json` 绑定源代码、数据、候选注意力头和 `plans.jsonl` 的 SHA256。计划中的原输入与父目录 geometry 完全一致；所有 1842 条原输入已经使用当前 tokenizer 再编码核对。每条原回答的第一个及最后一个 token 均保留，信号读在目标 token 之前的预测位置 `answer_position - 1`。

| 项目 | ReDeEP(Token) | LUMINA |
|---|---|---|
| 输入数 | fit1292、validation271、test279 | validation271、test279；fit仅作随机资料库 |
| 骨干 | Llama2-7B-Chat 完整 FP16，CPU offload | 同一基础权重，NF4 双重量化 / BF16 |
| 参数化检测器训练 | 无；仅fit选择头/层和MinMax | 无 |
| 输出 | 每个原BPE token的风险，及原生全答均值 | 每个原BPE token的风险，及原生全答均值 |
| 最终共同窗口/阈值 | 由 `audit/evaluate_extensions.py` 完成 | 同左 |

模型来源：`NousResearch/Llama-2-7b-chat-hf@351844e75ed0bcbbe3f10671b3c808d2b83894ee`。每次实际加载前按已有下载清单核权重 SHA；运行环境和实际设备另存 `RUNTIME.json`。

## ReDeEP 身份与训练粒度

采用已审计的**论文公式迁移**：资料区域的 top-10% 注意力目标对应末层隐藏态与当前预测态做余弦得到 ECS；每层 FFN 前后词表分布的标准 JSD 得到 PKS。保留作者固定 32 个候选复制头，Khead=1、Klayer=10、alpha=1、beta=0.2。

读取 `data/fit/sequences.jsonl` 中 141282 个**原生 token 标签**，逐 token 分别计算 `Pearson(ECS, 1-risk)`、`Pearson(PKS, risk)` 排序。没有先聚成4BPE窗口，没有读取 validation/test 标签。选中信号在所有 fit token 上求和，然后分别拟合 MinMax；风险为 `PKS_scaled - 0.2*ECS_scaled`。新样本沿用这两个边界，不裁剪，也不重新拟合。已以非恒定fit数据及越界新值验证，与 sklearn MinMaxScaler 数值相同；遇到恒定候选或归一化范围时停止检查，不悄悄采用其他定义。

官方 `token_level_reg.py` 的 `construct_dataframe` 确实是每 token 一行及相应 token 标签。它计算 Pearson，但实际返回并按 AUROC 排序；原论文写相关性。官方 detector 的 ECS 区域、PKS 散度也与论文不同，因此本轮只以预先固定的论文身份作正式基线。原代码公式的 ECS/PKS 随提取另存，作为可审计原始量，**不按测试成绩挑选代码/论文分支**。官方旧代码的 token 标签存在 predictor/target 位移问题；本轮使用共同数据 target token 的风险标签，不复制该错位。

最初 CPU 草稿沿旧本地实现用4BPE窗口进行相关性排序，经核对后在正式GPU提取前纠正为原生token。原草稿配置保存在 `CONFIG_PRE_TOKEN_RANK_CORRECTION.json`。这只影响ReDeEP拟合；LUMINA公式/输入未变，但为避免文件哈希身份混淆，早期两条LUMINA smoke保存于 `smoke_pre_token_rank_correction/`，正式550条重新提取，不复用该缓存。

## LUMINA 随机资料与公式

使用作者提交 `c43ff41d872b05f659dcb3ad3a6dd78226954319` 的逐词元公式：`0.5*IPR - 0.5*MMD_squared`。保留 top100 未重新归一化概率、余弦核、全部输出层及最后一层再次 norm 的作者代码约定。采用既有流式实现，减少显存，不改参数或数学公式。

随机资料库只来自本轮fit且按材料去重。每个目标材料以 `SHA(seed,target_material,donor_material)` 固定排序，剔除同来源组、相同材料、标题交集或完全相同正文；若总长超过骨干上下文上限，则按同一排序选择下一条。该筛选只看无标签输入。只替换原提示里的检索资料块，问题、指令与答案保持原样。550条随机输入重新编码后，答案 token IDs、原始/裁剪字符偏移与原输入逐项相同；两个前缀长度可以不同，各自按自己的 `P+t-1` 对齐。

供体不保证绝对语义无关，故称固定随机资料替换。与作者接口的差别是保留共同提示和答案边界，不采用其重新套模板、额外空格和字符截断。

## 已完成的CPU核查

- 原输入全1842条重新分词，与共同geometry的token/位置/偏移/资料区一致，最后token覆盖回答末尾。
- LUMINA 550条供体选取标签无关、fit-only，答案分词与偏移完全不变。
- ReDeEP tiny-Llama上候选头重算与dense attention完全一致；FFN前后投影与dense参考一致。
- LUMINA tiny-Llama上逐层IPR/MMD与独立dense公式一致；相同资料MMD为0；后续回答改变不影响此前词元信号；重复前向一致。
- MinMax非恒定情况与sklearn核对通过。

检查记录：`CPU_CHECK.json`、`REDEEP_CPU_CHECK.json`、`LUMINA_CPU_CHECK.json`。GPU实际结果由根任务串行调度，本文件不把CPU通过称作GPU已完成。

## 命令

从仓库根目录执行，使用原通用 prelab 环境（sklearn1.6.1 / transformers4.57.1），不要使用RAGognizer专属环境：

```powershell
$python = 'D:\Projects\Multi_Agent_Graph_Analysis\prelab\.venv\Scripts\python.exe'
$runner = 'intelligence_knowledge_boundary/src/prelab/benchmark_ragtruth_qa/research/same_domain_20260928_v1/baseline_extension_20260928/whitebox/run_whitebox.py'
& $python $runner prepare
& $python $runner cpu-check
& $python $runner extract --method lumina
& $python $runner score --method lumina
& $python $runner extract --method redeep --limit 2
& $python $runner extract --method redeep
& $python $runner score --method redeep
```

`extract` 才加载GPU。每个进程结束后再开始下一个。`--limit 2` 为可选smoke，不写全量完成标记；之后去掉limit，校验已有缓存并恢复其余记录。每条NPZ与JSON receipt绑定输入/配置/文件哈希，损坏或不完整缓存停止检查，不静默跳过。

`score` 生成 `{method}/validation_predictions.jsonl` 与 `test_predictions.jsonl`。每行包括 `response_id`、完整原BPE `token_scores`、`native_answer_score`、`response_token_offsets` 和特征哈希。raw score 不强行转成概率，越高越可疑。共同评测代理在验证集比较mean4/max4并冻结阈值；本提取器不根据测试成绩调整方法。

## 原始资源

- [ReDeEP官方代码](https://github.com/Jeryi-Sun/ReDEeP-ICLR)，本地固定提交 `4d081915b8fb4430fda65c411da61540cc73cc57`；论文身份见 `../../../../research/redeep_formal_baseline_v1/SOURCE_AUDIT.json`（实际位于benchmark根下research目录）。
- [LUMINA官方代码](https://github.com/deeplearning-wisc/LUMINA)，上文固定提交；复用 `src/lumina_qa_signals.py` 与 `round7_evidence_grounding/src/lumina7.py` 的已审计公式。
